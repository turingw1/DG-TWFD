from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import PIL.Image
import torch
import tqdm


ROOT = Path(__file__).resolve().parents[2]
EDM_ROOT = ROOT / "refs" / "edm"
FID_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*$")

DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "cifar10": {
        "ctm_root": ROOT / "refs" / "ctm-cifar10",
        "checkpoint": Path(
            "/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/"
            "ctm_cifar10/ctm-cifar10/conditional/model043000.pt"
        ),
        "fid_ref": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz",
        "image_size": 32,
        "num_classes": 10,
        "class_cond": True,
        "batch": 500,
        "reference_steps": 17,
        "attention_type": None,
        "entropic_time_path": ROOT
        / "refs"
        / "entropic_time_schedulers"
        / "EDM"
        / "Schedules"
        / "RE_function_CIFAR10_uncond_vp_80_128_FREQ.pt",
    },
    "imagenet64": {
        "ctm_root": ROOT / "refs" / "ctm" / "code",
        "checkpoint": Path("/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/ctm/ctm_imagenet64_ema_0.999.pt"),
        "fid_ref": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz",
        "image_size": 64,
        "num_classes": 1000,
        "class_cond": True,
        "batch": 250,
        "reference_steps": 39,
        "attention_type": "legacy",
        "entropic_time_path": ROOT
        / "refs"
        / "entropic_time_schedulers"
        / "EDM"
        / "Schedules"
        / "Rescaled_entropic_time_64.pt",
    },
}

STRATEGIES = ["optimalsteps_ctm", "entropic_ctm", "piecewise_linear_ctm", "spline_warp_ctm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CTM-specific schedule/time-warp baselines. "
            "Unlike EDM schedule baselines, this runner directly controls each "
            "CTM exact transition G_theta(x_t, t, s)."
        )
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fid-ref", default=None)
    parser.add_argument("--sample-root", default="runs/ctm_schedule_warp_5k_20260428/samples")
    parser.add_argument("--eval-root", default="eval/ctm_schedule_warp_5k_20260428")
    parser.add_argument("--result-root", default="results/baselines/ctm_schedule_warp_5k_20260428")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES + ["uniform_ctm"])
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--fid-batch", type=int, default=512)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--class-idx", type=int, default=None)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--reference-steps", type=int, default=None)
    parser.add_argument("--search-batch", type=int, default=8)
    parser.add_argument("--search-seed-base", type=int, default=123000)
    parser.add_argument("--proxy-gamma", type=float, default=0.5)
    parser.add_argument("--entropic-time-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subdirs", action="store_true", default=True)
    parser.add_argument("--no-subdirs", action="store_false", dest="subdirs")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--dry-run-schedules", action="store_true")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _dataset_defaults(args: argparse.Namespace) -> dict[str, Any]:
    defaults = dict(DATASET_DEFAULTS[args.dataset])
    defaults["checkpoint"] = _resolve(args.checkpoint) if args.checkpoint else defaults["checkpoint"]
    defaults["fid_ref"] = args.fid_ref or defaults["fid_ref"]
    defaults["batch"] = args.batch or defaults["batch"]
    defaults["reference_steps"] = args.reference_steps or defaults["reference_steps"]
    defaults["entropic_time_path"] = _resolve(args.entropic_time_path) if args.entropic_time_path else defaults["entropic_time_path"]
    return defaults


def _image_path(root: Path, seed: int, *, subdirs: bool) -> Path:
    directory = root / f"{seed - seed % 1000:06d}" if subdirs else root
    return directory / f"{seed:06d}.png"


def _count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*.png") if item.is_file() and item.stat().st_size > 0)


def _missing_seeds(root: Path, *, seed_start: int, num_samples: int, subdirs: bool) -> list[int]:
    missing = []
    for seed in range(seed_start, seed_start + num_samples):
        path = _image_path(root, seed, subdirs=subdirs)
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(seed)
    return missing


def _parse_fid(stdout: str) -> float | None:
    for line in reversed(stdout.splitlines()):
        match = FID_RE.match(line)
        if match:
            return float(match.group(1))
    return None


def _run_and_tee(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    with log_path.open("w", encoding="utf-8") as handle:
        for line in proc.stdout:
            captured.append(line)
            handle.write(line)
            handle.flush()
            print(line, end="")
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, command, output="".join(captured))
    return "".join(captured)


def _karras_nodes(
    interval_count: int,
    *,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if interval_count <= 0:
        raise ValueError(f"interval_count must be positive, got {interval_count}")
    ramp = torch.linspace(0.0, 1.0, interval_count + 1, dtype=torch.float64, device=device)
    sigmas = (sigma_max ** (1.0 / rho) + ramp * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    sigmas[0] = float(sigma_max)
    sigmas[-1] = float(sigma_min)
    return sigmas


def _tensor_to_float_list(values: torch.Tensor | np.ndarray | Iterable[float]) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return [float(item) for item in np.asarray(list(values) if not isinstance(values, np.ndarray) else values).reshape(-1)]


def _load_ctm(dataset: str, defaults: dict[str, Any], *, device: torch.device, use_fp16: bool):
    ctm_root = Path(defaults["ctm_root"])
    if not ctm_root.exists():
        raise FileNotFoundError(ctm_root)
    if str(ctm_root) not in sys.path:
        sys.path.insert(0, str(ctm_root))

    from cm.script_util import (  # noqa: E402
        cm_train_defaults,
        create_model_and_diffusion,
        ctm_data_defaults,
        ctm_eval_defaults,
        ctm_loss_defaults,
        ctm_train_defaults,
        model_and_diffusion_defaults,
        train_defaults,
    )

    data_name = dataset
    merged: dict[str, Any] = {}
    for defaults_fn in [
        train_defaults,
        model_and_diffusion_defaults,
        cm_train_defaults,
        ctm_train_defaults,
        ctm_eval_defaults,
        ctm_loss_defaults,
        ctm_data_defaults,
    ]:
        merged.update(defaults_fn(data_name))
    merged.update(
        {
            "data_name": data_name,
            "model_path": str(defaults["checkpoint"]),
            "training_mode": "ctm",
            "class_cond": bool(defaults["class_cond"]),
            "num_classes": int(defaults["num_classes"]),
            "use_fp16": bool(use_fp16),
            "sigma_min": float(merged.get("sigma_min", 0.002)),
            "sigma_max": float(merged.get("sigma_max", 80.0)),
        }
    )
    if defaults.get("attention_type") is not None:
        merged["attention_type"] = defaults["attention_type"]

    args = argparse.Namespace(**merged)
    model, diffusion = create_model_and_diffusion(args)
    state = torch.load(defaults["checkpoint"], map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    if use_fp16 and hasattr(model, "convert_to_fp16"):
        model.convert_to_fp16()
    model.eval()
    return model, diffusion, args


def _batch_noise_and_labels(
    *,
    seeds: list[int],
    seed_base: int,
    dataset_defaults: dict[str, Any],
    device: torch.device,
    sigma_max: float,
    class_idx: int | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    channels = 3
    image_size = int(dataset_defaults["image_size"])
    latents = []
    labels = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed_base) + int(seed))
        latents.append(torch.randn((channels, image_size, image_size), generator=generator, device=device))
        if dataset_defaults["class_cond"]:
            if class_idx is None:
                labels.append(int(torch.randint(0, int(dataset_defaults["num_classes"]), (1,), generator=generator, device=device).item()))
            else:
                labels.append(int(class_idx))
    x = torch.stack(latents, dim=0).to(torch.float32) * float(sigma_max)
    model_kwargs: dict[str, torch.Tensor] = {}
    if dataset_defaults["class_cond"]:
        model_kwargs["y"] = torch.as_tensor(labels, dtype=torch.long, device=device)
    return x, model_kwargs


@torch.no_grad()
def _ctm_transition(
    diffusion,
    model: torch.nn.Module,
    x: torch.Tensor,
    sigma_from: float,
    sigma_to: float,
    model_kwargs: dict[str, torch.Tensor],
) -> torch.Tensor:
    batch = x.shape[0]
    t = torch.full((batch,), float(sigma_from), device=x.device, dtype=torch.float32)
    s = torch.full((batch,), float(sigma_to), device=x.device, dtype=torch.float32)
    _, g_theta = diffusion.get_denoised_and_G(model, x, t, s=s, ctm=True, teacher=False, **model_kwargs)
    return g_theta


@torch.no_grad()
def _sample_with_schedule(
    diffusion,
    model: torch.nn.Module,
    x: torch.Tensor,
    sigma_nodes: torch.Tensor,
    model_kwargs: dict[str, torch.Tensor],
    *,
    clip_output: bool = True,
) -> torch.Tensor:
    sigmas = [float(item) for item in sigma_nodes.detach().cpu().numpy().reshape(-1)]
    if len(sigmas) < 2:
        raise ValueError("CTM exact schedule must contain at least two nonzero sigma nodes")
    if any(sigmas[i] < sigmas[i + 1] for i in range(len(sigmas) - 1)):
        raise ValueError(f"CTM sigma nodes must be non-increasing, got {sigmas}")
    x_next = x
    for sigma_from, sigma_to in zip(sigmas[:-1], sigmas[1:]):
        x_next = _ctm_transition(diffusion, model, x_next, sigma_from, sigma_to, model_kwargs)
    return x_next.clamp(-1, 1) if clip_output else x_next


@torch.no_grad()
def _reference_pack(
    diffusion,
    model: torch.nn.Module,
    *,
    dataset_defaults: dict[str, Any],
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    reference_steps: int,
    search_batch: int,
    search_seed_base: int,
    class_idx: int | None,
) -> dict[str, object]:
    sigmas_ref = _karras_nodes(reference_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, device=device)
    search_seeds = list(range(search_batch))
    x0, model_kwargs = _batch_noise_and_labels(
        seeds=search_seeds,
        seed_base=search_seed_base,
        dataset_defaults=dataset_defaults,
        device=device,
        sigma_max=sigma_max,
        class_idx=class_idx,
    )
    states = [x0]
    x = x0
    sigmas = [float(item) for item in sigmas_ref.detach().cpu().numpy().reshape(-1)]
    for sigma_from, sigma_to in zip(sigmas[:-1], sigmas[1:]):
        x = _ctm_transition(diffusion, model, x, sigma_from, sigma_to, model_kwargs)
        states.append(x)

    interval_costs = np.full((reference_steps + 1, reference_steps + 1), np.inf, dtype=np.float64)
    for i in tqdm.tqdm(range(reference_steps), desc="ctm optimalsteps cost", unit="from"):
        interval_costs[i, i + 1] = 0.0
        x_i = states[i]
        for j in range(i + 2, reference_steps + 1):
            direct = _ctm_transition(diffusion, model, x_i, sigmas[i], sigmas[j], model_kwargs)
            interval_costs[i, j] = float(torch.mean((direct - states[j]) ** 2).item())

    proxy_costs = []
    for i in tqdm.tqdm(range(reference_steps), desc="ctm warp proxy", unit="interval"):
        sigma_from = sigmas[i]
        sigma_to = sigmas[i + 1]
        mid = ((sigma_from ** (1.0 / rho) + sigma_to ** (1.0 / rho)) * 0.5) ** rho
        x_i = states[i]
        direct = states[i + 1]
        x_mid = _ctm_transition(diffusion, model, x_i, sigma_from, mid, model_kwargs)
        two_step = _ctm_transition(diffusion, model, x_mid, mid, sigma_to, model_kwargs)
        defect = float(torch.mean((two_step - direct) ** 2).item())
        step_mag = float(torch.mean((direct - x_i) ** 2).item())
        proxy_costs.append(max(defect + 1.0e-4 * step_mag, 1.0e-12))

    return {
        "sigmas_ref": sigmas_ref,
        "interval_costs": interval_costs,
        "proxy_costs": np.asarray(proxy_costs, dtype=np.float64),
        "reference_steps": int(reference_steps),
        "search_batch": int(search_batch),
        "search_seed_base": int(search_seed_base),
    }


def _optimal_path(costs: np.ndarray, *, step_count: int) -> list[int]:
    node_count = int(costs.shape[0])
    max_idx = node_count - 1
    if step_count < 1:
        raise ValueError(step_count)
    if step_count > max_idx:
        raise ValueError(f"step_count={step_count} exceeds reference interval count={max_idx}")
    dp = np.full((step_count + 1, node_count), np.inf, dtype=np.float64)
    prev = np.full((step_count + 1, node_count), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for k in range(1, step_count + 1):
        for j in range(k, max_idx + 1):
            best_val = np.inf
            best_i = -1
            for i in range(k - 1, j):
                val = dp[k - 1, i] + costs[i, j]
                if val < best_val:
                    best_val = val
                    best_i = i
            dp[k, j] = best_val
            prev[k, j] = best_i
    path = [max_idx]
    cur = max_idx
    for k in range(step_count, 0, -1):
        cur = int(prev[k, cur])
        if cur < 0:
            raise RuntimeError(f"Failed to recover optimal path for step_count={step_count}")
        path.append(cur)
    return list(reversed(path))


def _optimalsteps_schedules(
    ref: dict[str, object],
    *,
    step_counts: list[int],
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, dict[str, object]]]:
    sigmas_ref = ref["sigmas_ref"]
    costs = ref["interval_costs"]
    assert isinstance(sigmas_ref, torch.Tensor)
    assert isinstance(costs, np.ndarray)
    schedules: dict[int, tuple[torch.Tensor, dict[str, object]]] = {}
    for step_count in step_counts:
        path = _optimal_path(costs, step_count=step_count)
        sigma_nodes = sigmas_ref[path].to(device=device, dtype=torch.float64)
        payload = {
            "strategy": "optimalsteps_ctm",
            "description": (
                "CTM-specific OptimalSteps adaptation: dynamic programming chooses "
                "a sparse exact-transition node path that minimizes MSE to a dense CTM exact trajectory."
            ),
            "step_count": int(step_count),
            "nfe": int(step_count),
            "reference_steps": int(ref["reference_steps"]),
            "search_batch": int(ref["search_batch"]),
            "search_seed_base": int(ref["search_seed_base"]),
            "reference_node_path": [int(item) for item in path],
            "sigma_grid": _tensor_to_float_list(sigma_nodes),
            "dp_cost": float(costs[path[:-1], path[1:]].sum()),
        }
        schedules[step_count] = (sigma_nodes, payload)
    return schedules


def _entropic_schedules(
    *,
    path: Path,
    step_counts: list[int],
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, dict[str, object]]]:
    checkpoint = torch.load(path, map_location="cpu")
    time_values = np.asarray(checkpoint["time"], dtype=np.float64).reshape(-1)
    time_func = np.asarray(checkpoint["time_func"], dtype=np.float64).reshape(-1)
    order = np.argsort(time_func)
    phi = time_func[order]
    sigma = time_values[order]
    keep = np.concatenate([[True], np.diff(phi) > 0])
    phi = phi[keep]
    sigma = sigma[keep]
    schedules: dict[int, tuple[torch.Tensor, dict[str, object]]] = {}
    for step_count in step_counts:
        phi_nodes = np.linspace(float(phi[0]), float(phi[-1]), step_count + 1, dtype=np.float64)
        sigma_nodes = np.interp(phi_nodes, phi, sigma)[::-1].copy().astype(np.float64)
        sigma_nodes[0] = float(sigma_max)
        sigma_nodes[-1] = float(sigma_min)
        sigma_nodes = np.maximum(np.minimum(sigma_nodes, sigma_max), sigma_min)
        sigma_nodes = np.maximum.accumulate(sigma_nodes[::-1])[::-1].copy()
        sigma_tensor = torch.as_tensor(sigma_nodes, dtype=torch.float64, device=device)
        payload = {
            "strategy": "entropic_ctm",
            "description": (
                "Entropic Time Scheduler transfer to CTM: uniformly discretize the provided "
                "rescaled entropic-time function, then use the resulting sigma nodes as CTM exact t->s transitions."
            ),
            "time_path": str(path),
            "time_path_help": checkpoint.get("help"),
            "step_count": int(step_count),
            "nfe": int(step_count),
            "sigma_grid": [float(item) for item in sigma_nodes.tolist()],
            "entropic_phi_nodes_ascending": [float(item) for item in phi_nodes.tolist()],
            "note": "This is a schedule-transfer diagnostic on CTM, not an official Entropic reproduction.",
        }
        schedules[step_count] = (sigma_tensor, payload)
    return schedules


def _inverse_cdf_nodes(
    costs: np.ndarray,
    *,
    step_count: int,
    method: str,
    gamma: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if step_count < 1:
        raise ValueError(step_count)
    weights = np.maximum(np.asarray(costs, dtype=np.float64), 1.0e-12) ** float(gamma)
    x_edges = np.linspace(0.0, 1.0, weights.size + 1, dtype=np.float64)
    cdf = np.concatenate([[0.0], np.cumsum(weights)])
    cdf /= float(cdf[-1])
    mass_nodes = np.linspace(0.0, 1.0, step_count + 1, dtype=np.float64)
    if method == "piecewise_linear_ctm":
        x_nodes = np.interp(mass_nodes, cdf, x_edges)
        interpolation = "piecewise-linear inverse CDF"
    elif method == "spline_warp_ctm":
        try:
            from scipy.interpolate import PchipInterpolator
        except ImportError as exc:
            raise RuntimeError("spline_warp_ctm requires scipy.interpolate.PchipInterpolator") from exc
        interpolator = PchipInterpolator(cdf, x_edges, extrapolate=False)
        x_nodes = np.asarray(interpolator(mass_nodes), dtype=np.float64)
        interpolation = "monotone PCHIP inverse CDF"
    else:
        raise ValueError(method)
    x_nodes[0] = 0.0
    x_nodes[-1] = 1.0
    x_nodes = np.maximum.accumulate(np.clip(x_nodes, 0.0, 1.0))
    return x_nodes, {
        "proxy_costs": [float(item) for item in costs.tolist()],
        "proxy_weights": [float(item) for item in weights.tolist()],
        "proxy_gamma": float(gamma),
        "warp_interpolation": interpolation,
        "mass_nodes": [float(item) for item in mass_nodes.tolist()],
        "x_nodes": [float(item) for item in x_nodes.tolist()],
    }


def _warp_schedules(
    ref: dict[str, object],
    *,
    step_counts: list[int],
    strategy: str,
    gamma: float,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, dict[str, object]]]:
    sigmas_ref = ref["sigmas_ref"]
    proxy_costs = ref["proxy_costs"]
    assert isinstance(sigmas_ref, torch.Tensor)
    assert isinstance(proxy_costs, np.ndarray)
    reference_steps = int(ref["reference_steps"])
    x_ref = np.linspace(0.0, 1.0, reference_steps + 1, dtype=np.float64)
    sigma_ref = sigmas_ref.detach().cpu().numpy().astype(np.float64)
    schedules: dict[int, tuple[torch.Tensor, dict[str, object]]] = {}
    for step_count in step_counts:
        x_nodes, warp_payload = _inverse_cdf_nodes(proxy_costs, step_count=step_count, method=strategy, gamma=gamma)
        sigma_nodes = np.interp(x_nodes, x_ref, sigma_ref)
        sigma_nodes[0] = sigma_ref[0]
        sigma_nodes[-1] = sigma_ref[-1]
        sigma_tensor = torch.as_tensor(sigma_nodes, dtype=torch.float64, device=device)
        payload = {
            "strategy": strategy,
            "description": (
                "CTM-specific time-warp diagnostic: a fixed inverse-CDF warp from local CTM "
                "self-consistency residuals, evaluated without training a new model."
            ),
            "step_count": int(step_count),
            "nfe": int(step_count),
            "reference_steps": reference_steps,
            "search_batch": int(ref["search_batch"]),
            "search_seed_base": int(ref["search_seed_base"]),
            "sigma_grid": [float(item) for item in sigma_nodes.tolist()],
            **warp_payload,
        }
        schedules[step_count] = (sigma_tensor, payload)
    return schedules


def _uniform_schedules(
    *,
    step_counts: list[int],
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, dict[str, object]]]:
    schedules: dict[int, tuple[torch.Tensor, dict[str, object]]] = {}
    for step_count in step_counts:
        sigma_nodes = _karras_nodes(step_count, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, device=device)
        payload = {
            "strategy": "uniform_ctm",
            "description": "Official-style CTM exact Karras rho-grid, included only when explicitly requested.",
            "step_count": int(step_count),
            "nfe": int(step_count),
            "sigma_grid": _tensor_to_float_list(sigma_nodes),
        }
        schedules[step_count] = (sigma_nodes, payload)
    return schedules


def _write_schedule(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def _generate_images(
    diffusion,
    model: torch.nn.Module,
    *,
    dataset_defaults: dict[str, Any],
    sigma_nodes: torch.Tensor,
    sample_dir: Path,
    seeds: list[int],
    batch: int,
    device: torch.device,
    seed_base: int,
    sigma_max: float,
    class_idx: int | None,
    subdirs: bool,
) -> None:
    if not seeds:
        return
    sample_dir.mkdir(parents=True, exist_ok=True)
    for start in tqdm.tqdm(range(0, len(seeds), batch), desc=f"generate {sample_dir.name}", unit="batch"):
        batch_seeds = seeds[start : start + batch]
        x, model_kwargs = _batch_noise_and_labels(
            seeds=batch_seeds,
            seed_base=seed_base,
            dataset_defaults=dataset_defaults,
            device=device,
            sigma_max=sigma_max,
            class_idx=class_idx,
        )
        images = _sample_with_schedule(diffusion, model, x, sigma_nodes, model_kwargs)
        images_np = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            path = _image_path(sample_dir, seed, subdirs=subdirs)
            path.parent.mkdir(parents=True, exist_ok=True)
            PIL.Image.fromarray(image_np).save(path)


def _make_preview(sample_dir: Path, preview_path: Path, *, subdirs: bool, seed_start: int, max_images: int = 64) -> str | None:
    images = []
    for seed in range(seed_start, seed_start + max_images):
        path = _image_path(sample_dir, seed, subdirs=subdirs)
        if path.is_file():
            images.append(PIL.Image.open(path).convert("RGB"))
    if not images:
        return None
    width, height = images[0].size
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    canvas = PIL.Image.new("RGB", (cols * width, rows * height), (255, 255, 255))
    for idx, image in enumerate(images):
        canvas.paste(image, ((idx % cols) * width, (idx // cols) * height))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_path)
    return str(preview_path)


def _write_outputs(*, rows: list[dict[str, Any]], eval_root: Path, result_root: Path, dataset: str) -> None:
    report_dir = eval_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"records": rows}, handle, indent=2)
    if rows:
        with (report_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    result_root.mkdir(parents=True, exist_ok=True)
    if rows:
        for suffix in ["summary.csv", "summary.json"]:
            source = report_dir / suffix
            target = result_root / f"ctm_schedule_warp_{dataset}_5k_{suffix}"
            target.write_bytes(source.read_bytes())


def main() -> None:
    args = parse_args()
    defaults = _dataset_defaults(args)
    checkpoint = Path(defaults["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not EDM_ROOT.exists():
        raise FileNotFoundError(EDM_ROOT)

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sample_root = _resolve(args.sample_root) / args.dataset
    eval_root = _resolve(args.eval_root) / args.dataset
    result_root = _resolve(args.result_root)
    step_counts = [int(item) for item in args.steps]

    model, diffusion, model_args = _load_ctm(args.dataset, defaults, device=device, use_fp16=args.use_fp16)
    sigma_min = max(float(args.sigma_min), float(getattr(model_args, "sigma_min", args.sigma_min)))
    sigma_max = min(float(args.sigma_max), float(getattr(model_args, "sigma_max", args.sigma_max)))

    ref: dict[str, object] | None = None
    if any(strategy in args.strategies for strategy in ["optimalsteps_ctm", "piecewise_linear_ctm", "spline_warp_ctm"]):
        ref = _reference_pack(
            diffusion,
            model,
            dataset_defaults=defaults,
            device=device,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=float(args.rho),
            reference_steps=int(defaults["reference_steps"]),
            search_batch=int(args.search_batch),
            search_seed_base=int(args.search_seed_base),
            class_idx=args.class_idx,
        )

    schedules_by_strategy: dict[str, dict[int, tuple[torch.Tensor, dict[str, object]]]] = {}
    if "optimalsteps_ctm" in args.strategies:
        assert ref is not None
        schedules_by_strategy["optimalsteps_ctm"] = _optimalsteps_schedules(ref, step_counts=step_counts, device=device)
    if "entropic_ctm" in args.strategies:
        schedules_by_strategy["entropic_ctm"] = _entropic_schedules(
            path=Path(defaults["entropic_time_path"]),
            step_counts=step_counts,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            device=device,
        )
    if "piecewise_linear_ctm" in args.strategies:
        assert ref is not None
        schedules_by_strategy["piecewise_linear_ctm"] = _warp_schedules(
            ref,
            step_counts=step_counts,
            strategy="piecewise_linear_ctm",
            gamma=float(args.proxy_gamma),
            device=device,
        )
    if "spline_warp_ctm" in args.strategies:
        assert ref is not None
        schedules_by_strategy["spline_warp_ctm"] = _warp_schedules(
            ref,
            step_counts=step_counts,
            strategy="spline_warp_ctm",
            gamma=float(args.proxy_gamma),
            device=device,
        )
    if "uniform_ctm" in args.strategies:
        schedules_by_strategy["uniform_ctm"] = _uniform_schedules(
            step_counts=step_counts,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=float(args.rho),
            device=device,
        )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(EDM_ROOT) + os.pathsep + str(defaults["ctm_root"]) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("DNNLIB_CACHE_DIR", str(ROOT / ".torch" / "dnnlib"))
    env["NCCL_DEBUG"] = "WARN"
    env["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

    rows: list[dict[str, Any]] = []
    for strategy in args.strategies:
        for step_count in step_counts:
            sigma_nodes, schedule_payload = schedules_by_strategy[strategy][step_count]
            schedule_path = eval_root / "schedules" / strategy / f"steps{step_count}.json"
            _write_schedule(schedule_path, schedule_payload)
            sample_dir = sample_root / strategy / f"steps{step_count}" / "images"
            step_dir = eval_root / strategy / f"steps{step_count}"
            metrics_path = step_dir / "metrics.json"
            step_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()

            existing = None
            if args.skip_existing and metrics_path.exists():
                try:
                    existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing = None
                if existing is not None:
                    if (
                        existing.get("fid") is not None
                        and int(existing.get("num_fid_samples", 0) or 0) == int(args.num_samples)
                        and _count_pngs(sample_dir) >= int(args.num_samples)
                    ):
                        rows.append(existing)
                        _write_outputs(rows=rows, eval_root=eval_root, result_root=result_root, dataset=args.dataset)
                        print(f"reuse {args.dataset} {strategy} steps={step_count} fid={existing['fid']}", flush=True)
                        continue

            missing = _missing_seeds(
                sample_dir,
                seed_start=int(args.seed_start),
                num_samples=int(args.num_samples),
                subdirs=bool(args.subdirs),
            )
            if missing and not args.dry_run_schedules:
                _generate_images(
                    diffusion,
                    model,
                    dataset_defaults=defaults,
                    sigma_nodes=sigma_nodes,
                    sample_dir=sample_dir,
                    seeds=missing,
                    batch=int(defaults["batch"]),
                    device=device,
                    seed_base=int(args.seed_base),
                    sigma_max=sigma_max,
                    class_idx=args.class_idx,
                    subdirs=bool(args.subdirs),
                )
            png_count = _count_pngs(sample_dir)
            if png_count < int(args.num_samples) and not args.dry_run_schedules:
                raise RuntimeError(f"Only found {png_count}/{args.num_samples} PNGs in {sample_dir}")

            preview_path = _make_preview(
                sample_dir,
                eval_root / "previews" / f"{strategy}_steps{step_count}.png",
                subdirs=bool(args.subdirs),
                seed_start=int(args.seed_start),
            )
            fid = None
            if not args.skip_fid and not args.dry_run_schedules:
                fid_stdout = _run_and_tee(
                    [
                        sys.executable,
                        str(EDM_ROOT / "fid.py"),
                        "calc",
                        f"--images={sample_dir}",
                        f"--ref={defaults['fid_ref']}",
                        f"--num={args.num_samples}",
                        f"--batch={args.fid_batch}",
                    ],
                    cwd=EDM_ROOT,
                    env=env,
                    log_path=step_dir / "fid.stdout_stderr.txt",
                )
                fid = _parse_fid(fid_stdout)

            row = {
                "dataset": args.dataset,
                "strategy": strategy,
                "method": f"CTM exact + {strategy}",
                "step_count": int(step_count),
                "nfe": int(step_count),
                "fid": fid,
                "num_fid_samples": int(args.num_samples),
                "batch": int(defaults["batch"]),
                "fid_batch": int(args.fid_batch),
                "checkpoint": str(checkpoint),
                "fid_ref": str(defaults["fid_ref"]),
                "solver": "ctm_exact_custom_sigma_grid",
                "sigma_grid": json.dumps(_tensor_to_float_list(sigma_nodes)),
                "schedule_json": str(schedule_path),
                "sample_dir": str(sample_dir),
                "preview_path": preview_path or "",
                "seed_start": int(args.seed_start),
                "seed_end": int(args.seed_start) + int(args.num_samples) - 1,
                "elapsed_sec": time.time() - t0,
            }
            metrics_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            _write_outputs(rows=rows, eval_root=eval_root, result_root=result_root, dataset=args.dataset)
            print(
                f"ctm schedule {args.dataset} {strategy} steps={step_count} fid={fid} "
                f"elapsed_sec={row['elapsed_sec']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
