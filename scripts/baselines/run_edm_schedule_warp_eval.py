from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time
from typing import Iterable

import numpy as np
import PIL.Image
import torch
import tqdm


ROOT = Path(__file__).resolve().parents[2]
EDM_ROOT = ROOT / "refs" / "edm"
if str(EDM_ROOT) not in sys.path:
    sys.path.insert(0, str(EDM_ROOT))

import dnnlib  # noqa: E402
from generate import StackedRandomGenerator  # noqa: E402


DEFAULT_NETWORK = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"
DEFAULT_FID_REF = "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"
DEFAULT_ENTROPIC_TIME = (
    ROOT
    / "refs"
    / "entropic_time_schedulers"
    / "EDM"
    / "Schedules"
    / "RE_function_CIFAR10_uncond_vp_80_128_FREQ.pt"
)
FID_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate schedule/time-warp baselines on one official EDM checkpoint. "
            "The sampler is EDM/Heun with a custom monotone sigma grid."
        )
    )
    parser.add_argument("--network", default=DEFAULT_NETWORK, help="EDM network pkl path or URL")
    parser.add_argument("--fid-ref", default=DEFAULT_FID_REF, help="EDM FID reference npz path or URL")
    parser.add_argument("--sample-root", required=True, help="Directory for generated PNG samples")
    parser.add_argument("--eval-root", required=True, help="Directory for metrics, logs and schedules")
    parser.add_argument("--result-root", default=None, help="Optional directory for copied summary tables")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["optimalsteps_edm", "entropic_edm", "piecewise_linear", "spline_warp"],
        choices=["optimalsteps_edm", "entropic_edm", "piecewise_linear", "spline_warp", "uniform_edm"],
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--fid-batch", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--class-idx", type=int, default=None, help="Fixed class label; default uses EDM random labels")
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--reference-steps", type=int, default=64, help="Dense EDM teacher grid for search/proxy warp")
    parser.add_argument("--search-batch", type=int, default=8, help="Seed count for OptimalSteps/proxy schedule search")
    parser.add_argument("--search-seed-start", type=int, default=123000)
    parser.add_argument("--proxy-gamma", type=float, default=0.5, help="Mass compression exponent for proxy-density warps")
    parser.add_argument("--entropic-time-path", default=str(DEFAULT_ENTROPIC_TIME))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subdirs", action="store_true", default=True)
    parser.add_argument("--no-subdirs", action="store_false", dest="subdirs")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse metrics with enough images")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--dry-run-schedules", action="store_true", help="Only write schedule JSON files")
    return parser.parse_args()


def _as_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _nfe(step_count: int) -> int:
    return max(1, 2 * int(step_count) - 1)


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


def _net_sigma_limits(net: torch.nn.Module, sigma_min: float, sigma_max: float) -> tuple[float, float]:
    min_supported = float(getattr(net, "sigma_min", sigma_min))
    max_supported = float(getattr(net, "sigma_max", sigma_max))
    return max(float(sigma_min), min_supported), min(float(sigma_max), max_supported)


def _standard_t_steps(
    step_count: int,
    *,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    net: torch.nn.Module | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}")
    if step_count == 1:
        sigmas = torch.tensor([sigma_max], dtype=torch.float64, device=device)
    else:
        indices = torch.arange(step_count, dtype=torch.float64, device=device)
        sigmas = (
            sigma_max ** (1.0 / rho)
            + indices / float(step_count - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        ) ** rho
    if net is not None:
        sigmas = net.round_sigma(sigmas).to(torch.float64)
    return torch.cat([sigmas, torch.zeros(1, dtype=torch.float64, device=device)])


def _tensor_to_float_list(values: torch.Tensor | np.ndarray | Iterable[float]) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return [float(item) for item in np.asarray(list(values) if not isinstance(values, np.ndarray) else values).reshape(-1)]


def _interval_step(
    net: torch.nn.Module,
    x_cur: torch.Tensor,
    t_cur: torch.Tensor,
    t_next: torch.Tensor,
    class_labels: torch.Tensor | None,
    *,
    return_euler: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
    d_cur = (x_cur - denoised) / t_cur
    x_euler = x_cur + (t_next - t_cur) * d_cur
    if float(t_next.item()) > 0:
        denoised_next = net(x_euler, t_next, class_labels).to(torch.float64)
        d_prime = (x_euler - denoised_next) / t_next
        x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    else:
        x_next = x_euler
    if return_euler:
        return x_next, x_euler
    return x_next


@torch.no_grad()
def _sample_with_t_steps(
    net: torch.nn.Module,
    latents: torch.Tensor,
    class_labels: torch.Tensor | None,
    t_steps: torch.Tensor,
    *,
    randn_like,
    S_churn: float = 0.0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
) -> torch.Tensor:
    del randn_like, S_churn, S_min, S_max, S_noise
    # These schedule baselines are deterministic EDM/Heun samplers; stochastic
    # churn is intentionally disabled to isolate the time grid.
    t_steps = t_steps.to(device=latents.device, dtype=torch.float64)
    if t_steps.ndim != 1 or t_steps.numel() < 2:
        raise ValueError(f"t_steps must be 1D with at least two nodes, got {tuple(t_steps.shape)}")
    if abs(float(t_steps[-1].item())) > 1.0e-12:
        raise ValueError("t_steps must end at sigma=0")
    if not torch.all(t_steps[:-1] >= t_steps[1:]):
        raise ValueError("t_steps must be monotone non-increasing")

    x_next = latents.to(torch.float64) * t_steps[0]
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        x_next = _interval_step(net, x_next, t_cur, t_next, class_labels)
    return x_next


@torch.no_grad()
def _reference_pack(
    net: torch.nn.Module,
    *,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    reference_steps: int,
    search_batch: int,
    search_seed_start: int,
) -> dict[str, object]:
    search_seeds = list(range(search_seed_start, search_seed_start + search_batch))
    rnd = StackedRandomGenerator(device, search_seeds)
    latents = rnd.randn(
        [search_batch, int(net.img_channels), int(net.img_resolution), int(net.img_resolution)],
        device=device,
    )
    class_labels = None
    if int(getattr(net, "label_dim", 0) or 0) > 0:
        class_labels = torch.eye(int(net.label_dim), device=device)[
            rnd.randint(int(net.label_dim), size=[search_batch], device=device)
        ]
    t_ref = _standard_t_steps(
        reference_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        net=net,
        device=device,
    )

    states = [latents.to(torch.float64) * t_ref[0]]
    proxy_costs = []
    x_cur = states[0]
    for t_cur, t_next in zip(t_ref[:-1], t_ref[1:]):
        x_next, x_euler = _interval_step(net, x_cur, t_cur, t_next, class_labels, return_euler=True)
        heun_gap = (x_next - x_euler).detach().float().flatten(1).square().mean(dim=1).mean()
        step_magnitude = (x_next - x_cur).detach().float().flatten(1).square().mean(dim=1).mean()
        proxy_costs.append(float((heun_gap + 1.0e-4 * step_magnitude).item()))
        states.append(x_next)
        x_cur = x_next

    states_tensor = torch.stack(states, dim=0)
    return {
        "t_ref": t_ref,
        "states": states_tensor,
        "class_labels": class_labels,
        "proxy_costs": np.asarray(proxy_costs, dtype=np.float64),
        "search_seeds": search_seeds,
    }


@torch.no_grad()
def _optimalsteps_schedules(
    net: torch.nn.Module,
    ref: dict[str, object],
    *,
    step_counts: list[int],
) -> dict[int, tuple[torch.Tensor, dict[str, object]]]:
    t_ref = ref["t_ref"]
    states = ref["states"]
    class_labels = ref["class_labels"]
    assert isinstance(t_ref, torch.Tensor)
    assert isinstance(states, torch.Tensor)
    assert class_labels is None or isinstance(class_labels, torch.Tensor)
    reference_steps = int(t_ref.numel() - 1)

    inf = float("inf")
    costs = torch.full(
        (reference_steps + 1, reference_steps + 1),
        inf,
        dtype=torch.float64,
        device=t_ref.device,
    )
    for i in tqdm.tqdm(range(reference_steps), desc="optimalsteps cost", unit="src"):
        x_i = states[i]
        for j in range(i + 1, reference_steps + 1):
            pred = _interval_step(net, x_i, t_ref[i], t_ref[j], class_labels)
            cost = (pred - states[j]).detach().float().flatten(1).square().mean(dim=1).mean()
            costs[i, j] = cost.to(torch.float64)

    schedules: dict[int, tuple[torch.Tensor, dict[str, object]]] = {}
    for step_count in step_counts:
        if step_count > reference_steps:
            raise ValueError(f"step_count={step_count} cannot exceed reference_steps={reference_steps}")
        dp = torch.full(
            (step_count + 1, reference_steps + 1),
            inf,
            dtype=torch.float64,
            device=t_ref.device,
        )
        prev = torch.full(
            (step_count + 1, reference_steps + 1),
            -1,
            dtype=torch.long,
            device=t_ref.device,
        )
        dp[0, 0] = 0.0
        for k in range(1, step_count + 1):
            for j in range(1, reference_steps + 1):
                candidates = dp[k - 1, :j] + costs[:j, j]
                best_cost, best_idx = torch.min(candidates, dim=0)
                dp[k, j] = best_cost
                prev[k, j] = best_idx
        if not torch.isfinite(dp[step_count, reference_steps]):
            raise RuntimeError(f"OptimalSteps DP failed for step_count={step_count}")

        indices = [reference_steps]
        cursor = reference_steps
        for k in range(step_count, 0, -1):
            cursor = int(prev[k, cursor].item())
            indices.append(cursor)
        indices.reverse()
        if indices[0] != 0 or indices[-1] != reference_steps or len(indices) != step_count + 1:
            raise RuntimeError(f"Invalid OptimalSteps path for step_count={step_count}: {indices}")
        t_steps = t_ref[torch.as_tensor(indices, dtype=torch.long, device=t_ref.device)]
        t_steps[0] = t_ref[0]
        t_steps[-1] = torch.tensor(0.0, device=t_steps.device, dtype=t_steps.dtype)
        payload = {
            "strategy": "optimalsteps_edm",
            "description": (
                "DP schedule search adapted from OptimalSteps: dense EDM/Heun teacher trajectory, "
                "one custom EDM interval as student transition, MSE in image space as interval cost."
            ),
            "reference_steps": reference_steps,
            "search_batch": int(states.shape[1]),
            "search_seeds": ref["search_seeds"],
            "indices": [int(item) for item in indices],
            "step_count": int(step_count),
            "nfe": _nfe(step_count),
            "total_dp_cost": float(dp[step_count, reference_steps].item()),
            "interval_costs": [float(costs[i, j].item()) for i, j in zip(indices, indices[1:])],
            "sigma_grid": _tensor_to_float_list(t_steps),
        }
        schedules[step_count] = (t_steps.detach().clone(), payload)
    return schedules


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    if n != len(y):
        raise ValueError("x/y length mismatch")
    if n < 2:
        raise ValueError("PCHIP needs at least two knots")
    h = np.diff(x)
    delta = np.diff(y) / h
    if n == 2:
        return np.asarray([delta[0], delta[0]], dtype=np.float64)

    d = np.zeros(n, dtype=np.float64)
    for k in range(1, n - 1):
        if delta[k - 1] == 0 or delta[k] == 0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
            d[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    d[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(d[0]) != np.sign(delta[0]):
        d[0] = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(d[0]) > abs(3.0 * delta[0]):
        d[0] = 3.0 * delta[0]

    d[-1] = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(d[-1]) != np.sign(delta[-1]):
        d[-1] = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(d[-1]) > abs(3.0 * delta[-1]):
        d[-1] = 3.0 * delta[-1]
    return d


def _pchip_eval(x: np.ndarray, y: np.ndarray, slopes: np.ndarray, xq: np.ndarray) -> np.ndarray:
    xq = np.asarray(xq, dtype=np.float64)
    idx = np.searchsorted(x, xq, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)
    h = x[idx + 1] - x[idx]
    s = (xq - x[idx]) / h
    h00 = (2.0 * s**3) - (3.0 * s**2) + 1.0
    h10 = (s**3) - (2.0 * s**2) + s
    h01 = (-2.0 * s**3) + (3.0 * s**2)
    h11 = (s**3) - (s**2)
    return h00 * y[idx] + h10 * h * slopes[idx] + h01 * y[idx + 1] + h11 * h * slopes[idx + 1]


def _inverse_cdf_nodes(
    weights: np.ndarray,
    *,
    step_count: int,
    method: str,
    gamma: float,
) -> tuple[np.ndarray, dict[str, object]]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1.0e-12)
    weights = weights ** float(gamma)
    weights = weights / weights.sum()
    cdf = np.concatenate([[0.0], np.cumsum(weights)])
    cdf[-1] = 1.0
    x_knots = np.linspace(0.0, 1.0, len(cdf), dtype=np.float64)
    r_nodes = np.linspace(0.0, 1.0, step_count + 1, dtype=np.float64)
    if method == "piecewise_linear":
        x_nodes = np.interp(r_nodes, cdf, x_knots)
        smoothness = "piecewise-linear inverse CDF"
    elif method == "spline_warp":
        slopes = _pchip_slopes(cdf, x_knots)
        x_nodes = _pchip_eval(cdf, x_knots, slopes, r_nodes)
        x_nodes = np.clip(x_nodes, 0.0, 1.0)
        x_nodes = np.maximum.accumulate(x_nodes)
        x_nodes[0] = 0.0
        x_nodes[-1] = 1.0
        smoothness = "monotone PCHIP inverse CDF"
    else:
        raise ValueError(f"Unknown warp method: {method}")
    payload = {
        "proxy_weights": [float(item) for item in weights.tolist()],
        "cdf_knots": [float(item) for item in cdf.tolist()],
        "coordinate_nodes": [float(item) for item in x_nodes.tolist()],
        "mass_nodes": [float(item) for item in r_nodes.tolist()],
        "warp_interpolation": smoothness,
        "proxy_gamma": float(gamma),
    }
    return x_nodes, payload


def _warp_schedules(
    ref: dict[str, object],
    *,
    step_counts: list[int],
    strategy: str,
    gamma: float,
) -> dict[int, tuple[torch.Tensor, dict[str, object]]]:
    t_ref = ref["t_ref"]
    proxy_costs = ref["proxy_costs"]
    assert isinstance(t_ref, torch.Tensor)
    assert isinstance(proxy_costs, np.ndarray)
    reference_steps = int(t_ref.numel() - 1)
    x_ref = np.linspace(0.0, 1.0, reference_steps + 1, dtype=np.float64)
    sigma_ref = t_ref.detach().cpu().numpy().astype(np.float64)
    schedules: dict[int, tuple[torch.Tensor, dict[str, object]]] = {}
    for step_count in step_counts:
        x_nodes, warp_payload = _inverse_cdf_nodes(
            proxy_costs,
            step_count=step_count,
            method=strategy,
            gamma=gamma,
        )
        sigmas = np.interp(x_nodes, x_ref, sigma_ref)
        sigmas[0] = sigma_ref[0]
        sigmas[-1] = 0.0
        t_steps = torch.as_tensor(sigmas, dtype=torch.float64, device=t_ref.device)
        payload = {
            "strategy": strategy,
            "description": (
                "Fixed EDM time-warp baseline using a local dense-grid proxy "
                "(Euler-Heun gap plus small step magnitude) from the same checkpoint."
            ),
            "reference_steps": reference_steps,
            "step_count": int(step_count),
            "nfe": _nfe(step_count),
            "sigma_grid": [float(item) for item in sigmas.tolist()],
            **warp_payload,
        }
        schedules[step_count] = (t_steps, payload)
    return schedules


def _entropic_schedules(
    *,
    path: Path,
    step_counts: list[int],
    sigma_min: float,
    sigma_max: float,
    net: torch.nn.Module,
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
        if step_count == 1:
            sigma_nodes = np.asarray([sigma_max], dtype=np.float64)
            phi_nodes = np.asarray([float(phi[-1])], dtype=np.float64)
        else:
            phi_nodes = np.linspace(float(phi[0]), float(phi[-1]), step_count, dtype=np.float64)
            sigma_nodes = np.interp(phi_nodes, phi, sigma)[::-1].astype(np.float64)
            sigma_nodes[0] = sigma_max
            sigma_nodes[-1] = max(sigma_min, min(sigma_nodes[-1], sigma_max))
        t_steps = torch.cat(
            [
                net.round_sigma(torch.as_tensor(sigma_nodes, dtype=torch.float64, device=device)).to(torch.float64),
                torch.zeros(1, dtype=torch.float64, device=device),
            ]
        )
        payload = {
            "strategy": "entropic_edm",
            "description": (
                "Entropic Time Scheduler schedule only: uniformly discretize the provided "
                "rescaled entropic-time function and run the same EDM/Heun sampler/checkpoint."
            ),
            "time_path": str(path),
            "time_path_help": checkpoint.get("help"),
            "step_count": int(step_count),
            "nfe": _nfe(step_count),
            "sigma_grid": _tensor_to_float_list(t_steps),
            "entropic_phi_nodes_ascending": [float(item) for item in phi_nodes.tolist()],
            "note": "The bundled CIFAR-10 entropic schedule is labeled uncond-VP; this run isolates schedule transfer on the cond-VP EDM checkpoint.",
        }
        schedules[step_count] = (t_steps, payload)
    return schedules


def _uniform_schedules(
    *,
    step_counts: list[int],
    sigma_min: float,
    sigma_max: float,
    rho: float,
    net: torch.nn.Module,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, dict[str, object]]]:
    schedules: dict[int, tuple[torch.Tensor, dict[str, object]]] = {}
    for step_count in step_counts:
        t_steps = _standard_t_steps(
            step_count,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            net=net,
            device=device,
        )
        payload = {
            "strategy": "uniform_edm",
            "description": "Official EDM/Karras rho-grid, included as a reference schedule.",
            "step_count": int(step_count),
            "nfe": _nfe(step_count),
            "sigma_grid": _tensor_to_float_list(t_steps),
        }
        schedules[step_count] = (t_steps, payload)
    return schedules


def _write_schedule(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def _generate_images(
    net: torch.nn.Module,
    *,
    t_steps: torch.Tensor,
    sample_dir: Path,
    seeds: list[int],
    batch: int,
    device: torch.device,
    subdirs: bool,
    class_idx: int | None,
) -> None:
    if not seeds:
        return
    sample_dir.mkdir(parents=True, exist_ok=True)
    for start in tqdm.tqdm(range(0, len(seeds), batch), desc=f"generate {sample_dir.name}", unit="batch"):
        batch_seeds = seeds[start : start + batch]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [len(batch_seeds), int(net.img_channels), int(net.img_resolution), int(net.img_resolution)],
            device=device,
        )
        class_labels = None
        if int(getattr(net, "label_dim", 0) or 0) > 0:
            class_labels = torch.eye(int(net.label_dim), device=device)[
                rnd.randint(int(net.label_dim), size=[len(batch_seeds)], device=device)
            ]
            if class_idx is not None:
                class_labels[:, :] = 0
                class_labels[:, int(class_idx)] = 1
        images = _sample_with_t_steps(net, latents, class_labels, t_steps, randn_like=rnd.randn_like)
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            path = _image_path(sample_dir, seed, subdirs=subdirs)
            path.parent.mkdir(parents=True, exist_ok=True)
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0]).save(path)
            else:
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
    for image in images:
        image.close()
    return str(preview_path)


def _run_fid(
    *,
    image_dir: Path,
    fid_ref: str,
    num_samples: int,
    batch: int,
    log_path: Path,
) -> float | None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(EDM_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("DNNLIB_CACHE_DIR", "/cache/Zhengwei/DG-TWFD-runtime/.torch/dnnlib")
    env["NCCL_DEBUG"] = "WARN"
    env["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
    cmd = [
        sys.executable,
        str(EDM_ROOT / "fid.py"),
        "calc",
        f"--images={image_dir}",
        f"--ref={fid_ref}",
        f"--num={num_samples}",
        f"--batch={batch}",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(EDM_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured = []
    assert proc.stdout is not None
    with log_path.open("w", encoding="utf-8") as handle:
        for line in proc.stdout:
            print(line, end="", flush=True)
            captured.append(line)
            handle.write(line)
            handle.flush()
    returncode = proc.wait()
    stdout = "".join(captured)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd, output=stdout)
    return _parse_fid(stdout)


def _write_reports(records: list[dict[str, object]], report_dir: Path, result_root: Path | None) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_json = report_dir / "summary.json"
    summary_csv = report_dir / "summary.csv"
    summary_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    if records:
        fieldnames = list(records[0].keys())
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        best_by_strategy = {}
        for record in records:
            fid = record.get("fid")
            if fid is None:
                continue
            strategy = str(record["strategy"])
            if strategy not in best_by_strategy or float(fid) < float(best_by_strategy[strategy]["fid"]):
                best_by_strategy[strategy] = record
        (report_dir / "best_by_strategy.json").write_text(
            json.dumps(best_by_strategy, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if result_root is not None:
        result_root.mkdir(parents=True, exist_ok=True)
        if summary_json.exists():
            (result_root / "edm_schedule_warp_cifar10_5k_summary.json").write_text(
                summary_json.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        if summary_csv.exists():
            (result_root / "edm_schedule_warp_cifar10_5k_summary.csv").write_text(
                summary_csv.read_text(encoding="utf-8"),
                encoding="utf-8",
            )


def main() -> None:
    args = parse_args()
    if not EDM_ROOT.exists():
        raise FileNotFoundError(f"refs/edm not found: {EDM_ROOT}")
    sample_root = _as_path(args.sample_root)
    eval_root = _as_path(args.eval_root)
    result_root = _as_path(args.result_root) if args.result_root else None
    schedules_dir = eval_root / "schedules"
    report_dir = eval_root / "reports"
    preview_dir = eval_root / "previews"
    logs_dir = eval_root / "logs"
    sample_root.mkdir(parents=True, exist_ok=True)
    schedules_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    os.environ.setdefault("DNNLIB_CACHE_DIR", "/cache/Zhengwei/DG-TWFD-runtime/.torch/dnnlib")
    print(f"Loading EDM network: {args.network}", flush=True)
    with dnnlib.util.open_url(args.network) as handle:
        net = pickle.load(handle)["ema"].to(device)
    net.eval().requires_grad_(False)
    sigma_min, sigma_max = _net_sigma_limits(net, args.sigma_min, args.sigma_max)
    step_counts = [int(item) for item in args.steps]

    print(
        "building schedules "
        f"strategies={args.strategies} steps={step_counts} "
        f"sigma_min={sigma_min:g} sigma_max={sigma_max:g}",
        flush=True,
    )
    schedules_by_strategy: dict[str, dict[int, tuple[torch.Tensor, dict[str, object]]]] = {}
    need_ref = any(strategy in {"optimalsteps_edm", "piecewise_linear", "spline_warp"} for strategy in args.strategies)
    ref = None
    if need_ref:
        ref = _reference_pack(
            net,
            device=device,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=float(args.rho),
            reference_steps=int(args.reference_steps),
            search_batch=int(args.search_batch),
            search_seed_start=int(args.search_seed_start),
        )
    if "optimalsteps_edm" in args.strategies:
        assert ref is not None
        schedules_by_strategy["optimalsteps_edm"] = _optimalsteps_schedules(net, ref, step_counts=step_counts)
    if "piecewise_linear" in args.strategies:
        assert ref is not None
        schedules_by_strategy["piecewise_linear"] = _warp_schedules(
            ref,
            step_counts=step_counts,
            strategy="piecewise_linear",
            gamma=float(args.proxy_gamma),
        )
    if "spline_warp" in args.strategies:
        assert ref is not None
        schedules_by_strategy["spline_warp"] = _warp_schedules(
            ref,
            step_counts=step_counts,
            strategy="spline_warp",
            gamma=float(args.proxy_gamma),
        )
    if "entropic_edm" in args.strategies:
        schedules_by_strategy["entropic_edm"] = _entropic_schedules(
            path=_as_path(args.entropic_time_path),
            step_counts=step_counts,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            net=net,
            device=device,
        )
    if "uniform_edm" in args.strategies:
        schedules_by_strategy["uniform_edm"] = _uniform_schedules(
            step_counts=step_counts,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=float(args.rho),
            net=net,
            device=device,
        )

    for strategy, schedules in schedules_by_strategy.items():
        for step_count, (_t_steps, payload) in schedules.items():
            _write_schedule(schedules_dir / strategy / f"steps{step_count}.json", payload)
    if args.dry_run_schedules:
        print(f"wrote schedules only: {schedules_dir}")
        return

    records: list[dict[str, object]] = []
    for strategy in args.strategies:
        strategy_schedules = schedules_by_strategy[strategy]
        for step_count in step_counts:
            t_steps, payload = strategy_schedules[step_count]
            sample_dir = sample_root / strategy / f"steps{step_count}" / "images"
            metric_dir = eval_root / strategy / f"steps{step_count}"
            metric_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = metric_dir / "metrics.json"
            existing = _count_pngs(sample_dir)
            if args.skip_existing and existing >= args.num_samples and metrics_path.exists():
                try:
                    records.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                    print(f"reuse existing {strategy} steps={step_count}", flush=True)
                    continue
                except json.JSONDecodeError:
                    pass

            t0 = time.time()
            missing = _missing_seeds(
                sample_dir,
                seed_start=int(args.seed_start),
                num_samples=int(args.num_samples),
                subdirs=bool(args.subdirs),
            )
            print(
                f"{strategy} steps={step_count}: {existing} existing, {len(missing)} missing",
                flush=True,
            )
            _generate_images(
                net,
                t_steps=t_steps,
                sample_dir=sample_dir,
                seeds=missing,
                batch=int(args.batch),
                device=device,
                subdirs=bool(args.subdirs),
                class_idx=args.class_idx,
            )
            preview_path = _make_preview(
                sample_dir,
                preview_dir / f"{strategy}_steps{step_count}.png",
                subdirs=bool(args.subdirs),
                seed_start=int(args.seed_start),
            )
            fid = None
            if not args.skip_fid:
                fid = _run_fid(
                    image_dir=sample_dir,
                    fid_ref=str(args.fid_ref),
                    num_samples=int(args.num_samples),
                    batch=int(args.fid_batch),
                    log_path=metric_dir / "fid.stdout_stderr.txt",
                )

            elapsed = time.time() - t0
            record = {
                "dataset": "cifar10",
                "strategy": strategy,
                "method": f"EDM-checkpoint + {strategy}",
                "step_count": int(step_count),
                "nfe": _nfe(step_count),
                "fid": fid,
                "num_fid_samples": int(args.num_samples),
                "batch": int(args.batch),
                "fid_batch": int(args.fid_batch),
                "network": str(args.network),
                "fid_ref": str(args.fid_ref),
                "solver": "edm_heun_custom_sigma_grid",
                "sigma_grid": json.dumps(payload["sigma_grid"]),
                "schedule_json": str(schedules_dir / strategy / f"steps{step_count}.json"),
                "sample_dir": str(sample_dir),
                "preview_path": preview_path,
                "seed_start": int(args.seed_start),
                "seed_end": int(args.seed_start) + int(args.num_samples) - 1,
                "elapsed_sec": elapsed,
            }
            metrics_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            records.append(record)
            _write_reports(records, report_dir, result_root)
            print(
                f"done {strategy} steps={step_count} nfe={_nfe(step_count)} "
                f"fid={fid} elapsed_sec={elapsed:.1f}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _write_reports(records, report_dir, result_root)
    print(f"completed. summary: {report_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
