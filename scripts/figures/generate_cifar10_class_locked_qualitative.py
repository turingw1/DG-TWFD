#!/usr/bin/env python3
"""Generate class-locked CIFAR-10 qualitative samples.

Each generated column uses a fixed pair of (class_id, latent_seed) across all
supported conditional models. This makes cross-model qualitative panels compare
the same semantic target rather than merely reusing a seed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[2]
EDM_FIRST_SRC = ROOT / "experiments" / "edm_first"
EDM_REF = ROOT / "refs" / "edm"
CTM_RUNNER = ROOT / "scripts" / "baselines"

for path in [str(EDM_FIRST_SRC), str(EDM_REF), str(CTM_RUNNER)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.edm_map_lib import (  # noqa: E402
    build_sigma_grid,
    clone_student_from_teacher,
    init_warp,
    load_config,
    load_edm_network,
    make_labels,
    student_transition,
    teacher_transition,
    to_unit,
)
from run_ctm_schedule_warp_eval import (  # noqa: E402
    DATASET_DEFAULTS,
    _ctm_transition,
    _dataset_defaults,
    _karras_nodes,
    _load_ctm,
)


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

EDM_REFERENCE_STEPS = {1: 32, 2: 48, 4: 64, 8: 128}
DEFAULT_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7]
DEFAULT_LATENT_SEEDS = list(range(1000, 1018))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/edm_first/configs/cifar10_edm_map_prior_fullstack_timewarp_v17_rqs_fastwarp.yaml",
    )
    parser.add_argument(
        "--dg-checkpoint",
        default="runs/edm_first_cifar10_prior_fullstack_timewarp_v17_rqs_fastwarp_from_step11855/checkpoints/best.pt",
    )
    parser.add_argument(
        "--ctm-official-checkpoint",
        default="/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/ctm_cifar10/ctm-cifar10/conditional/model043000.pt",
    )
    parser.add_argument(
        "--ctm-nogan-checkpoint",
        default="/cache/Zhengwei/DG-TWFD-runtime/runs/ctm_nogan_20260429/cifar10_nogan_dsm_10k_mb4_gb16_resume_from8000/ema_0.999_010000.pt",
    )
    parser.add_argument(
        "--output-root",
        default="docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260502_paper",
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--class-ids",
        nargs="+",
        type=int,
        default=DEFAULT_CLASS_IDS,
        help="One class id per visual column.",
    )
    parser.add_argument(
        "--latent-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_LATENT_SEEDS,
        help="One latent seed per visual column.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-fp16-ctm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-dg-full",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also generate the DG-TWFD student with the learned checkpoint warp/clock.",
    )
    return parser.parse_args()


def _resolve(path: str | Path) -> Path | str:
    raw = str(path)
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    value = Path(raw)
    return value if value.is_absolute() else ROOT / value


def _save_tensor_images(samples: torch.Tensor, sample_dir: Path, seeds: list[int]) -> list[str]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    written = []
    x = samples.detach().cpu()
    if float(x.min()) < -0.01:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    for image, seed in zip(x, seeds):
        arr = (image.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
        path = sample_dir / f"{int(seed):06d}.png"
        Image.fromarray(arr, mode="RGB").save(path)
        written.append(str(path.relative_to(ROOT)))
    return written


def _label_tensor(class_ids: list[int], *, label_dim: int, device: torch.device) -> torch.Tensor:
    labels = torch.as_tensor(class_ids, dtype=torch.long, device=device)
    return make_labels(labels, label_dim=label_dim, device=device)


def _latents(
    *,
    seeds: list[int],
    channels: int,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    values = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        values.append(torch.randn((channels, image_size, image_size), generator=generator, device=device))
    return torch.stack(values, dim=0)


@torch.no_grad()
def _sample_edm_student(
    *,
    student: torch.nn.Module,
    warp: torch.nn.Module | None,
    cfg: dict[str, Any],
    step_count: int,
    class_ids: list[int],
    seeds: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_cfg = cfg.get("train", {})
    sigma_min = float(train_cfg.get("sigma_min", 0.002))
    sigma_max = float(train_cfg.get("sigma_max", 80.0))
    rho = float(train_cfg.get("rho", 7.0))
    latents = _latents(
        seeds=seeds,
        channels=int(getattr(student, "img_channels", 3)),
        image_size=int(getattr(student, "img_resolution", 32)),
        device=device,
    )
    labels = _label_tensor(class_ids, label_dim=int(getattr(student, "label_dim", 0) or 0), device=device)
    u_grid, sigma_grid = build_sigma_grid(
        step_count=step_count,
        warp=warp,
        device=device,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        net=student,
    )
    x = latents * sigma_grid[0].to(latents.dtype)
    for sigma_t, sigma_s in zip(sigma_grid[:-1], sigma_grid[1:]):
        x = student_transition(student, x, sigma_t.expand(x.shape[0]), sigma_s.expand(x.shape[0]), labels)
    return to_unit(x).detach(), u_grid.detach(), sigma_grid.detach()


@torch.no_grad()
def _sample_edm_teacher(
    *,
    teacher: torch.nn.Module,
    cfg: dict[str, Any],
    step_count: int,
    class_ids: list[int],
    seeds: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_cfg = cfg.get("train", {})
    sigma_min = float(train_cfg.get("sigma_min", 0.002))
    sigma_max = float(train_cfg.get("sigma_max", 80.0))
    rho = float(train_cfg.get("rho", 7.0))
    latents = _latents(
        seeds=seeds,
        channels=int(getattr(teacher, "img_channels", 3)),
        image_size=int(getattr(teacher, "img_resolution", 32)),
        device=device,
    )
    labels = _label_tensor(class_ids, label_dim=int(getattr(teacher, "label_dim", 0) or 0), device=device)
    u_grid, sigma_grid = build_sigma_grid(
        step_count=step_count,
        warp=None,
        device=device,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        net=teacher,
    )
    x = latents * sigma_grid[0].to(latents.dtype)
    for sigma_t, sigma_s in zip(sigma_grid[:-1], sigma_grid[1:]):
        x = teacher_transition(teacher, x, sigma_t.expand(x.shape[0]), sigma_s.expand(x.shape[0]), labels)
    return to_unit(x).detach(), u_grid.detach(), sigma_grid.detach()


def _load_dg_models(args: argparse.Namespace, device: torch.device):
    cfg = load_config(_resolve(args.config))
    teacher = load_edm_network(cfg["paths"]["network"], device=device, use_fp16=False)
    student = clone_student_from_teacher(teacher, cfg=cfg).to(device)
    ckpt = torch.load(_resolve(args.dg_checkpoint), map_location=device)
    student.load_state_dict(ckpt["student"])
    student.eval().requires_grad_(False)
    warp, _q_base, _q_target = init_warp(cfg, device=device)
    if warp is not None and ckpt.get("warp") is not None:
        warp.load_state_dict(ckpt["warp"])
        warp.eval()
    return cfg, teacher, student, warp, ckpt


def _sample_ctm(
    *,
    checkpoint: str | Path,
    steps: list[int],
    class_ids: list[int],
    seeds: list[int],
    output_root: Path,
    row_name: str,
    device: torch.device,
    use_fp16: bool,
) -> list[dict[str, Any]]:
    defaults = dict(DATASET_DEFAULTS["cifar10"])
    defaults["checkpoint"] = Path(checkpoint)
    model, diffusion, model_args = _load_ctm("cifar10", defaults, device=device, use_fp16=use_fp16)
    sigma_min = max(0.002, float(getattr(model_args, "sigma_min", 0.002)))
    sigma_max = min(80.0, float(getattr(model_args, "sigma_max", 80.0)))
    latents = _latents(seeds=seeds, channels=3, image_size=32, device=device) * sigma_max
    model_kwargs = {"y": torch.as_tensor(class_ids, dtype=torch.long, device=device)}
    rows = []
    for step_count in steps:
        sigmas = _karras_nodes(step_count, sigma_min=sigma_min, sigma_max=sigma_max, rho=7.0, device=device)
        x = latents.to(torch.float32)
        for sigma_from, sigma_to in zip(sigmas[:-1], sigmas[1:]):
            x = _ctm_transition(diffusion, model, x, float(sigma_from.item()), float(sigma_to.item()), model_kwargs)
        sample_dir = output_root / row_name / f"steps{step_count}"
        written = _save_tensor_images((x.clamp(-1, 1) + 1.0) / 2.0, sample_dir, seeds)
        rows.append(
            {
                "row": row_name,
                "step_count": int(step_count),
                "checkpoint": str(checkpoint),
                "sigma_grid": [float(item) for item in sigmas.detach().cpu().tolist()],
                "sample_dir": str(sample_dir.relative_to(ROOT)),
                "files": written,
            }
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = _parse_args()
    if len(args.class_ids) != len(args.latent_seeds):
        raise ValueError("--class-ids and --latent-seeds must have the same length")
    for class_id in args.class_ids:
        if class_id < 0 or class_id >= len(CIFAR10_CLASS_NAMES):
            raise ValueError(f"invalid CIFAR-10 class id: {class_id}")

    os.environ.setdefault("DNNLIB_CACHE_DIR", "/cache/Zhengwei/DG-TWFD-runtime/.torch/dnnlib")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    output_root = Path(_resolve(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    steps = [int(item) for item in args.steps]
    class_ids = [int(item) for item in args.class_ids]
    seeds = [int(item) for item in args.latent_seeds]

    manifest: dict[str, Any] = {
        "note": "Each column is class-locked and latent-seed-locked across conditional CIFAR-10 rows.",
        "steps": steps,
        "columns": [
            {
                "index": idx,
                "class_id": class_id,
                "class_name": CIFAR10_CLASS_NAMES[class_id],
                "latent_seed": seed,
            }
            for idx, (class_id, seed) in enumerate(zip(class_ids, seeds))
        ],
        "step_mapping": {
            "edm_reference_replaces_dg_twfd_best": {str(key): value for key, value in EDM_REFERENCE_STEPS.items()},
            "dg_twfd_full_learned_clock": {str(key): key for key in steps},
            "dg_twfd_identity_same_checkpoint": {str(key): key for key in steps},
            "ctm_rows": {str(key): key for key in steps},
        },
        "rows": [],
        "environment_notes": [],
    }

    cfg, teacher, student, warp, ckpt = _load_dg_models(args, device)

    for display_step in steps:
        actual_step = int(EDM_REFERENCE_STEPS[display_step])
        samples, u_grid, sigma_grid = _sample_edm_teacher(
            teacher=teacher,
            cfg=cfg,
            step_count=actual_step,
            class_ids=class_ids,
            seeds=seeds,
            device=device,
        )
        sample_dir = output_root / "edm_cifar10_cond_vp_32_48_64_128" / f"steps{display_step}"
        written = _save_tensor_images(samples, sample_dir, seeds)
        manifest["rows"].append(
            {
                "row": "edm_cifar10_cond_vp_32_48_64_128",
                "description": (
                    "Official EDM CIFAR-10 class-conditional teacher. This row replaces "
                    "the previous DG-TWFD best row for qualitative reference quality."
                ),
                "display_step": int(display_step),
                "actual_edm_steps": actual_step,
                "network": str(cfg["paths"]["network"]),
                "u_grid": [float(item) for item in u_grid.detach().cpu().tolist()],
                "sigma_grid": [float(item) for item in sigma_grid.detach().cpu().tolist()],
                "sample_dir": str(sample_dir.relative_to(ROOT)),
                "files": written,
            }
        )

    if args.include_dg_full:
        for step_count in steps:
            samples, u_grid, sigma_grid = _sample_edm_student(
                student=student,
                warp=warp,
                cfg=cfg,
                step_count=step_count,
                class_ids=class_ids,
                seeds=seeds,
                device=device,
            )
            sample_dir = output_root / "dg_twfd_full_learned_clock" / f"steps{step_count}"
            written = _save_tensor_images(samples, sample_dir, seeds)
            manifest["rows"].append(
                {
                    "row": "dg_twfd_full_learned_clock",
                    "description": "DG-TWFD student checkpoint with its learned warp/clock enabled.",
                    "step_count": int(step_count),
                    "checkpoint": str(_resolve(args.dg_checkpoint)),
                    "checkpoint_step": int(ckpt.get("step", -1)),
                    "u_grid": [float(item) for item in u_grid.detach().cpu().tolist()],
                    "sigma_grid": [float(item) for item in sigma_grid.detach().cpu().tolist()],
                    "sample_dir": str(sample_dir.relative_to(ROOT)),
                    "files": written,
                }
            )

    for step_count in steps:
        samples, u_grid, sigma_grid = _sample_edm_student(
            student=student,
            warp=None,
            cfg=cfg,
            step_count=step_count,
            class_ids=class_ids,
            seeds=seeds,
            device=device,
        )
        sample_dir = output_root / "dg_twfd_identity_same_checkpoint" / f"steps{step_count}"
        written = _save_tensor_images(samples, sample_dir, seeds)
        manifest["rows"].append(
            {
                "row": "dg_twfd_identity_same_checkpoint",
                "description": "Same DG-TWFD checkpoint with identity time grid.",
                "step_count": int(step_count),
                "checkpoint": str(_resolve(args.dg_checkpoint)),
                "checkpoint_step": int(ckpt.get("step", -1)),
                "u_grid": [float(item) for item in u_grid.detach().cpu().tolist()],
                "sigma_grid": [float(item) for item in sigma_grid.detach().cpu().tolist()],
                "sample_dir": str(sample_dir.relative_to(ROOT)),
                "files": written,
            }
        )

    del teacher, student, warp
    if device.type == "cuda":
        torch.cuda.empty_cache()

    manifest["rows"].extend(
        _sample_ctm(
            checkpoint=_resolve(args.ctm_official_checkpoint),
            steps=steps,
            class_ids=class_ids,
            seeds=seeds,
            output_root=output_root,
            row_name="ctm_official_cond",
            device=device,
            use_fp16=bool(args.use_fp16_ctm),
        )
    )
    manifest["rows"].extend(
        _sample_ctm(
            checkpoint=_resolve(args.ctm_nogan_checkpoint),
            steps=steps,
            class_ids=class_ids,
            seeds=seeds,
            output_root=output_root,
            row_name="ctm_nogan_dsm_10k",
            device=device,
            use_fp16=bool(args.use_fp16_ctm),
        )
    )
    manifest["separate_seed_only_rows"] = [
        {
            "model": "OpenAI CIFAR-10 JAX consistency models",
            "reason": "Generated by scripts/figures/generate_cifar10_consistency_jax_qualitative.py in a separate JAX environment. These checkpoints are unconditional for class labels, so they are seed-locked only.",
        }
    ]

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
