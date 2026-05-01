#!/usr/bin/env python3
"""Generate class-locked ImageNet64 qualitative samples.

The OpenAI ImageNet64 consistency baselines are class-conditional, so this
script fixes the same class-id vector for EDM, CD-LPIPS, CD-L2, and CT rows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[2]
EDM_FIRST_SRC = ROOT / "experiments" / "edm_first"
EDM_REF = ROOT / "refs" / "edm"
CM_ROOT = ROOT / "refs" / "consistency_models"

for path in [str(EDM_FIRST_SRC), str(EDM_REF), str(CM_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# The OpenAI consistency_models helper imports mpi4py at module import time.
# Qualitative figure generation is strictly single-process, and this sandbox
# can block MPI socket initialization, so provide the tiny subset it uses.
if "mpi4py" not in sys.modules:
    class _FakeComm:
        rank = 0
        size = 1

        def Get_rank(self) -> int:
            return 0

        def Get_size(self) -> int:
            return 1

        def bcast(self, value, root: int = 0):
            return value

    fake_mpi_module = types.ModuleType("mpi4py")
    fake_mpi = types.SimpleNamespace(COMM_WORLD=_FakeComm())
    fake_mpi_module.MPI = fake_mpi
    sys.modules["mpi4py"] = fake_mpi_module
    sys.modules["mpi4py.MPI"] = fake_mpi

from src.edm_map_lib import build_sigma_grid, load_edm_network, make_labels, teacher_transition, to_unit  # noqa: E402
from cm.karras_diffusion import karras_sample  # noqa: E402
from cm.random_util import get_generator  # noqa: E402
from cm.script_util import create_model_and_diffusion, model_and_diffusion_defaults  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edm-network",
        default="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
    )
    parser.add_argument(
        "--cd-lpips-checkpoint",
        default="/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/consistency_models/cd_imagenet64_lpips.pt",
    )
    parser.add_argument(
        "--cd-l2-checkpoint",
        default="/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/consistency_models/cd_imagenet64_l2.pt",
    )
    parser.add_argument(
        "--ct-checkpoint",
        default="/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/consistency_models/ct_imagenet64.pt",
    )
    parser.add_argument("--output-root", default="docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/imagenet64_20260501")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--class-ids",
        nargs="+",
        type=int,
        default=[8, 22, 207, 281, 404, 555, 751, 817],
        help="One ImageNet class id per visual column; names are intentionally kept out of the image-only PDF.",
    )
    parser.add_argument("--seed", type=int, default=31, help="Deterministic latent/noise seed shared across rows.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _resolve(path: str | Path) -> Path | str:
    raw = str(path)
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    value = Path(raw)
    return value if value.is_absolute() else ROOT / value


def _save_tensor_images(samples: torch.Tensor, sample_dir: Path) -> list[str]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    x = samples.detach().cpu()
    if float(x.min()) < -0.01:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    written = []
    for idx, image in enumerate(x):
        arr = (image.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
        path = sample_dir / f"{idx:06d}.png"
        Image.fromarray(arr, mode="RGB").save(path)
        written.append(str(path.relative_to(ROOT)))
    return written


def _ts_for_step_count(step_count: int, *, cm_steps_total: int) -> tuple[int, ...] | None:
    if step_count <= 1:
        return None
    if step_count == 2 and cm_steps_total == 40:
        return (0, 22, 39)
    if step_count == 2 and cm_steps_total == 201:
        return (0, 106, 200)
    values = np.rint(np.linspace(0, cm_steps_total - 1, step_count + 1)).astype(int).tolist()
    deduped: list[int] = []
    for value in values:
        if not deduped or value != deduped[-1]:
            deduped.append(value)
    deduped[0] = 0
    deduped[-1] = cm_steps_total - 1
    return tuple(int(item) for item in deduped)


def _load_cm_model(*, checkpoint: str | Path, training_mode: str, device: torch.device, use_fp16: bool):
    defaults = model_and_diffusion_defaults()
    defaults.update(
        {
            "attention_resolutions": "32,16,8",
            "class_cond": True,
            "dropout": 0.0,
            "image_size": 64,
            "num_channels": 192,
            "num_head_channels": 64,
            "num_res_blocks": 3,
            "resblock_updown": True,
            "use_fp16": bool(use_fp16 and device.type == "cuda"),
            "use_scale_shift_norm": True,
            "weight_schedule": "uniform",
        }
    )
    model, diffusion = create_model_and_diffusion(
        **defaults,
        distillation=("consistency" in training_mode),
    )
    state = torch.load(_resolve(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    if bool(use_fp16 and device.type == "cuda"):
        model.convert_to_fp16()
    model.eval()
    return model, diffusion


@torch.no_grad()
def _sample_cm(
    *,
    model: torch.nn.Module,
    diffusion,
    step_count: int,
    class_ids: list[int],
    seed: int,
    device: torch.device,
    cm_steps_total: int,
) -> torch.Tensor:
    labels = torch.as_tensor(class_ids, dtype=torch.long, device=device)
    generator = get_generator("determ-indiv", num_samples=len(class_ids), seed=int(seed))
    sampler = "onestep" if step_count <= 1 else "multistep"
    return karras_sample(
        diffusion,
        model,
        (len(class_ids), 3, 64, 64),
        steps=cm_steps_total if step_count > 1 else 40,
        model_kwargs={"y": labels},
        device=device,
        clip_denoised=True,
        sampler=sampler,
        sigma_min=0.002,
        sigma_max=80.0,
        generator=generator,
        ts=_ts_for_step_count(step_count, cm_steps_total=cm_steps_total),
    )


@torch.no_grad()
def _sample_edm(
    *,
    net: torch.nn.Module,
    step_count: int,
    class_ids: list[int],
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = get_generator("determ-indiv", num_samples=len(class_ids), seed=int(seed))
    latents = generator.randn(len(class_ids), 3, 64, 64, device=device)
    labels = make_labels(torch.as_tensor(class_ids, dtype=torch.long, device=device), label_dim=int(net.label_dim), device=device)
    _u_grid, sigma_grid = build_sigma_grid(
        step_count=step_count,
        warp=None,
        device=device,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        net=net,
    )
    x = latents * sigma_grid[0].to(latents.dtype)
    for sigma_t, sigma_s in zip(sigma_grid[:-1], sigma_grid[1:]):
        x = teacher_transition(net, x, sigma_t.expand(x.shape[0]), sigma_s.expand(x.shape[0]), labels)
    return to_unit(x).detach(), sigma_grid.detach()


def main() -> None:
    args = _parse_args()
    for class_id in args.class_ids:
        if class_id < 0 or class_id >= 1000:
            raise ValueError(f"invalid ImageNet class id: {class_id}")
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
    manifest: dict[str, Any] = {
        "note": "Each column fixes the same ImageNet class id and deterministic latent/noise seed across class-conditional rows.",
        "steps": steps,
        "columns": [{"index": idx, "class_id": class_id, "seed": int(args.seed)} for idx, class_id in enumerate(class_ids)],
        "rows": [],
    }

    edm = load_edm_network(args.edm_network, device=device, use_fp16=bool(args.use_fp16 and device.type == "cuda"))
    for step_count in steps:
        samples, sigma_grid = _sample_edm(net=edm, step_count=step_count, class_ids=class_ids, seed=args.seed, device=device)
        sample_dir = output_root / "edm_imagenet64_cond_adm" / f"steps{step_count}"
        manifest["rows"].append(
            {
                "row": "edm_imagenet64_cond_adm",
                "step_count": int(step_count),
                "network": args.edm_network,
                "sigma_grid": [float(item) for item in sigma_grid.detach().cpu().tolist()],
                "sample_dir": str(sample_dir.relative_to(ROOT)),
                "files": _save_tensor_images(samples, sample_dir),
            }
        )
    del edm
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = [
        ("cd_lpips_imagenet64", args.cd_lpips_checkpoint, "consistency_distillation", 40),
        ("cd_l2_imagenet64", args.cd_l2_checkpoint, "consistency_distillation", 40),
        ("ct_imagenet64", args.ct_checkpoint, "consistency_training", 201),
    ]
    for row_name, checkpoint, training_mode, cm_steps_total in rows:
        model, diffusion = _load_cm_model(
            checkpoint=checkpoint,
            training_mode=training_mode,
            device=device,
            use_fp16=bool(args.use_fp16),
        )
        for step_count in steps:
            samples = _sample_cm(
                model=model,
                diffusion=diffusion,
                step_count=step_count,
                class_ids=class_ids,
                seed=args.seed,
                device=device,
                cm_steps_total=int(cm_steps_total),
            )
            sample_dir = output_root / row_name / f"steps{step_count}"
            manifest["rows"].append(
                {
                    "row": row_name,
                    "step_count": int(step_count),
                    "checkpoint": str(_resolve(checkpoint)),
                    "training_mode": training_mode,
                    "cm_steps_total": int(cm_steps_total),
                    "ts": None if step_count <= 1 else list(_ts_for_step_count(step_count, cm_steps_total=int(cm_steps_total))),
                    "sample_dir": str(sample_dir.relative_to(ROOT)),
                    "files": _save_tensor_images((samples + 1.0) / 2.0, sample_dir),
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
