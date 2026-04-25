from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgfm.config import load_experiment_config
from dgfm.datasets import build_image_dataloaders
from dgtd.train_dgtd import _device_from_config, _extract_online_teacher_batch
from dgtd.teacher import build_teacher_adapter


def _tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    x = x.detach().float().cpu()
    flat = x.flatten()
    return {
        "shape": list(x.shape),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def _teacher_range_input(config: dict, images: torch.Tensor) -> torch.Tensor:
    teacher_cfg = config.get("teacher", {})
    input_range = str(teacher_cfg.get("clean_input_range", "unit"))
    if input_range == "unit":
        return torch.clamp(images, 0.0, 1.0) * 2.0 - 1.0
    if input_range == "minus_one_one":
        return torch.clamp(images, -1.0, 1.0)
    raise ValueError(f"Unsupported teacher.clean_input_range: {input_range}")


def _mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x.detach().float() - y.detach().float()).flatten(1).square().mean(dim=1).mean().item())


def _condition_summary(cond: torch.Tensor | None) -> dict[str, Any]:
    if cond is None:
        return {"available": False}
    cond_cpu = cond.detach().cpu()
    unique = torch.unique(cond_cpu)
    return {
        "available": True,
        "shape": list(cond_cpu.shape),
        "dtype": str(cond_cpu.dtype),
        "unique_count": int(unique.numel()),
        "first_values": [int(item) for item in cond_cpu.view(-1)[:16].tolist()],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose online teacher trajectory endpoint semantics.")
    parser.add_argument("--config", required=True, help="Experiment config path.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="Dataloader split to inspect.")
    parser.add_argument("--batch-size", type=int, default=8, help="Temporary batch size for endpoint diagnosis.")
    parser.add_argument("--seed", type=int, default=42, help="Torch RNG seed.")
    parser.add_argument("--set", action="append", default=[], help="Config override in key=value form.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    overrides.extend(
        [
            f"train.batch_size={args.batch_size}",
            "train.num_workers=0",
            "train.persistent_workers=false",
        ]
    )
    config = load_experiment_config(args.config, overrides=overrides)
    device = _device_from_config(config)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    teacher_adapter = build_teacher_adapter(config)
    teacher_adapter.prepare(device)
    if not teacher_adapter.has_online_teacher():
        raise RuntimeError("Online teacher is not available for endpoint diagnosis.")

    dataloaders = build_image_dataloaders(config)
    if args.split not in dataloaders:
        raise ValueError(f"Split not available: {args.split}")
    raw_batch = next(iter(dataloaders[args.split]))
    images, cond = _extract_online_teacher_batch(raw_batch, device)

    start = time.perf_counter()
    trajectory = teacher_adapter.online_trajectory_from_x0(images, cond=cond, device=device)
    latency_sec = time.perf_counter() - start

    states = trajectory["states"].to(device=device)
    times = trajectory["times"].to(device=device)
    if states.ndim != 5:
        raise ValueError(f"Expected batched teacher states [B,M,C,H,W], got {tuple(states.shape)}")
    u0 = states[:, 0]
    u1 = states[:, -1]
    images_teacher_range = _teacher_range_input(config, images).to(device=device, dtype=u0.dtype)

    mse_input_u0 = _mse(images_teacher_range, u0)
    mse_input_u1 = _mse(images_teacher_range, u1)
    verdict = {
        "u0_is_not_clean_input": mse_input_u0 > 1.0e-4,
        "u1_closer_to_clean_than_u0": mse_input_u1 < mse_input_u0,
        "endpoint_order_ok": bool(torch.all(times[:, 1:] >= times[:, :-1]).item()) if times.ndim == 2 else bool(torch.all(times[1:] >= times[:-1]).item()),
    }
    verdict["pass"] = all(bool(value) for value in verdict.values())

    report = {
        "config": args.config,
        "split": args.split,
        "batch_size": int(images.shape[0]),
        "device": str(device),
        "teacher": {
            "type": config.get("teacher", {}).get("type"),
            "backend": config.get("teacher", {}).get("backend"),
            "name_or_path": config.get("teacher", {}).get("name_or_path"),
            "clean_input_range": config.get("teacher", {}).get("clean_input_range"),
            "retain_num_points": config.get("teacher", {}).get("retain_num_points"),
        },
        "latency_sec": latency_sec,
        "times": times.detach().cpu().tolist(),
        "input_unit_stats": _tensor_stats(images),
        "input_teacher_range_stats": _tensor_stats(images_teacher_range),
        "u0_stats": _tensor_stats(u0),
        "u1_stats": _tensor_stats(u1),
        "mse_input_u0": mse_input_u0,
        "mse_input_u1": mse_input_u1,
        "condition": _condition_summary(cond),
        "verdict": verdict,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    print(f"teacher endpoint report: {output}")


if __name__ == "__main__":
    main()
