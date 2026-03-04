from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dg_twfd.config import load_config
from dg_twfd.data import build_teacher
from dg_twfd.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect teacher trajectories into shard files")
    parser.add_argument("--mode", default="debug_4060", help="Config profile name")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split")
    parser.add_argument("--num-samples", type=int, required=True, help="Number of trajectories to collect")
    parser.add_argument("--shard-size", type=int, default=64, help="Samples per shard")
    parser.add_argument("--output-dir", required=True, help="Root output directory for shards")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional config overrides in key=value form",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_t_grid(cfg) -> torch.Tensor:
    return torch.linspace(1.0, 0.0, steps=cfg.data.time_grid_size, dtype=torch.float32)


def maybe_sample_labels(cfg, batch_size: int, device: torch.device) -> torch.Tensor | None:
    if not cfg.teacher.class_cond:
        return None
    return torch.randint(0, cfg.teacher.num_classes, (batch_size,), device=device)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.mode, overrides=args.override)
    seed_everything(cfg.experiment.seed)
    device = resolve_device(cfg.runtime.device)
    teacher = build_teacher(cfg)
    t_grid = build_t_grid(cfg)

    output_root = Path(args.output_dir) / args.split
    output_root.mkdir(parents=True, exist_ok=True)
    num_shards = math.ceil(args.num_samples / args.shard_size)
    written = 0
    for shard_index in range(num_shards):
        remaining = args.num_samples - written
        batch_size = min(args.shard_size, remaining)
        labels = maybe_sample_labels(cfg, batch_size, device)
        payload = teacher.sample_trajectory(
            batch_size=batch_size,
            t_grid=t_grid,
            device=device,
            labels=labels,
        )
        samples = []
        for sample_index in range(batch_size):
            item = {
                "t_grid": payload["t_grid"].clone(),
                "x_grid": payload["x_grid"][sample_index].clone(),
                "x0": payload["x0"][sample_index].clone(),
                "seed": cfg.experiment.seed + written + sample_index,
            }
            if "y" in payload:
                item["y"] = payload["y"][sample_index].clone()
            samples.append(item)

        shard_path = output_root / f"{args.split}_{shard_index:05d}.pt"
        torch.save(samples, shard_path)
        written += batch_size
        print(f"wrote {shard_path} ({batch_size} samples)")

    print(f"done: {written} samples -> {output_root}")


if __name__ == "__main__":
    main()
