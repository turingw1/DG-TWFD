from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

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
        "--emit-supervision-overrides",
        action="store_true",
        help="Write recommended train.py overrides based on collection smoke stats",
    )
    parser.add_argument(
        "--target-mem-util",
        type=float,
        default=0.7,
        help="Target GPU memory utilization ratio for recommended training batch size",
    )
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


@dataclass
class CollectionSummary:
    split: str
    samples: int
    shards: int
    shard_size: int
    elapsed_sec: float
    samples_per_sec: float
    peak_mem_mib: float
    total_mem_mib: float | None
    teacher_solver: str
    teacher_steps: int
    time_grid_size: int


def _round_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, value)
    return max(multiple, (value // multiple) * multiple)


def recommend_train_batch_size(
    observed_batch: int,
    peak_mem_mib: float,
    total_mem_mib: float | None,
    target_mem_util: float,
) -> int:
    if peak_mem_mib <= 0.0 or total_mem_mib is None or total_mem_mib <= 0.0:
        return max(128, observed_batch)
    target_mib = max(1.0, min(0.95, target_mem_util) * total_mem_mib)
    raw = int(observed_batch * (target_mib / peak_mem_mib) * 0.9)
    capped = max(128, min(4096, raw))
    return _round_to_multiple(capped, 64)


def write_supervision_overrides(
    output_dir: Path,
    cfg,
    summary: CollectionSummary,
    shard_root: Path,
    target_mem_util: float,
) -> None:
    recommended_batch_size = recommend_train_batch_size(
        observed_batch=summary.shard_size,
        peak_mem_mib=summary.peak_mem_mib,
        total_mem_mib=summary.total_mem_mib,
        target_mem_util=target_mem_util,
    )
    recommended_overrides = [
        "data.dataset_type='trajectory_shards'",
        f"data.trajectory_shard_dir='{str(shard_root)}'",
        f"teacher.solver='{cfg.teacher.solver}'",
        f"teacher.num_inference_steps={cfg.teacher.num_inference_steps}",
        f"data.time_grid_size={cfg.data.time_grid_size}",
        f"data.batch_size={recommended_batch_size}",
        "data.num_workers=16",
        "data.prefetch_factor=8",
        "train.warp_update_every=1",
        "boundary.enable_until_step=0",
    ]
    payload = {
        "meta": asdict(summary),
        "recommended_overrides": recommended_overrides,
    }
    output_path = output_dir / f"supervision_overrides_{summary.split}.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)
    print(f"wrote supervision overrides: {output_path}")
    cmd_path = output_dir / f"supervision_overrides_{summary.split}.txt"
    with cmd_path.open("w", encoding="utf-8") as handle:
        for override in recommended_overrides:
            handle.write(f"--override {override}\n")
    print(f"wrote supervision override flags: {cmd_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.mode, overrides=args.override)
    split_seed_offset = 0 if args.split == "train" else 1_000_000
    seed_everything(cfg.experiment.seed + split_seed_offset)
    device = resolve_device(cfg.runtime.device)
    teacher = build_teacher(cfg)
    t_grid = build_t_grid(cfg)

    output_root = Path(args.output_dir) / args.split
    output_root.mkdir(parents=True, exist_ok=True)
    num_shards = math.ceil(args.num_samples / args.shard_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
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
                "seed": cfg.experiment.seed + split_seed_offset + written + sample_index,
            }
            if "y" in payload:
                item["y"] = payload["y"][sample_index].clone()
            samples.append(item)

        shard_path = output_root / f"{args.split}_{shard_index:05d}.pt"
        torch.save(samples, shard_path)
        written += batch_size
        print(f"wrote {shard_path} ({batch_size} samples)")

    elapsed = max(1e-6, time.perf_counter() - wall_start)
    peak_mem_mib = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    total_mem_mib = (
        torch.cuda.get_device_properties(device).total_memory / (1024**2) if device.type == "cuda" else None
    )
    summary = CollectionSummary(
        split=args.split,
        samples=written,
        shards=num_shards,
        shard_size=args.shard_size,
        elapsed_sec=elapsed,
        samples_per_sec=written / elapsed,
        peak_mem_mib=peak_mem_mib,
        total_mem_mib=total_mem_mib,
        teacher_solver=cfg.teacher.solver,
        teacher_steps=cfg.teacher.num_inference_steps,
        time_grid_size=cfg.data.time_grid_size,
    )
    print(
        "done: %d samples -> %s | elapsed=%.2fs | throughput=%.2f samples/s | peak_mem=%.2f MiB"
        % (written, output_root, summary.elapsed_sec, summary.samples_per_sec, summary.peak_mem_mib)
    )
    if args.emit_supervision_overrides:
        write_supervision_overrides(
            output_dir=Path(args.output_dir),
            cfg=cfg,
            summary=summary,
            shard_root=Path(args.output_dir),
            target_mem_util=args.target_mem_util,
        )


if __name__ == "__main__":
    main()
