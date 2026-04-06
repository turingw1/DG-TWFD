from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgfm.config import load_experiment_config
from dgfm.teachers import build_teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare offline teacher trajectories for dgfm map branch")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--output-root", default=None, help="Override target.shard_root")
    parser.add_argument("--batch-size", type=int, default=64, help="Teacher rollout batch size")
    parser.add_argument("--set", action="append", default=[], help="Config override in key=value form")
    return parser.parse_args()


def _storage_dtype(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered == "float16":
        return torch.float16
    if lowered == "float32":
        return torch.float32
    raise ValueError(f"Unsupported teacher.store_dtype: {name}")


def _shard_payload(batch, start_index: int, store_dtype: torch.dtype) -> list[dict]:
    payload = []
    t_grid = batch.t_grid.float().cpu()
    for item_idx in range(batch.x_grid.shape[0]):
        payload.append(
            {
                "sample_id": start_index + item_idx,
                "t_grid": t_grid.clone(),
                "x_grid": batch.x_grid[item_idx].to(dtype=store_dtype).cpu(),
            }
        )
    return payload


def _write_manifest(split_root: Path, split_name: str, total_count: int, shard_counts: list[tuple[str, int]], config: dict) -> None:
    teacher_cfg = config.get("teacher", {})
    manifest = {
        "split": split_name,
        "count": total_count,
        "teacher": {
            "type": teacher_cfg.get("type", "sampler"),
            "backend": teacher_cfg.get("backend", "diffusers_ddpm"),
            "name_or_path": teacher_cfg.get("name_or_path"),
            "solver": teacher_cfg.get("solver", "ddim"),
            "num_inference_steps": int(teacher_cfg.get("num_inference_steps", 128)),
            "retain_num_points": int(teacher_cfg.get("retain_num_points", 33)),
            "time_semantics": "dgfm_u_grid_ascending_0_noise_1_clean",
        },
        "shards": [{"file": file_name, "count": count} for file_name, count in shard_counts],
    }
    with (split_root / "manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)


def _generate_split(split_name: str, num_samples: int, batch_size: int, output_root: Path, config: dict, teacher, device: torch.device) -> None:
    split_root = output_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)
    teacher_cfg = config.get("teacher", {})
    retain_num_points = int(teacher_cfg.get("retain_num_points", 33))
    shard_size = int(teacher_cfg.get("shard_size", 1024))
    store_dtype = _storage_dtype(str(teacher_cfg.get("store_dtype", "float16")))
    u_grid = torch.linspace(0.0, 1.0, steps=retain_num_points, dtype=torch.float32)

    shard_items: list[dict] = []
    shard_counts: list[tuple[str, int]] = []
    sample_index = 0
    shard_index = 0
    while sample_index < num_samples:
        current_batch = min(batch_size, num_samples - sample_index)
        batch = teacher.sample_trajectory(batch_size=current_batch, u_grid=u_grid, device=device)
        shard_items.extend(_shard_payload(batch, start_index=sample_index, store_dtype=store_dtype))
        sample_index += current_batch
        while len(shard_items) >= shard_size:
            shard_path = split_root / f"{split_name}_{shard_index:05d}.pt"
            to_save = shard_items[:shard_size]
            torch.save(to_save, shard_path)
            shard_counts.append((shard_path.name, len(to_save)))
            shard_items = shard_items[shard_size:]
            shard_index += 1
        print(f"{split_name}: generated {sample_index}/{num_samples} teacher trajectories", flush=True)

    if shard_items:
        shard_path = split_root / f"{split_name}_{shard_index:05d}.pt"
        torch.save(shard_items, shard_path)
        shard_counts.append((shard_path.name, len(shard_items)))

    _write_manifest(split_root, split_name, total_count=num_samples, shard_counts=shard_counts, config=config)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, overrides=args.set)
    output_root = Path(args.output_root or config.get("target", {}).get("shard_root", ""))
    if not str(output_root):
        raise ValueError("target.shard_root must be set or --output-root must be provided")
    output_root.mkdir(parents=True, exist_ok=True)

    requested = str(config.get("runtime", {}).get("device", "auto"))
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    teacher = build_teacher(config)
    teacher.prepare(device)

    teacher_cfg = config.get("teacher", {})
    _generate_split(
        split_name="train",
        num_samples=int(teacher_cfg.get("train_num_samples", 50000)),
        batch_size=args.batch_size,
        output_root=output_root,
        config=config,
        teacher=teacher,
        device=device,
    )
    _generate_split(
        split_name="val",
        num_samples=int(teacher_cfg.get("val_num_samples", 10000)),
        batch_size=args.batch_size,
        output_root=output_root,
        config=config,
        teacher=teacher,
        device=device,
    )
    print(f"teacher trajectories ready at {output_root}", flush=True)


if __name__ == "__main__":
    main()
