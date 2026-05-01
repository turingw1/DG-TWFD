from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from eval_edm_map import _fid_for_samples, _sample_many
from src.edm_map_lib import (
    clone_student_from_teacher,
    load_config,
    load_edm_network,
    sample_stats,
    sample_with_student,
    save_grid,
    set_seed,
    write_json,
)


class TwoStepMidpointWarp(torch.nn.Module):
    def __init__(self, midpoint: float, *, device: torch.device) -> None:
        super().__init__()
        if not 0.0 < midpoint < 1.0:
            raise ValueError(f"midpoint must be in (0, 1), got {midpoint}")
        self.register_buffer("u_nodes", torch.tensor([0.0, float(midpoint), 1.0], device=device, dtype=torch.float32))

    @torch.no_grad()
    def r_to_t(self, r: torch.Tensor) -> torch.Tensor:
        flat = r.reshape(-1).clamp(0.0, 1.0)
        scaled = flat * 2.0
        indices = torch.clamp(scaled.floor().long(), max=1)
        alpha = scaled - indices.to(dtype=flat.dtype)
        u0 = self.u_nodes.to(device=r.device, dtype=r.dtype)[indices]
        u1 = self.u_nodes.to(device=r.device, dtype=r.dtype)[indices + 1]
        return (u0 + alpha * (u1 - u0)).reshape_as(r)


def _parse_midpoints(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one midpoint is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe 2-step EDM-first FID across fixed midpoint schedules.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--midpoints", default="0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55")
    parser.add_argument("--fid-samples", type=int, default=1024)
    parser.add_argument("--fid-batch-size", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--fixed-grid-size", type=int, default=64)
    parser.add_argument("--fixed-seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_root = Path(args.output_root)
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg.get("runtime", {}).get("seed", 42)))
    device = torch.device(str(cfg.get("runtime", {}).get("device", "cuda")))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    teacher = load_edm_network(cfg["paths"]["network"], device=device, use_fp16=False)
    student = clone_student_from_teacher(teacher).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    student.load_state_dict(ckpt["student"])
    student.eval().requires_grad_(False)

    eval_cfg = cfg.get("eval", {})
    fid_samples = int(args.fid_samples)
    fid_batch_size = int(args.fid_batch_size or eval_cfg.get("fid_batch_size", 64))
    sample_batch_size = int(args.sample_batch_size or eval_cfg.get("sample_batch_size", fid_batch_size))
    fixed_grid_size = int(args.fixed_grid_size)
    fixed_seed = int(args.fixed_seed if args.fixed_seed is not None else eval_cfg.get("fixed_seed", 42))
    fid_stats = cfg["paths"]["fid_stats"]

    records = []
    for midpoint in _parse_midpoints(args.midpoints):
        midpoint_tag = f"u{midpoint:.4f}".replace(".", "p")
        step_dir = output_root / midpoint_tag
        step_dir.mkdir(parents=True, exist_ok=True)
        warp = TwoStepMidpointWarp(midpoint, device=device).eval()
        grid_samples, u_grid, sigma_grid = sample_with_student(
            student=student,
            warp=warp,
            cfg=cfg,
            step_count=2,
            batch_size=fixed_grid_size,
            device=device,
            seed=fixed_seed,
        )
        torch.save(u_grid.cpu(), step_dir / "u_grid.pt")
        torch.save(sigma_grid.cpu(), step_dir / "sigma_grid.pt")
        save_grid(grid_samples, step_dir / "fixed_seed_grid.png", nrow=max(1, int(fixed_grid_size**0.5)))
        all_samples = _sample_many(
            student=student,
            warp=warp,
            cfg=cfg,
            step_count=2,
            total=fid_samples,
            batch_size=sample_batch_size,
            device=device,
            seed=fixed_seed + 200000,
        )
        fid = _fid_for_samples(all_samples, fid_stats, device=device, batch_size=fid_batch_size)
        record = {
            "midpoint": float(midpoint),
            "fid": float(fid),
            "num_fid_samples": int(fid_samples),
            "checkpoint": str(args.checkpoint),
            "u_grid": [float(item) for item in u_grid.detach().cpu().tolist()],
            "sigma_grid": [float(item) for item in sigma_grid.detach().cpu().tolist()],
            **{f"sample_{key}": value for key, value in sample_stats(grid_samples).items() if key != "shape"},
        }
        write_json(step_dir / "metrics.json", record)
        records.append(record)
        print(f"midpoint={midpoint:.4f} fid={fid:.4f} sigma_mid={record['sigma_grid'][1]:.6f}", flush=True)

    records.sort(key=lambda item: float(item["fid"]))
    write_json(report_dir / "summary.json", {"records": records, "best": records[0]})
    with (report_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"best midpoint={records[0]['midpoint']:.4f} fid={records[0]['fid']:.4f}")


if __name__ == "__main__":
    main()
