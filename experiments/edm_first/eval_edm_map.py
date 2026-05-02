from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch

from src.edm_map_lib import (
    clone_student_from_teacher,
    init_warp,
    load_config,
    load_edm_network,
    sample_stats,
    sample_with_student,
    save_grid,
    set_seed,
    write_json,
)


class FixedGridWarp(torch.nn.Module):
    def __init__(self, u_grid: list[float], *, device: torch.device) -> None:
        super().__init__()
        if len(u_grid) < 2:
            raise ValueError("fixed u_grid must contain at least two nodes")
        if abs(float(u_grid[0])) > 1.0e-6 or abs(float(u_grid[-1]) - 1.0) > 1.0e-6:
            raise ValueError(f"fixed u_grid must start at 0 and end at 1, got {u_grid}")
        if any(float(b) <= float(a) for a, b in zip(u_grid, u_grid[1:])):
            raise ValueError(f"fixed u_grid must be strictly increasing, got {u_grid}")
        self.register_buffer("u_nodes", torch.tensor([float(item) for item in u_grid], device=device, dtype=torch.float32))

    @torch.no_grad()
    def r_to_t(self, r: torch.Tensor) -> torch.Tensor:
        flat = r.reshape(-1).clamp(0.0, 1.0)
        interval_count = int(self.u_nodes.numel() - 1)
        scaled = flat * float(interval_count)
        indices = torch.clamp(scaled.floor().long(), max=interval_count - 1)
        alpha = scaled - indices.to(dtype=flat.dtype)
        u_nodes = self.u_nodes.to(device=r.device, dtype=r.dtype)
        restored = u_nodes[indices] + alpha * (u_nodes[indices + 1] - u_nodes[indices])
        return restored.reshape_as(r)


def _fixed_budget_grid(eval_cfg: dict, *, step_count: int) -> list[float] | None:
    raw = eval_cfg.get("budget_fixed_u_grids", {})
    if not isinstance(raw, dict):
        return None
    value = raw.get(step_count, raw.get(str(step_count)))
    if value is None:
        return None
    return [float(item) for item in value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an isolated EDM-first continuous map checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=None)
    parser.add_argument("--fid-samples", type=int, default=None)
    parser.add_argument(
        "--warp-mode",
        choices=("auto", "identity", "budget"),
        default="auto",
        help="auto loads the learned checkpoint warp; identity disables it; budget uses identity below eval.budget_warp_min_steps and learned warp at/above it.",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Evaluate checkpoint student_ema weights when available. Config eval.use_ema=true has the same effect.",
    )
    return parser.parse_args()


def _device(cfg: dict) -> torch.device:
    requested = str(cfg.get("runtime", {}).get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _fid_for_samples(samples: torch.Tensor, fid_stats_path: str | Path, *, device: torch.device, batch_size: int) -> float:
    from dgfm.evaluators.fid import InceptionFeatureExtractor, RunningStats, frechet_distance, load_stats

    feature_extractor = InceptionFeatureExtractor().to(device).eval()
    running = None
    with torch.no_grad():
        for chunk in samples.split(max(1, int(batch_size)), dim=0):
            feats = feature_extractor(chunk.to(device))
            if running is None:
                running = RunningStats(int(feats.shape[1]))
            running.update(feats)
    if running is None:
        raise ValueError("No samples provided for FID")
    fake = running.finalize()
    real = load_stats(fid_stats_path)
    return frechet_distance(real, fake)


def _sample_many(*, student, warp, cfg, step_count: int, total: int, batch_size: int, device: torch.device, seed: int) -> torch.Tensor:
    chunks = []
    produced = 0
    while produced < total:
        current = min(batch_size, total - produced)
        samples, _u, _sigma = sample_with_student(
            student=student,
            warp=warp,
            cfg=cfg,
            step_count=step_count,
            batch_size=current,
            device=device,
            seed=seed + produced,
        )
        chunks.append(samples.cpu())
        produced += current
    return torch.cat(chunks, dim=0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    eval_root = Path(args.eval_root)
    report_dir = eval_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, eval_root / "config.yaml")
    set_seed(int(cfg.get("runtime", {}).get("seed", 42)))
    device = _device(cfg)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    eval_cfg = cfg.get("eval", {})
    teacher = load_edm_network(cfg["paths"]["network"], device=device, use_fp16=False)
    student = clone_student_from_teacher(teacher, cfg=cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    use_ema = bool(args.use_ema or eval_cfg.get("use_ema", False))
    student_state_key = "student"
    if use_ema and ckpt.get("student_ema") is not None:
        student_state_key = "student_ema"
    student.load_state_dict(ckpt[student_state_key])
    student.eval().requires_grad_(False)
    warp, _q_base, _q_target = init_warp(cfg, device=device)
    timewarp_enabled = warp is not None
    checkpoint_has_warp = ckpt.get("warp") is not None
    warp_loaded = False
    if args.warp_mode == "identity":
        warp = None
    elif warp is not None and checkpoint_has_warp:
        warp.load_state_dict(ckpt["warp"])
        warp.eval()
        warp_loaded = True

    steps = [int(item) for item in (args.steps or eval_cfg.get("steps", [1, 2, 4, 8]))]
    fid_samples = int(args.fid_samples or eval_cfg.get("num_fid_samples", 1024))
    fid_batch_size = int(eval_cfg.get("fid_batch_size", 64))
    sample_batch_size = int(eval_cfg.get("sample_batch_size", fid_batch_size))
    fixed_grid_size = int(eval_cfg.get("fixed_grid_size", 64))
    fixed_seed = int(eval_cfg.get("fixed_seed", 42))
    budget_warp_min_steps = int(eval_cfg.get("budget_warp_min_steps", cfg.get("timewarp", {}).get("budget_warp_min_steps", 4)))
    fid_stats = cfg["paths"]["fid_stats"]

    records = []
    for step_count in steps:
        effective_warp = warp
        effective_warp_mode = args.warp_mode
        if args.warp_mode == "budget":
            fixed_u_grid = _fixed_budget_grid(eval_cfg, step_count=step_count)
            if fixed_u_grid is not None:
                if len(fixed_u_grid) != step_count + 1:
                    raise ValueError(
                        f"eval.budget_fixed_u_grids[{step_count}] must have {step_count + 1} nodes, got {len(fixed_u_grid)}"
                    )
                effective_warp = FixedGridWarp(fixed_u_grid, device=device).eval()
                effective_warp_mode = "fixed"
            elif step_count < budget_warp_min_steps or not warp_loaded:
                effective_warp = None
                effective_warp_mode = "identity"
            else:
                effective_warp_mode = "auto"
        step_dir = eval_root / f"steps{step_count}"
        step_dir.mkdir(parents=True, exist_ok=True)
        grid_samples, u_grid, sigma_grid = sample_with_student(
            student=student,
            warp=effective_warp,
            cfg=cfg,
            step_count=step_count,
            batch_size=fixed_grid_size,
            device=device,
            seed=fixed_seed,
        )
        torch.save(grid_samples.cpu(), step_dir / "fixed_seed_samples.pt")
        torch.save(u_grid.cpu(), step_dir / "u_grid.pt")
        torch.save(sigma_grid.cpu(), step_dir / "sigma_grid.pt")
        save_grid(grid_samples, step_dir / "fixed_seed_grid.png", nrow=max(1, int(fixed_grid_size**0.5)))

        all_samples = _sample_many(
            student=student,
            warp=effective_warp,
            cfg=cfg,
            step_count=step_count,
            total=fid_samples,
            batch_size=sample_batch_size,
            device=device,
            seed=fixed_seed + 100000 * step_count,
        )
        fid = _fid_for_samples(all_samples, fid_stats, device=device, batch_size=fid_batch_size)
        stats = sample_stats(grid_samples)
        record = {
            "step_count": step_count,
            "nfe": step_count,
            "fid": fid,
            "num_fid_samples": fid_samples,
            "checkpoint": str(args.checkpoint),
            "warp_mode": args.warp_mode,
            "effective_warp_mode": effective_warp_mode,
            "budget_warp_min_steps": budget_warp_min_steps if args.warp_mode == "budget" else None,
            "timewarp_enabled": timewarp_enabled,
            "checkpoint_has_warp": checkpoint_has_warp,
            "warp_loaded": warp_loaded,
            "use_ema": use_ema,
            "student_state_key": student_state_key,
            "u_grid": [float(item) for item in u_grid.detach().cpu().tolist()],
            "sigma_grid": [float(item) for item in sigma_grid.detach().cpu().tolist()],
            **{f"sample_{key}": value for key, value in stats.items() if key != "shape"},
        }
        write_json(step_dir / "metrics.json", record)
        records.append(record)
        print(f"eval step={step_count} fid={fid:.4f} corr_h={record['sample_neighbor_corr_h']:.4f}", flush=True)

    write_json(
        report_dir / "summary.json",
        {
            "checkpoint": str(args.checkpoint),
            "warp_mode": args.warp_mode,
            "timewarp_enabled": timewarp_enabled,
            "checkpoint_has_warp": checkpoint_has_warp,
            "warp_loaded": warp_loaded,
            "use_ema": use_ema,
            "student_state_key": student_state_key,
            "records": records,
        },
    )
    with (report_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    best = min(records, key=lambda item: float(item["fid"]))
    write_json(report_dir / "best.json", best)
    print(f"edm-first evaluation completed: {eval_root}")


if __name__ == "__main__":
    main()
