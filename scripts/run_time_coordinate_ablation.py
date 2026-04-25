from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgfm.config import load_experiment_config
from dgfm.evaluators import build_evaluator
from dgfm.evaluators.common import (
    device_from_config,
    load_model_from_checkpoint,
    load_timewarp_from_checkpoint,
    sample_condition_labels,
)
from dgtd.sample_dgtd import save_schedule, search_oss_time_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run time-coordinate design ablation: learned timewarp vs OSS-like schedules.")
    parser.add_argument("--config", required=True, help="Experiment config path matching the checkpoint architecture.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to compare under both schedule policies.")
    parser.add_argument("--output-root", required=True, help="Independent output directory for this ablation.")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8], help="Step counts to compare.")
    parser.add_argument("--fid-samples", type=int, default=1024, help="FID sample count for both policies.")
    parser.add_argument("--fid-batch-size", type=int, default=64, help="FID batch size for both policies.")
    parser.add_argument("--sample-batch-size", type=int, default=64, help="Sampling batch size for both policies.")
    parser.add_argument("--fixed-grid-size", type=int, default=64, help="Fixed-seed qualitative grid size.")
    parser.add_argument("--dump-image-count", type=int, default=64, help="How many fixed-seed images to dump per step.")
    parser.add_argument("--reference-steps", type=int, default=32, help="Dense rollout grid size for OSS-like search.")
    parser.add_argument("--search-batch-size", type=int, default=256, help="Noise batch size for OSS-like schedule search.")
    parser.add_argument("--search-cost-batch-size", type=int, default=64, help="Chunk size for OSS pairwise interval cost.")
    parser.add_argument("--search-seed", type=int, default=123, help="Seed for OSS-like schedule search noise.")
    parser.add_argument("--fixed-seed", type=int, default=42, help="Seed for qualitative fixed-grid generation.")
    parser.add_argument("--force-search", action="store_true", help="Regenerate schedule JSON even if it already exists.")
    parser.add_argument("--set", action="append", default=[], help="Config override in key=value form.")
    return parser.parse_args()


def _git_head(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _identity_schedule_payload(step_count: int) -> dict[str, Any]:
    if step_count != 1:
        raise ValueError(f"identity schedule helper only supports step_count=1, got {step_count}")
    return {
        "mode": "oss_schedule_search",
        "status": "identity_single_step",
        "cost": "not_applicable",
        "step_count": 1,
        "reference_steps": 1,
        "search_batch_size": 0,
        "reference_source": "identity_single_step",
        "indices": [0, 1],
        "reference_time_grid": [0.0, 1.0],
        "time_grid": [0.0, 1.0],
        "interval_costs": [0.0],
        "total_cost": 0.0,
        "mean_interval_cost": 0.0,
        "schedule_time_grid": [0.0, 1.0],
        "schedule_delta_min": 1.0,
        "schedule_delta_max": 1.0,
        "schedule_delta_mean": 1.0,
        "schedule_delta_std": 0.0,
    }


def _eval_overrides(args: argparse.Namespace, *, schedule_dir: Path | None = None) -> list[str]:
    overrides = list(args.set)
    overrides.extend(
        [
            f"eval.num_fid_samples={int(args.fid_samples)}",
            f"eval.fid_batch_size={int(args.fid_batch_size)}",
            f"eval.sample_batch_size={int(args.sample_batch_size)}",
            f"eval.fixed_grid_size={int(args.fixed_grid_size)}",
            f"eval.fixed_grid_batch_size={int(args.sample_batch_size)}",
            f"eval.dump_image_count={int(args.dump_image_count)}",
            f"eval.fixed_seed={int(args.fixed_seed)}",
        ]
    )
    if schedule_dir is not None:
        overrides.append(f"eval.time_grid_dir={schedule_dir}")
    return overrides


def _search_schedule_bundle(args: argparse.Namespace, output_root: Path) -> tuple[Path, dict[int, dict[str, Any]]]:
    config = load_experiment_config(args.config, overrides=list(args.set))
    device = device_from_config(config)
    checkpoint = Path(args.checkpoint)
    model = load_model_from_checkpoint(config, checkpoint, device=device)
    timewarp = load_timewarp_from_checkpoint(config, checkpoint, device=device)
    if timewarp is None:
        raise RuntimeError("This ablation expects a checkpoint with a learned timewarp module.")

    steps = sorted(set(int(step) for step in args.steps))
    schedule_dir = output_root / "oss_schedules"
    schedule_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(int(args.search_seed))
    channels = int(config["dataset"]["channels"])
    image_size = int(config["dataset"]["image_size"])
    search_noise = torch.randn(
        int(args.search_batch_size),
        channels,
        image_size,
        image_size,
        generator=generator,
        device=device,
    )
    search_labels = sample_condition_labels(config, int(args.search_batch_size), device=device, generator=generator)
    search_extra = {"label": search_labels} if search_labels is not None else None

    payloads: dict[int, dict[str, Any]] = {}
    for step_count in steps:
        schedule_path = schedule_dir / f"oss_schedule_steps{step_count}.json"
        if schedule_path.exists() and not args.force_search:
            payload = json.loads(schedule_path.read_text(encoding="utf-8"))
            payloads[step_count] = payload
            continue

        if step_count == 1:
            payload = _identity_schedule_payload(step_count)
            save_schedule(payload, schedule_path)
            payloads[step_count] = payload
            continue

        search_t0 = time.time()
        _, payload = search_oss_time_grid(
            model=model,
            x_search=search_noise,
            step_count=step_count,
            reference_steps=int(args.reference_steps),
            warp=timewarp,
            extra=search_extra,
            cost_batch_size=int(args.search_cost_batch_size),
        )
        payload["search_elapsed_sec"] = time.time() - search_t0
        payload["schedule_source"] = "optimalsteps_like_from_refs"
        payload["schedule_ref_repo"] = "refs/optimalsteps"
        save_schedule(payload, schedule_path)
        payloads[step_count] = payload

    return schedule_dir, payloads


def _run_eval(config_path: str, checkpoint: str, eval_root: Path, steps: list[int], overrides: list[str]) -> None:
    config = load_experiment_config(config_path, overrides=overrides)
    runner = build_evaluator(config=config, checkpoint=Path(checkpoint), eval_root=eval_root)
    runner.run(step_counts=steps)


def _load_summary(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected list summary at {path}, got {type(payload).__name__}")
    return payload


def _comparison_rows(
    *,
    steps: list[int],
    default_summary: list[dict[str, Any]],
    oss_summary: list[dict[str, Any]],
    default_eval_root: Path,
    oss_eval_root: Path,
    schedule_dir: Path,
) -> list[dict[str, Any]]:
    default_by_step = {int(row["step_count"]): row for row in default_summary}
    oss_by_step = {int(row["step_count"]): row for row in oss_summary}
    rows: list[dict[str, Any]] = []
    for step_count in steps:
        if step_count not in default_by_step:
            raise KeyError(f"Default eval missing step_count={step_count}")
        if step_count not in oss_by_step:
            raise KeyError(f"OSS eval missing step_count={step_count}")
        default_row = default_by_step[step_count]
        oss_row = oss_by_step[step_count]
        schedule_path = schedule_dir / f"oss_schedule_steps{step_count}.json"
        schedule_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "step_count": step_count,
                "timewarp_fid": float(default_row["fid"]),
                "optimalsteps_fid": float(oss_row["fid"]),
                "delta_fid_optimalsteps_minus_timewarp": float(oss_row["fid"]) - float(default_row["fid"]),
                "timewarp_grid_png": str(default_eval_root / f"steps{step_count}" / "fixed_seed_grid.png"),
                "optimalsteps_grid_png": str(oss_eval_root / f"steps{step_count}" / "fixed_seed_grid.png"),
                "schedule_json": str(schedule_path),
                "schedule_status": str(schedule_payload.get("status")),
                "schedule_reference_source": str(schedule_payload.get("reference_source")),
                "schedule_time_grid": schedule_payload.get("time_grid"),
                "schedule_total_cost": schedule_payload.get("total_cost"),
                "schedule_mean_interval_cost": schedule_payload.get("mean_interval_cost"),
                "num_fid_samples": int(default_row["num_fid_samples"]),
            }
        )
    return rows


def _acceptance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    for row in rows:
        step_count = int(row["step_count"])
        if not Path(row["timewarp_grid_png"]).exists():
            failures.append(f"timewarp grid missing for step_count={step_count}")
        if not Path(row["optimalsteps_grid_png"]).exists():
            failures.append(f"optimalsteps grid missing for step_count={step_count}")
        if not Path(row["schedule_json"]).exists():
            failures.append(f"schedule json missing for step_count={step_count}")
        time_grid = row.get("schedule_time_grid") or []
        if len(time_grid) != step_count + 1:
            failures.append(f"schedule time_grid length mismatch for step_count={step_count}")
        if step_count == 1 and time_grid != [0.0, 1.0]:
            failures.append("step_count=1 OSS schedule must be the identity [0.0, 1.0]")
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": {
            "all_steps_have_fid": True,
            "all_steps_have_sample_grids": not any("grid missing" in item for item in failures),
            "all_steps_have_schedule_json": not any("schedule json missing" in item for item in failures),
            "single_step_identity": not any("step_count=1 OSS schedule" in item for item in failures),
        },
    }


def _write_comparison_files(
    output_root: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    acceptance: dict[str, Any],
) -> None:
    csv_lines = [
        "step_count,timewarp_fid,optimalsteps_fid,delta_fid_optimalsteps_minus_timewarp,num_fid_samples,schedule_status,schedule_reference_source,schedule_total_cost,schedule_mean_interval_cost,schedule_json,timewarp_grid_png,optimalsteps_grid_png"
    ]
    for row in rows:
        csv_lines.append(
            ",".join(
                [
                    str(row["step_count"]),
                    f"{row['timewarp_fid']:.6f}",
                    f"{row['optimalsteps_fid']:.6f}",
                    f"{row['delta_fid_optimalsteps_minus_timewarp']:.6f}",
                    str(row["num_fid_samples"]),
                    str(row["schedule_status"]),
                    str(row["schedule_reference_source"]),
                    str(row["schedule_total_cost"]),
                    str(row["schedule_mean_interval_cost"]),
                    str(row["schedule_json"]),
                    str(row["timewarp_grid_png"]),
                    str(row["optimalsteps_grid_png"]),
                ]
            )
        )
    _write_text(output_root / "comparison.csv", "\n".join(csv_lines) + "\n")

    table_lines = [
        "# Time-Coordinate Design Ablation",
        "",
        "This report compares the learned default timewarp against an OSS-like schedule baseline adapted from `refs/optimalsteps` on the same checkpoint.",
        "",
        f"- Config: `{metadata['config']}`",
        f"- Checkpoint: `{metadata['checkpoint']}`",
        f"- Refs/optimalsteps HEAD: `{metadata.get('optimalsteps_head')}`",
        f"- Requested steps: `{metadata['steps']}`",
        f"- FID samples: `{metadata['fid_samples']}`",
        f"- Acceptance: `{acceptance['status']}`",
        "",
        "| Steps | Timewarp FID | OptimalSteps-like FID | Delta (OSS - Timewarp) | OSS Schedule Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['step_count']} | {row['timewarp_fid']:.3f} | {row['optimalsteps_fid']:.3f} | "
            f"{row['delta_fid_optimalsteps_minus_timewarp']:+.3f} | {row['schedule_status']} |"
        )
    table_lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for row in rows:
        table_lines.extend(
            [
                f"### Steps = {row['step_count']}",
                "",
                f"- Timewarp grid: `{row['timewarp_grid_png']}`",
                f"- OptimalSteps-like grid: `{row['optimalsteps_grid_png']}`",
                f"- OSS schedule JSON: `{row['schedule_json']}`",
                f"- OSS time grid: `{row['schedule_time_grid']}`",
                "",
            ]
        )
    _write_text(output_root / "comparison.md", "\n".join(table_lines))

    _write_json(output_root / "comparison.json", rows)
    _write_json(output_root / "acceptance.json", acceptance)
    _write_json(output_root / "metadata.json", metadata)


def main() -> None:
    args = parse_args()
    steps = sorted(set(int(step) for step in args.steps))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint = str(Path(args.checkpoint).resolve())
    config_path = str(Path(args.config))
    default_eval_root = output_root / "eval_timewarp"
    oss_eval_root = output_root / "eval_optimalsteps_like"

    metadata = {
        "config": config_path,
        "checkpoint": checkpoint,
        "steps": steps,
        "fid_samples": int(args.fid_samples),
        "fid_batch_size": int(args.fid_batch_size),
        "sample_batch_size": int(args.sample_batch_size),
        "reference_steps": int(args.reference_steps),
        "search_batch_size": int(args.search_batch_size),
        "search_cost_batch_size": int(args.search_cost_batch_size),
        "search_seed": int(args.search_seed),
        "fixed_seed": int(args.fixed_seed),
        "optimalsteps_head": _git_head(ROOT / "refs" / "optimalsteps"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    schedule_dir, _ = _search_schedule_bundle(args, output_root)

    default_eval_overrides = _eval_overrides(args)
    _run_eval(config_path, checkpoint, default_eval_root, steps, default_eval_overrides)

    oss_eval_overrides = _eval_overrides(args, schedule_dir=schedule_dir)
    _run_eval(config_path, checkpoint, oss_eval_root, steps, oss_eval_overrides)

    default_summary = _load_summary(default_eval_root / "reports" / "summary.json")
    oss_summary = _load_summary(oss_eval_root / "reports" / "summary.json")
    rows = _comparison_rows(
        steps=steps,
        default_summary=default_summary,
        oss_summary=oss_summary,
        default_eval_root=default_eval_root,
        oss_eval_root=oss_eval_root,
        schedule_dir=schedule_dir,
    )
    acceptance = _acceptance(rows)
    _write_comparison_files(output_root, rows, metadata, acceptance)

    print("time-coordinate ablation completed")
    print(f"output_root: {output_root}")
    print(f"acceptance: {acceptance['status']}")
    for row in rows:
        print(
            f"steps={row['step_count']} timewarp_fid={row['timewarp_fid']:.4f} "
            f"optimalsteps_fid={row['optimalsteps_fid']:.4f} "
            f"delta={row['delta_fid_optimalsteps_minus_timewarp']:+.4f}"
        )


if __name__ == "__main__":
    main()
