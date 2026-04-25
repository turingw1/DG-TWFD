from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "debug" / ".mplconfig"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgfm.config import load_experiment_config
from dgfm.evaluators.common import (
    device_from_config,
    load_model_from_checkpoint,
    load_timewarp_from_checkpoint,
    objective_mode,
    sample_condition_labels,
    sample_from_model_batched,
    to_unit_interval,
)
from dgfm.schedulers import summarize_time_grid
from dgtd.sample_dgtd import (
    build_mode_a_time_grid,
    export_dp_schedule_stub,
    load_time_grid_from_schedule,
    save_schedule,
    search_oss_time_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DGTD sampling")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Output sample directory")
    parser.add_argument("--steps", type=int, default=16, help="Sampling step count")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples")
    parser.add_argument("--mode", choices=["mode_a", "mode_b_oss", "mode_b_stub"], default="mode_a", help="DGTD sampling mode")
    parser.add_argument("--fixed-seed", type=int, default=42, help="Fixed seed for generation")
    parser.add_argument("--sample-batch-size", type=int, default=0, help="Maximum per-forward sampling batch size")
    parser.add_argument("--schedule-json", default=None, help="OSS schedule JSON to read or write")
    parser.add_argument("--force-search", action="store_true", help="Recompute OSS schedule even if --schedule-json exists")
    parser.add_argument("--reference-steps", type=int, default=32, help="Dense reference grid size for OSS search")
    parser.add_argument("--search-batch-size", type=int, default=256, help="Noise batch size for OSS schedule search")
    parser.add_argument("--search-cost-batch-size", type=int, default=0, help="Optional chunk size for OSS interval costs")
    parser.add_argument("--search-seed", type=int, default=123, help="Seed for OSS schedule search noise")
    parser.add_argument("--set", action="append", default=[], help="Config override in key=value form")
    return parser.parse_args()


def _save_samples(
    *,
    samples: torch.Tensor,
    labels: torch.Tensor | None,
    time_grid: torch.Tensor,
    output_dir: Path,
    nrow: int,
) -> None:
    samples = to_unit_interval(samples.detach().cpu())
    torch.save(samples, output_dir / "samples.pt")
    if labels is not None:
        torch.save(labels.detach().cpu(), output_dir / "labels.pt")
    save_image(samples, output_dir / "grid.png", nrow=nrow)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(samples.shape[0]):
        save_image(samples[idx], image_dir / f"{idx:06d}.png")
    torch.save(time_grid.detach().cpu(), output_dir / "time_grid.pt")


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, overrides=args.set)
    device = device_from_config(config)
    model = load_model_from_checkpoint(config, args.checkpoint, device=device)
    timewarp = load_timewarp_from_checkpoint(config, args.checkpoint, device=device)
    if objective_mode(config) != "explicit_map":
        raise ValueError("DGTD sampling requires an explicit map objective checkpoint")
    if timewarp is None:
        raise ValueError("DGTD sampling requires a checkpoint with scheduler.timewarp.enabled=true")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.fixed_seed)

    noise = torch.randn(
        args.num_samples,
        int(config["dataset"]["channels"]),
        int(config["dataset"]["image_size"]),
        int(config["dataset"]["image_size"]),
        device=device,
    )
    labels = sample_condition_labels(config, args.num_samples, device=device)
    extra = {"label": labels} if labels is not None else None

    if args.mode == "mode_b_stub":
        payload = export_dp_schedule_stub(out_path=output_dir / "dp_schedule_stub.json", step_count=args.steps)
        print("dgtd sampling stub written")
        print(payload)
        return

    sample_batch_size = int(args.sample_batch_size)
    if sample_batch_size <= 0:
        sample_batch_size = int(config.get("eval", {}).get("sample_batch_size", 0) or 0)

    schedule_payload = None
    if args.mode == "mode_a":
        time_grid = build_mode_a_time_grid(warp=timewarp, step_count=args.steps, device=device, dtype=noise.dtype)
    else:
        schedule_path = Path(args.schedule_json) if args.schedule_json else output_dir / f"oss_schedule_steps{args.steps}.json"
        if schedule_path.exists() and not args.force_search:
            time_grid, schedule_payload = load_time_grid_from_schedule(
                schedule_path,
                step_count=args.steps,
                device=device,
                dtype=noise.dtype,
            )
        else:
            search_generator = torch.Generator(device=device).manual_seed(args.search_seed)
            search_noise = torch.randn(
                args.search_batch_size,
                int(config["dataset"]["channels"]),
                int(config["dataset"]["image_size"]),
                int(config["dataset"]["image_size"]),
                generator=search_generator,
                device=device,
            )
            search_labels = sample_condition_labels(config, args.search_batch_size, device=device, generator=search_generator)
            search_extra = {"label": search_labels} if search_labels is not None else None
            time_grid, schedule_payload = search_oss_time_grid(
                model=model,
                x_search=search_noise,
                step_count=args.steps,
                reference_steps=args.reference_steps,
                warp=timewarp,
                extra=search_extra,
                cost_batch_size=args.search_cost_batch_size,
            )
            save_schedule(schedule_payload, schedule_path)
        torch.save(time_grid.detach().cpu(), output_dir / f"oss_time_grid_steps{args.steps}.pt")

    samples = sample_from_model_batched(
        config=config,
        model=model,
        x_init=noise,
        step_count=args.steps,
        method="map_rollout",
        time_grid=time_grid,
        max_batch_size=sample_batch_size,
        move_to_cpu=True,
        extra=extra,
    )
    _save_samples(
        samples=samples,
        labels=labels,
        time_grid=time_grid,
        output_dir=output_dir,
        nrow=max(1, int(args.num_samples**0.5)),
    )

    print("dgtd sampling completed")
    print(f"checkpoint: {args.checkpoint}")
    print(f"output_dir: {output_dir}")
    print(f"mode: {args.mode}")
    print(f"steps: {args.steps}")
    print(f"num_samples: {args.num_samples}")
    print(f"fixed_seed: {args.fixed_seed}")
    if schedule_payload is not None:
        print(f"schedule_status: {schedule_payload.get('status')}")
        print(f"schedule_cost: {schedule_payload.get('total_cost')}")
    print(f"time_grid: {summarize_time_grid(time_grid.detach().to(device=device, dtype=torch.float32))['time_grid']}")


if __name__ == "__main__":
    main()
