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
    objective_mode,
    sample_from_model_batched,
    to_unit_interval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run refactored DGFM qualitative sampling")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Output sample directory")
    parser.add_argument("--steps", type=int, default=16, help="Sampling step count")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples")
    parser.add_argument("--sample-batch-size", type=int, default=0, help="Maximum per-forward sampling batch size")
    parser.add_argument("--fixed-seed", type=int, default=42, help="Fixed seed for qualitative generation")
    parser.add_argument("--set", action="append", default=[], help="Config override in key=value form")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, overrides=args.set)
    device = device_from_config(config)
    model = load_model_from_checkpoint(config, args.checkpoint, device=device)

    torch.manual_seed(args.fixed_seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    noise = torch.randn(
        args.num_samples,
        int(config["dataset"]["channels"]),
        int(config["dataset"]["image_size"]),
        int(config["dataset"]["image_size"]),
        device=device,
    )
    sample_batch_size = int(args.sample_batch_size)
    if sample_batch_size <= 0:
        sample_batch_size = int(config.get("eval", {}).get("sample_batch_size", 0) or 0)
    samples = sample_from_model_batched(
        config=config,
        model=model,
        x_init=noise,
        step_count=args.steps,
        max_batch_size=sample_batch_size,
        move_to_cpu=True,
    )
    samples = to_unit_interval(samples)
    torch.save(samples, output_dir / "samples.pt")
    save_image(samples, output_dir / "grid.png", nrow=max(1, int(args.num_samples**0.5)))
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(samples.shape[0]):
        save_image(samples[idx], image_dir / f"{idx:06d}.png")
    print("dgfm sampling completed")
    print(f"checkpoint: {args.checkpoint}")
    print(f"output_dir: {output_dir}")
    print(f"objective_mode: {objective_mode(config)}")
    print(f"steps: {args.steps}")
    print(f"num_samples: {args.num_samples}")
    print(f"fixed_seed: {args.fixed_seed}")


if __name__ == "__main__":
    main()
