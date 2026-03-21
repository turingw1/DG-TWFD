from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgfm.config import load_experiment_config
from dgfm.models import build_velocity_model
from dgfm.paths import ensure_flow_matching_on_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run refactored DGFM qualitative sampling")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Output sample directory")
    parser.add_argument("--steps", type=int, default=16, help="Sampling step count")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples")
    parser.add_argument("--fixed-seed", type=int, default=42, help="Fixed seed for qualitative generation")
    return parser.parse_args()


def _device_from_config(config: dict) -> torch.device:
    requested = config.get("runtime", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = _device_from_config(config)
    ensure_flow_matching_on_path()
    from flow_matching.solver import ODESolver

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_velocity_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    solver = ODESolver(velocity_model=model)

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
    time_grid = torch.linspace(0.0, 1.0, steps=args.steps + 1, device=device)
    samples = solver.sample(x_init=noise, time_grid=time_grid, step_size=None, method="midpoint")
    samples = torch.clamp(samples * 0.5 + 0.5, 0.0, 1.0)
    torch.save(samples.detach().cpu(), output_dir / "samples.pt")
    save_image(samples, output_dir / "grid.png", nrow=max(1, int(args.num_samples**0.5)))
    print("dgfm sampling completed")
    print(f"checkpoint: {args.checkpoint}")
    print(f"output_dir: {output_dir}")
    print(f"steps: {args.steps}")
    print(f"num_samples: {args.num_samples}")
    print(f"fixed_seed: {args.fixed_seed}")


if __name__ == "__main__":
    main()
