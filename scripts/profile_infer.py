from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sample import build_models, resolve_device
from dg_twfd.config import load_config
from dg_twfd.engine.checkpoint import load_checkpoint, load_model_state_dict
from dg_twfd.infer import profile_sampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile DG-TWFD sampling")
    parser.add_argument("--mode", default="train_a100", help="Config profile name")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default from config)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.mode)
    device = resolve_device(cfg.runtime.device)
    models = build_models(cfg, device)
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = Path(cfg.train.checkpoint_dir) / "best.pt"
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    print(f"profile_checkpoint: {checkpoint_path}")
    for name, model in models.items():
        load_model_state_dict(model, checkpoint["models"][name])
        model.eval()

    noise = torch.randn(1, cfg.data.channels, cfg.data.image_size, cfg.data.image_size, device=device)
    rows = profile_sampling(
        models=models,
        timewarp=models["timewarp"],
        boundary=models["boundary"],
        noise=noise,
        steps_list=[1, 2, 4, 8, 16],
        device=device,
        amp=cfg.runtime.amp,
        enable_boundary=True,
        gate_weight=cfg.boundary.gate_weight,
    )
    print("steps | nfe | latency_ms | peak_mem_mib")
    for row in rows:
        print(
            f"{int(row['steps']):>5} | {int(row['nfe']):>3} | "
            f"{row['latency_ms']:>10.3f} | {row['peak_mem_mib']:>12.2f}"
        )


if __name__ == "__main__":
    main()
