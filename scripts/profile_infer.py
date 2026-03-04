from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sample import build_models, resolve_device
from dg_twfd.config import load_config
from dg_twfd.engine.checkpoint import load_checkpoint
from dg_twfd.infer import profile_sampling


def main() -> None:
    cfg = load_config("debug_4060")
    device = resolve_device(cfg.runtime.device)
    models = build_models(cfg, device)
    checkpoint = load_checkpoint(ROOT / "checkpoints" / "best.pt", map_location=device)
    for name, model in models.items():
        model.load_state_dict(checkpoint["models"][name])
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
