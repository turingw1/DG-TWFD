from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dg_twfd.config import load_config
from dg_twfd.engine.checkpoint import load_checkpoint, load_model_state_dict
from dg_twfd.infer import profile_sampling, sample_dg_twfd
from dg_twfd.models import BoundaryCorrector, FlowStudent, TimeWarpMonotone
from dg_twfd.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from DG-TWFD checkpoints")
    parser.add_argument("--mode", default="debug_4060", help="Config profile name")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Checkpoint path")
    parser.add_argument("--output-dir", default=None, help="Directory for sample artifacts")
    parser.add_argument("--steps", type=int, default=16, choices=[1, 2, 4, 8, 16], help="Sampling steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of samples")
    parser.add_argument("--disable-boundary", action="store_true", help="Skip boundary corrector")
    parser.add_argument("--force-boundary", action="store_true", help="Force enable boundary corrector")
    parser.add_argument("--override", action="append", default=[], help="Config overrides in key=value form")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_models(cfg, device: torch.device) -> dict[str, torch.nn.Module]:
    return {
        "student": FlowStudent(
            channels=cfg.data.channels,
            hidden_channels=cfg.model.hidden_channels,
            time_embed_dim=cfg.model.time_embed_dim,
            cond_dim=cfg.model.cond_dim,
            num_blocks=cfg.model.student_num_blocks,
            predict_residual=cfg.model.predict_residual,
        ).to(device),
        "timewarp": TimeWarpMonotone(
            num_bins=cfg.model.timewarp_num_bins,
            init_bias=cfg.model.timewarp_init_bias,
        ).to(device),
        "boundary": BoundaryCorrector(
            channels=cfg.data.channels,
            hidden_channels=cfg.model.boundary_hidden_channels,
            num_blocks=cfg.model.boundary_num_blocks,
        ).to(device),
    }


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent / "samples"
    return ckpt_path.parent / "samples"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.mode, overrides=args.override)
    seed_everything(cfg.experiment.seed)
    device = resolve_device(cfg.runtime.device)

    models = build_models(cfg, device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    for name, model in models.items():
        load_model_state_dict(model, ckpt["models"][name])
        model.eval()

    noise = torch.randn(
        args.batch_size,
        cfg.data.channels,
        cfg.data.image_size,
        cfg.data.image_size,
        device=device,
    )
    enable_boundary = (not args.disable_boundary) and (
        args.force_boundary or cfg.boundary.enable_until_step > 0
    )
    if enable_boundary:
        print("sampling_boundary: enabled")
    else:
        print(
            "sampling_boundary: disabled "
            f"(disable_flag={args.disable_boundary}, force_flag={args.force_boundary}, "
            f"cfg_enable_until_step={cfg.boundary.enable_until_step})"
        )

    samples, diagnostics = sample_dg_twfd(
        models=models,
        timewarp=models["timewarp"],
        boundary=models["boundary"],
        noise=noise,
        steps=args.steps,
        device=device,
        enable_boundary=enable_boundary,
        amp=cfg.runtime.amp,
        gate_weight=cfg.boundary.gate_weight,
    )
    profiles = profile_sampling(
        models=models,
        timewarp=models["timewarp"],
        boundary=models["boundary"],
        noise=noise[:1],
        steps_list=[1, 2, 4, 8, 16],
        device=device,
        amp=cfg.runtime.amp,
        enable_boundary=enable_boundary,
        gate_weight=cfg.boundary.gate_weight,
    )

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"samples_steps{args.steps}.pt"
    diag_path = output_dir / f"sample_diag_steps{args.steps}.pt"
    torch.save(samples.detach().cpu(), sample_path)
    torch.save(
        {
            "diagnostics": diagnostics,
            "profiles": profiles,
            "defect_curve_placeholder": ckpt["scheduler"]["defect_ema"],
        },
        diag_path,
    )

    print(f"saved_samples: {sample_path}")
    print(f"saved_diagnostics: {diag_path}")
    print(
        "sample_stats: min=%.6f max=%.6f mean=%.6f std=%.6f"
        % (
            float(samples.min().item()),
            float(samples.max().item()),
            float(samples.mean().item()),
            float(samples.std().item()),
        )
    )
    print("sampling profile:")
    for row in profiles:
        print(
            f"steps={int(row['steps'])} nfe={int(row['nfe'])} "
            f"latency_ms={row['latency_ms']:.3f} peak_mem_mib={row['peak_mem_mib']:.2f}"
        )
    print(f"sample_shape: {tuple(samples.shape)}")
    print(f"t_schedule: {diagnostics['t_schedule'].tolist()}")


if __name__ == "__main__":
    main()
