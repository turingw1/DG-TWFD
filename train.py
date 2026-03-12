from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dg_twfd.config import load_config
from dg_twfd.data import build_dataloader, build_teacher
from dg_twfd.engine.trainer import Trainer
from dg_twfd.losses import BoundaryLoss, MatchLoss, SemigroupDefectLoss, WarpLoss
from dg_twfd.models import BoundaryCorrector, FlowStudent, TimeWarpMonotone
from dg_twfd.schedule import DefectAdaptiveScheduler
from dg_twfd.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DG-TWFD debug pipeline")
    parser.add_argument("--mode", default="debug_4060", help="Config profile name")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional config overrides in key=value form",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def stage_log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | train.py | {message}", flush=True)


def main() -> None:
    args = parse_args()
    overrides = list(args.override)
    if args.epochs is not None:
        overrides.append(f"train.epochs={args.epochs}")

    stage_log(f"loading config profile={args.mode}")
    cfg = load_config(args.mode, overrides=overrides)
    if args.mode == "debug_4060":
        cfg.experiment.name = "dg_twfd_phase4_debug"
    stage_log(f"seed_everything(seed={cfg.experiment.seed})")
    seed_everything(cfg.experiment.seed)
    device = resolve_device(cfg.runtime.device)
    stage_log(
        "resolved device=%s dataset_type=%s batch_size=%d num_workers=%d prefetch=%s"
        % (
            device,
            cfg.data.dataset_type,
            cfg.data.batch_size,
            cfg.data.num_workers,
            str(cfg.data.prefetch_factor),
        )
    )

    stage_log("building teacher")
    teacher = build_teacher(cfg)
    stage_log("building dataloaders (train/val)")
    dataloaders = {
        "train": build_dataloader(cfg, teacher, split="train"),
        "val": build_dataloader(cfg, teacher, split="val"),
    }
    stage_log(
        "dataloaders ready train_len=%d val_len=%d"
        % (len(dataloaders["train"].dataset), len(dataloaders["val"].dataset))
    )
    stage_log("building models/losses/scheduler")
    models = {
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
    losses = {
        "match": MatchLoss(cfg.loss.match_loss_type, cfg.loss.huber_delta),
        "defect": SemigroupDefectLoss(
            per_pixel_mean=cfg.loss.per_pixel_mean,
            short_weight=cfg.loss.semigroup_short_weight,
            mid_weight=cfg.loss.semigroup_mid_weight,
            long_weight=cfg.loss.semigroup_long_weight,
        ),
        "warp": WarpLoss(cfg.loss.per_pixel_mean),
        "boundary": BoundaryLoss(cfg.loss.match_loss_type, cfg.loss.huber_delta),
    }
    defect_scheduler = DefectAdaptiveScheduler(
        num_bins=cfg.schedule.num_bins,
        ema_decay=cfg.schedule.ema_decay,
        eta=cfg.schedule.eta,
        eps=cfg.schedule.eps,
        seed=cfg.schedule.seed,
    )

    trainer = Trainer(
        cfg=cfg,
        teacher=teacher,
        models=models,
        losses=losses,
        scheduler=defect_scheduler,
        dataloaders=dataloaders,
        device=device,
    )
    stage_log("starting trainer.fit()")
    epoch_losses = trainer.fit()
    print("Loss trend by epoch:")
    for idx, value in enumerate(epoch_losses, start=1):
        print(f"epoch {idx}: {value:.6f}")


if __name__ == "__main__":
    main()
