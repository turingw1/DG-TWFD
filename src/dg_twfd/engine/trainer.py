"""Training loop for DG-TWFD."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from dg_twfd.config import DGConfig
from dg_twfd.engine.amp import autocast_context, build_grad_scaler
from dg_twfd.engine.checkpoint import load_checkpoint, save_checkpoint
from dg_twfd.engine.metrics import MetricTracker
from dg_twfd.schedule import DefectAdaptiveScheduler
from dg_twfd.utils.logging import setup_logger


class _NoOpWriter:
    def log(self, metrics: dict[str, float], step: int) -> None:
        del metrics, step

    def close(self) -> None:
        return None


class _WandbWriter:
    def __init__(self, run: Any) -> None:
        self.run = run

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.run.log(metrics, step=step)

    def close(self) -> None:
        self.run.finish()


class _TensorBoardWriter:
    def __init__(self, writer: Any) -> None:
        self.writer = writer

    def log(self, metrics: dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()


def build_writer(cfg: DGConfig, log_dir: Path):
    try:
        import wandb  # type: ignore

        mode = "offline" if "debug" in cfg.experiment.name else "online"
        run = wandb.init(
            project="dg-twfd",
            name=cfg.experiment.name,
            dir=str(log_dir),
            mode=mode,
            reinit=True,
        )
        return _WandbWriter(run), "wandb"
    except Exception:
        try:
            from torch.utils.tensorboard import SummaryWriter

            return _TensorBoardWriter(SummaryWriter(log_dir=str(log_dir))), "tensorboard"
        except Exception:
            return _NoOpWriter(), "noop"


@dataclass(slots=True)
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")


class Trainer:
    """Injectable trainer for DG-TWFD Phase 4.

    Student and boundary parameters are updated every step. The time-warp uses
    an alternating optimizer schedule so `g_phi` is nudged by warp/defect terms
    only every `warp_update_every` steps, which reduces early training
    instability compared to updating every parameter with every objective.
    """

    def __init__(
        self,
        cfg: DGConfig,
        teacher: Any,
        models: dict[str, torch.nn.Module],
        losses: dict[str, torch.nn.Module],
        scheduler: DefectAdaptiveScheduler,
        dataloaders: dict[str, DataLoader],
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.teacher = teacher
        self.models = models
        self.losses = losses
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.device = device
        self.device_type = device.type
        self.logger = setup_logger(cfg.logging.level, "dg_twfd.trainer")
        self.metrics = MetricTracker()
        self.state = TrainerState()
        self.scaler = build_grad_scaler(self.device_type, cfg.runtime.amp)
        self.grad_accum = max(1, cfg.runtime.gradient_accumulation)
        self.use_channels_last = self.device_type == "cuda"

        if self.device_type == "cuda" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            self.models["student"] = self.models["student"].to(memory_format=torch.channels_last)
            self.models["boundary"] = self.models["boundary"].to(memory_format=torch.channels_last)
            if os.environ.get("DG_TWFD_COMPILE", "0") == "1" and hasattr(torch, "compile"):
                self.models["student"] = torch.compile(self.models["student"])
                self.models["boundary"] = torch.compile(self.models["boundary"])

        student_params = list(models["student"].parameters()) + list(models["boundary"].parameters())
        warp_params = list(models["timewarp"].parameters())
        self.student_optimizer = torch.optim.AdamW(
            student_params,
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.warp_optimizer = torch.optim.AdamW(
            warp_params,
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )

        ckpt_dir = Path(cfg.train.checkpoint_dir)
        self.writer, self.writer_backend = build_writer(cfg, ckpt_dir / "logs")
        self.logger.info("Logging backend: %s", self.writer_backend)
        self.logger.info("Training on device=%s with AMP=%s", device, cfg.runtime.amp)

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        moved: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            tensor = value.to(self.device, non_blocking=True)
            if self.use_channels_last and tensor.ndim == 4:
                tensor = tensor.to(memory_format=torch.channels_last)
            moved[key] = tensor
        return moved

    def _boundary_enabled(self) -> bool:
        return self.state.global_step < self.cfg.boundary.enable_until_step

    def _grad_norm_step(self, optimizer: torch.optim.Optimizer, params: list[torch.nn.Parameter]) -> None:
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, self.cfg.train.grad_clip_norm)
        self.scaler.step(optimizer)

    def _compute_losses(
        self,
        batch: dict[str, torch.Tensor],
        dataset: Any,
        update_scheduler: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        student = self.models["student"]
        timewarp = self.models["timewarp"]
        boundary = self.models["boundary"]

        x_t = batch["x_t"]
        x_s = batch["x_s"]
        t = batch["t"]
        s = batch["s"]

        with autocast_context(self.device_type, self.cfg.runtime.amp):
            x_s_pred = student(x_t, t, s)
            match_loss = self.losses["match"](x_s_pred, x_s)

            defect_loss, _, defect_per_sample = self.losses["defect"](
                student=student,
                x_t=x_t,
                t=t,
                s=s,
                scheduler=self.scheduler,
            )
            if update_scheduler:
                self.scheduler.update(t.detach(), defect_per_sample.detach())

            triplet = self.losses["warp"].sample_triplet_batch(
                dataset=dataset,
                batch_size=x_t.shape[0],
                device=self.device,
            )
            warp_loss, warp_stats = self.losses["warp"](timewarp, triplet)

            x_boundary = self.teacher.sample_x0(x_t.shape[0], self.device)
            t_max = torch.ones(x_t.shape[0], device=self.device)
            t_prev = torch.full((x_t.shape[0],), 0.9, device=self.device)
            x_boundary_target = self.teacher.forward_map(x_boundary, t_max, t_prev)
            boundary_loss, _ = self.losses["boundary"](
                boundary_model=boundary,
                x_boundary=x_boundary,
                target=x_boundary_target,
                gate_weight=self.cfg.boundary.gate_weight,
                enabled=self._boundary_enabled(),
            )

        scalars = {
            "match": float(match_loss.detach().item()),
            "defect": float(defect_loss.detach().item()),
            "warp": float(warp_loss.detach().item()),
            "boundary": float(boundary_loss.detach().item()),
            "warp_balance": float(warp_stats["balance"].item()),
        }
        tensors = {
            "match": match_loss,
            "defect": defect_loss,
            "warp": warp_loss,
            "boundary": boundary_loss,
        }
        return tensors, scalars

    def _student_total(self, loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            loss_dict["match"]
            + self.cfg.loss.defect_weight * loss_dict["defect"]
            + self.cfg.loss.boundary_weight * loss_dict["boundary"]
        )

    def _warp_total(self, loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.cfg.loss.warp_weight * loss_dict["warp"]

    def train_epoch(self) -> float:
        self.models["student"].train()
        self.models["timewarp"].train()
        self.models["boundary"].train()
        running_total = 0.0
        accumulation_count = 0

        self.student_optimizer.zero_grad(set_to_none=True)
        self.warp_optimizer.zero_grad(set_to_none=True)
        train_dataset = self.dataloaders["train"].dataset
        if not hasattr(train_dataset, "sample_triplet_batch"):
            raise TypeError("Training dataset must provide sample_triplet_batch()")
        epoch_start_time = time.perf_counter()
        self.logger.info(
            "Epoch %d start | train_batches=%d | grad_accum=%d | workers=%d",
            self.state.epoch,
            len(self.dataloaders["train"]),
            self.grad_accum,
            self.cfg.data.num_workers,
        )
        for batch_idx, batch in enumerate(self.dataloaders["train"], start=1):
            if batch_idx == 1:
                self.logger.info(
                    "Epoch %d first batch ready after %.2fs",
                    self.state.epoch,
                    time.perf_counter() - epoch_start_time,
                )
            batch = self._move_batch(batch)
            loss_dict, scalar_dict = self._compute_losses(
                batch=batch,
                dataset=train_dataset,
                update_scheduler=True,
            )
            student_loss = self._student_total(loss_dict) / self.grad_accum
            warp_should_step = self.state.global_step % self.cfg.train.warp_update_every == 0
            warp_loss = self._warp_total(loss_dict) / self.grad_accum if warp_should_step else None

            self.scaler.scale(student_loss).backward(retain_graph=warp_should_step)
            if warp_should_step and warp_loss is not None:
                self.scaler.scale(warp_loss).backward()

            accumulation_count += 1
            total_for_log = self._student_total(loss_dict)
            if warp_should_step and warp_loss is not None:
                total_for_log = total_for_log + self.cfg.loss.warp_weight * loss_dict["warp"]

            if accumulation_count == self.grad_accum:
                student_params = [p for p in self.models["student"].parameters()] + [
                    p for p in self.models["boundary"].parameters()
                ]
                self._grad_norm_step(self.student_optimizer, student_params)
                if warp_should_step:
                    warp_params = [p for p in self.models["timewarp"].parameters()]
                    self._grad_norm_step(self.warp_optimizer, warp_params)
                self.scaler.update()
                self.student_optimizer.zero_grad(set_to_none=True)
                self.warp_optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0

            self.state.global_step += 1
            running_total += float(total_for_log.detach().item())
            metrics = {
                "train/l_match": scalar_dict["match"],
                "train/l_def": scalar_dict["defect"],
                "train/l_warp": scalar_dict["warp"],
                "train/l_boundary": scalar_dict["boundary"],
                "train/warp_balance": scalar_dict["warp_balance"],
                "train/total_loss": float(total_for_log.detach().item()),
            }
            elapsed = max(time.perf_counter() - epoch_start_time, 1e-6)
            metrics["train/steps_per_sec"] = batch_idx / elapsed
            self.metrics.update(**metrics, scheduler_eta=self.scheduler.eta)
            self.writer.log(metrics, self.state.global_step)

            if self.state.global_step % self.cfg.train.log_every == 0:
                peak_mem = (
                    torch.cuda.max_memory_allocated() / (1024**2)
                    if self.device_type == "cuda" and torch.cuda.is_available()
                    else 0.0
                )
                self.logger.info(
                    "epoch=%d step=%d total=%.6f match=%.6f defect=%.6f warp=%.6f boundary=%.6f sps=%.2f peak_mem=%.2fMiB",
                    self.state.epoch,
                    self.state.global_step,
                    metrics["train/total_loss"],
                    metrics["train/l_match"],
                    metrics["train/l_def"],
                    metrics["train/l_warp"],
                    metrics["train/l_boundary"],
                    metrics["train/steps_per_sec"],
                    peak_mem,
                )
            if self.cfg.train.max_train_steps is not None and self.state.global_step >= self.cfg.train.max_train_steps:
                break

        if accumulation_count > 0:
            student_params = [p for p in self.models["student"].parameters()] + [
                p for p in self.models["boundary"].parameters()
            ]
            self._grad_norm_step(self.student_optimizer, student_params)
            warp_params = [p for p in self.models["timewarp"].parameters()]
            if any(param.grad is not None for param in warp_params):
                self._grad_norm_step(self.warp_optimizer, warp_params)
            self.scaler.update()
            self.student_optimizer.zero_grad(set_to_none=True)
            self.warp_optimizer.zero_grad(set_to_none=True)

        num_batches = min(batch_idx, len(self.dataloaders["train"]))
        return running_total / max(1, num_batches)

    @torch.no_grad()
    def validate(self) -> float:
        self.models["student"].eval()
        self.models["boundary"].eval()
        self.models["timewarp"].eval()
        total = 0.0
        num_batches = 0
        val_dataset = self.dataloaders["val"].dataset
        if not hasattr(val_dataset, "sample_triplet_batch"):
            raise TypeError("Validation dataset must provide sample_triplet_batch()")
        val_start_time = time.perf_counter()
        for batch in self.dataloaders["val"]:
            if num_batches == 0:
                self.logger.info("Validation first batch ready after %.2fs", time.perf_counter() - val_start_time)
            batch = self._move_batch(batch)
            loss_dict, _ = self._compute_losses(
                batch=batch,
                dataset=val_dataset,
                update_scheduler=False,
            )
            total_loss = self._student_total(loss_dict) + self.cfg.loss.warp_weight * loss_dict["warp"]
            total += float(total_loss.detach().item())
            num_batches += 1
        value = total / max(1, num_batches)
        self.metrics.update(**{"val/total_loss": value})
        self.writer.log({"val/total_loss": value}, self.state.global_step)
        return value

    def checkpoint_state(self) -> dict[str, Any]:
        t_grid, u_grid = self.models["timewarp"].grid_cache()
        return {
            "cfg_name": self.cfg.experiment.name,
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_val_loss": self.state.best_val_loss,
            },
            "models": {name: model.state_dict() for name, model in self.models.items()},
            "optimizers": {
                "student": self.student_optimizer.state_dict(),
                "warp": self.warp_optimizer.state_dict(),
            },
            "scaler": self.scaler.state_dict(),
            "scheduler": {
                "defect_ema": self.scheduler.defect_ema,
                "eta": self.scheduler.eta,
            },
            "artifacts": {
                "t_grid": t_grid.detach().cpu(),
                "u_grid": u_grid.detach().cpu(),
            },
        }

    def maybe_save_checkpoint(self, is_best: bool = False) -> None:
        ckpt_dir = Path(self.cfg.train.checkpoint_dir)
        epoch_path = ckpt_dir / f"epoch_{self.state.epoch:03d}.pt"
        save_checkpoint(epoch_path, self.checkpoint_state())
        if is_best:
            save_checkpoint(ckpt_dir / "best.pt", self.checkpoint_state())

    def resume(self, path: str | Path) -> None:
        payload = load_checkpoint(path, map_location=self.device)
        for name, model in self.models.items():
            model.load_state_dict(payload["models"][name])
        self.student_optimizer.load_state_dict(payload["optimizers"]["student"])
        self.warp_optimizer.load_state_dict(payload["optimizers"]["warp"])
        self.scaler.load_state_dict(payload["scaler"])
        self.scheduler.defect_ema = payload["scheduler"]["defect_ema"]
        self.scheduler.eta = payload["scheduler"]["eta"]
        self.state.epoch = payload["state"]["epoch"]
        self.state.global_step = payload["state"]["global_step"]
        self.state.best_val_loss = payload["state"]["best_val_loss"]
        self.logger.info("Resumed from %s", path)

    def fit(self) -> list[float]:
        if self.cfg.train.resume_path:
            self.resume(self.cfg.train.resume_path)
        epoch_losses: list[float] = []
        for epoch in range(self.state.epoch + 1, self.cfg.train.epochs + 1):
            self.state.epoch = epoch
            if self.device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            train_loss = self.train_epoch()
            val_loss = self.validate()
            epoch_losses.append(train_loss)
            improved = val_loss < self.state.best_val_loss
            if improved:
                self.state.best_val_loss = val_loss
            if epoch % self.cfg.train.save_every == 0:
                self.maybe_save_checkpoint(is_best=improved)
            peak_mem = (
                torch.cuda.max_memory_allocated() / (1024**2)
                if self.device_type == "cuda" and torch.cuda.is_available()
                else 0.0
            )
            self.logger.info(
                "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | peak_mem=%.2fMiB",
                epoch,
                self.cfg.train.epochs,
                train_loss,
                val_loss,
                peak_mem,
            )
        self.writer.close()
        return epoch_losses
