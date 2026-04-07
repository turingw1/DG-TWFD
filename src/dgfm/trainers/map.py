from __future__ import annotations

from dataclasses import dataclass, field
import json
import time

import torch
from torch import nn
import torch.nn.functional as F
import yaml

from dgfm.config import RunRoots
from dgfm.datasets import build_map_training_dataloaders
from dgfm.losses import build_perceptual_metric
from dgfm.models import ModelEMA, build_map_model
from dgfm.paths import build_path, ensure_flow_matching_on_path
from dgfm.samplers import rollout_with_map
from dgfm.schedulers import build_config_time_grid
from dgfm.targets import build_target_builder
from dgfm.utils import build_experiment_archive


def _device_from_config(config: dict) -> torch.device:
    requested = config.get("runtime", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _autocast_context(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def _compute_map_loss(pred: torch.Tensor, target: torch.Tensor, target_cfg: dict) -> torch.Tensor:
    loss_type = str(target_cfg.get("loss_type", "mse"))
    if loss_type == "mse":
        return torch.mean((pred - target) ** 2)
    if loss_type == "huber":
        return F.huber_loss(pred, target, delta=float(target_cfg.get("huber_delta", 0.1)))
    raise ValueError(f"Unsupported map loss_type: {loss_type}")


def _loss_weights(config: dict) -> dict[str, float]:
    loss_cfg = config.get("loss", {})
    return {
        "pixel": float(loss_cfg.get("pixel_weight", 1.0)),
        "perceptual": float(loss_cfg.get("perceptual_weight", 0.0)),
        "endpoint": float(loss_cfg.get("endpoint_weight", 0.0)),
    }


def _compute_prediction_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    target_cfg: dict,
    perceptual_metric: nn.Module | None,
    perceptual_weight: float,
    pixel_weight: float,
) -> dict[str, torch.Tensor]:
    pixel_loss = _compute_map_loss(pred, target, target_cfg)
    perceptual_loss = pred.new_tensor(0.0)
    if perceptual_metric is not None and perceptual_weight > 0.0:
        perceptual_loss = perceptual_metric(pred, target)
    total = pixel_weight * pixel_loss + perceptual_weight * perceptual_loss
    return {
        "total": total,
        "pixel": pixel_loss,
        "perceptual": perceptual_loss,
    }


def _sample_endpoint_step(loss_cfg: dict) -> int:
    step_choices = list(loss_cfg.get("endpoint_steps", [8, 16]))
    if not step_choices:
        raise ValueError("loss.endpoint_steps must be non-empty when endpoint loss is enabled")
    weights = list(loss_cfg.get("endpoint_step_weights", [1.0] * len(step_choices)))
    if len(weights) != len(step_choices):
        raise ValueError("loss.endpoint_step_weights must have the same length as loss.endpoint_steps")
    probs = torch.tensor(weights, dtype=torch.float32)
    probs = probs / probs.sum()
    choice = int(torch.multinomial(probs, 1).item())
    return int(step_choices[choice])


@dataclass(slots=True)
class MapTrainer:
    config: dict
    roots: RunRoots
    archive: object | None = field(init=False, default=None)

    def prepare(self) -> None:
        self.roots.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.roots.sample_dir.mkdir(parents=True, exist_ok=True)
        self.roots.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.roots.log_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=False)
        self.archive = build_experiment_archive(self.roots)
        self.archive.dump_yaml("config_resolved.yaml", self.config)

    def _run_epoch(
        self,
        model: nn.Module,
        ema: ModelEMA | None,
        loader,
        optimizer,
        scaler: torch.amp.GradScaler,
        path,
        target_builder,
        perceptual_metric: nn.Module | None,
        device: torch.device,
        train: bool,
        global_step_start: int,
    ) -> dict[str, float]:
        model.train(train)
        total_loss = 0.0
        total_pixel = 0.0
        total_perceptual = 0.0
        total_endpoint = 0.0
        total_endpoint_pixel = 0.0
        total_endpoint_perceptual = 0.0
        total_endpoint_step = 0.0
        total_t = 0.0
        total_s = 0.0
        total_delta = 0.0
        count = 0
        train_cfg = self.config.get("train", {})
        target_cfg = self.config.get("target", {})
        loss_cfg = self.config.get("loss", {})
        weights = _loss_weights(self.config)
        batch_limit_key = "max_train_batches" if train else "max_val_batches"
        batch_limit = int(train_cfg.get(batch_limit_key, 0) or 0)
        use_amp = bool(self.config.get("runtime", {}).get("amp", True))
        ctx = torch.enable_grad if train else torch.no_grad
        global_step = global_step_start
        with ctx():
            for batch_idx, batch in enumerate(loader):
                if batch_limit > 0 and batch_idx >= batch_limit:
                    break
                target_batch = target_builder.build_from_batch(batch, device=device, path=path)
                with _autocast_context(device, use_amp):
                    pred = model(target_batch.x_t, target_batch.t, target_batch.s, extra={})
                    pred_losses = _compute_prediction_losses(
                        pred,
                        target_batch.x_s_target,
                        target_cfg=target_cfg,
                        perceptual_metric=perceptual_metric,
                        perceptual_weight=weights["perceptual"],
                        pixel_weight=weights["pixel"],
                    )
                    loss = pred_losses["total"]
                    endpoint_total = pred.new_tensor(0.0)
                    endpoint_pixel = pred.new_tensor(0.0)
                    endpoint_perceptual = pred.new_tensor(0.0)
                    endpoint_step = 0
                    endpoint_enabled = weights["endpoint"] > 0.0 and target_batch.x_0 is not None and target_batch.x_1 is not None
                    endpoint_interval = max(1, int(loss_cfg.get("endpoint_every", 8)))
                    if endpoint_enabled and (not train or global_step % endpoint_interval == 0):
                        endpoint_step = _sample_endpoint_step(loss_cfg)
                        endpoint_batch_size = min(int(loss_cfg.get("endpoint_batch_size", 32)), target_batch.x_0.shape[0])
                        if endpoint_batch_size > 0:
                            if train:
                                subset = torch.randperm(target_batch.x_0.shape[0], device=device)[:endpoint_batch_size]
                            else:
                                subset = torch.arange(endpoint_batch_size, device=device)
                            student_endpoint = rollout_with_map(
                                model=model,
                                x_init=target_batch.x_0.index_select(0, subset),
                                step_count=endpoint_step,
                                time_grid=build_config_time_grid(
                                    config=self.config,
                                    step_count=endpoint_step,
                                    device=device,
                                    dtype=target_batch.x_0.dtype,
                                ),
                            )
                            endpoint_losses = _compute_prediction_losses(
                                student_endpoint,
                                target_batch.x_1.index_select(0, subset),
                                target_cfg=target_cfg,
                                perceptual_metric=perceptual_metric,
                                perceptual_weight=weights["perceptual"],
                                pixel_weight=weights["pixel"],
                            )
                            endpoint_total = endpoint_losses["total"]
                            endpoint_pixel = endpoint_losses["pixel"]
                            endpoint_perceptual = endpoint_losses["perceptual"]
                            loss = loss + weights["endpoint"] * endpoint_total
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if ema is not None:
                        ema.update(model)
                total_loss += float(loss.detach().item())
                total_pixel += float(pred_losses["pixel"].detach().item())
                total_perceptual += float(pred_losses["perceptual"].detach().item())
                total_endpoint += float(endpoint_total.detach().item())
                total_endpoint_pixel += float(endpoint_pixel.detach().item())
                total_endpoint_perceptual += float(endpoint_perceptual.detach().item())
                total_endpoint_step += float(endpoint_step)
                total_t += float(target_batch.t.mean().item())
                total_s += float(target_batch.s.mean().item())
                total_delta += float((target_batch.s - target_batch.t).mean().item())
                count += 1
                if train:
                    global_step += 1
        denom = max(1, count)
        return {
            "loss": total_loss / denom,
            "pixel_loss": total_pixel / denom,
            "perceptual_loss": total_perceptual / denom,
            "endpoint_loss": total_endpoint / denom,
            "endpoint_pixel_loss": total_endpoint_pixel / denom,
            "endpoint_perceptual_loss": total_endpoint_perceptual / denom,
            "endpoint_step": total_endpoint_step / denom,
            "t_mean": total_t / denom,
            "s_mean": total_s / denom,
            "delta_mean": total_delta / denom,
            "global_step_end": float(global_step),
        }

    def run(self, resume: str | None = None) -> None:
        self.prepare()
        device = _device_from_config(self.config)
        target_builder = build_target_builder(self.config)
        dataloaders = build_map_training_dataloaders(self.config)
        path = None
        if bool(getattr(target_builder, "needs_path", False)):
            ensure_flow_matching_on_path()
            path = build_path(self.config)
        model = build_map_model(self.config).to(device)
        ema = ModelEMA(model, decay=float(self.config["train"].get("ema_decay", 0.9999)))
        perceptual_metric = build_perceptual_metric(self.config)
        if perceptual_metric is not None:
            perceptual_metric = perceptual_metric.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["train"].get("lr", 1.0e-4)),
            weight_decay=float(self.config["train"].get("weight_decay", 0.0)),
            betas=tuple(self.config["train"].get("optimizer_betas", [0.9, 0.95])),
        )
        scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda" and bool(self.config.get("runtime", {}).get("amp", True)))
        start_epoch = 0
        best_val = float("inf")
        global_step = 0
        if resume:
            ckpt = torch.load(resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "ema_model" in ckpt:
                ema.load_state_dict(ckpt["ema_model"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))
            global_step = int(ckpt.get("global_step", global_step))
            if "scaler" in ckpt and ckpt["scaler"] is not None and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler"])

        epochs = int(self.config["train"].get("epochs", 1))
        history_path = self.roots.log_dir / "train.jsonl"
        target_mode = str(self.config.get("target", {}).get("builder", "analytic_path"))
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            train_stats = self._run_epoch(
                model,
                ema,
                dataloaders["train"],
                optimizer,
                scaler,
                path,
                target_builder,
                perceptual_metric,
                device,
                train=True,
                global_step_start=global_step,
            )
            global_step = int(train_stats["global_step_end"])
            eval_model = ema.shadow if ema is not None else model
            val_stats = self._run_epoch(
                eval_model,
                None,
                dataloaders["val"],
                optimizer,
                scaler,
                path,
                target_builder,
                perceptual_metric,
                device,
                train=False,
                global_step_start=global_step,
            )
            elapsed = time.time() - t0
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_pixel_loss": train_stats["pixel_loss"],
                "train_perceptual_loss": train_stats["perceptual_loss"],
                "train_endpoint_loss": train_stats["endpoint_loss"],
                "train_endpoint_pixel_loss": train_stats["endpoint_pixel_loss"],
                "train_endpoint_perceptual_loss": train_stats["endpoint_perceptual_loss"],
                "train_endpoint_step": train_stats["endpoint_step"],
                "val_loss": val_stats["loss"],
                "val_pixel_loss": val_stats["pixel_loss"],
                "val_perceptual_loss": val_stats["perceptual_loss"],
                "val_endpoint_loss": val_stats["endpoint_loss"],
                "val_endpoint_pixel_loss": val_stats["endpoint_pixel_loss"],
                "val_endpoint_perceptual_loss": val_stats["endpoint_perceptual_loss"],
                "val_endpoint_step": val_stats["endpoint_step"],
                "train_t_mean": train_stats["t_mean"],
                "train_s_mean": train_stats["s_mean"],
                "train_delta_mean": train_stats["delta_mean"],
                "val_t_mean": val_stats["t_mean"],
                "val_s_mean": val_stats["s_mean"],
                "val_delta_mean": val_stats["delta_mean"],
                "target_builder": target_mode,
                "global_step": global_step,
                "elapsed_sec": elapsed,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
            self.archive.append_jsonl("train.jsonl", payload)
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "best_val": best_val,
                "global_step": global_step,
                "config": self.config,
            }
            torch.save(ckpt, self.roots.checkpoint_dir / "last.pt")
            self.archive.save_checkpoint("last.pt", ckpt)
            if val_stats["loss"] <= best_val:
                best_val = val_stats["loss"]
                ckpt["best_val"] = best_val
                torch.save(ckpt, self.roots.checkpoint_dir / "best.pt")
                self.archive.save_checkpoint("best.pt", ckpt)
            print(
                f"epoch={epoch + 1}/{epochs} train_loss={train_stats['loss']:.6f} "
                f"val_loss={val_stats['loss']:.6f} pixel={train_stats['pixel_loss']:.6f} "
                f"perc={train_stats['perceptual_loss']:.6f} endpoint={train_stats['endpoint_loss']:.6f} "
                f"target={target_mode} "
                f"t_mean={train_stats['t_mean']:.4f} s_mean={train_stats['s_mean']:.4f} "
                f"delta_mean={train_stats['delta_mean']:.4f} endpoint_step={train_stats['endpoint_step']:.2f} "
                f"elapsed_sec={elapsed:.2f}",
                flush=True,
            )
