from __future__ import annotations

from dataclasses import dataclass
import json
import time

import torch
from torch import nn
import torch.nn.functional as F
import yaml

from dgfm.config import RunRoots
from dgfm.datasets import build_image_dataloaders
from dgfm.models import ModelEMA, build_map_model
from dgfm.paths import build_path, ensure_flow_matching_on_path
from dgfm.targets import build_target_builder


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


@dataclass(slots=True)
class MapTrainer:
    config: dict
    roots: RunRoots

    def prepare(self) -> None:
        self.roots.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.roots.sample_dir.mkdir(parents=True, exist_ok=True)
        self.roots.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.roots.log_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=False)

    def _run_epoch(
        self,
        model: nn.Module,
        ema: ModelEMA | None,
        loader,
        optimizer,
        scaler: torch.amp.GradScaler,
        path,
        target_builder,
        device: torch.device,
        train: bool,
    ) -> dict[str, float]:
        model.train(train)
        total_loss = 0.0
        total_t = 0.0
        total_s = 0.0
        total_delta = 0.0
        count = 0
        train_cfg = self.config.get("train", {})
        target_cfg = self.config.get("target", {})
        batch_limit_key = "max_train_batches" if train else "max_val_batches"
        batch_limit = int(train_cfg.get(batch_limit_key, 0) or 0)
        use_amp = bool(self.config.get("runtime", {}).get("amp", True))
        ctx = torch.enable_grad if train else torch.no_grad
        with ctx():
            for batch_idx, (images, _labels) in enumerate(loader):
                if batch_limit > 0 and batch_idx >= batch_limit:
                    break
                images = images.to(device) * 2.0 - 1.0
                noise = torch.randn_like(images)
                target_batch = target_builder.build(x_0=noise, x_1=images, path=path)
                with _autocast_context(device, use_amp):
                    pred = model(target_batch.x_t, target_batch.t, target_batch.s, extra={})
                    loss = _compute_map_loss(pred, target_batch.x_s_target, target_cfg)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if ema is not None:
                        ema.update(model)
                total_loss += float(loss.detach().item())
                total_t += float(target_batch.t.mean().item())
                total_s += float(target_batch.s.mean().item())
                total_delta += float((target_batch.s - target_batch.t).mean().item())
                count += 1
        denom = max(1, count)
        return {
            "loss": total_loss / denom,
            "t_mean": total_t / denom,
            "s_mean": total_s / denom,
            "delta_mean": total_delta / denom,
        }

    def run(self, resume: str | None = None) -> None:
        self.prepare()
        ensure_flow_matching_on_path()
        device = _device_from_config(self.config)
        dataloaders = build_image_dataloaders(self.config)
        path = build_path(self.config)
        target_builder = build_target_builder(self.config)
        model = build_map_model(self.config).to(device)
        ema = ModelEMA(model, decay=float(self.config["train"].get("ema_decay", 0.9999)))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["train"].get("lr", 1.0e-4)),
            weight_decay=float(self.config["train"].get("weight_decay", 0.0)),
            betas=tuple(self.config["train"].get("optimizer_betas", [0.9, 0.95])),
        )
        scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda" and bool(self.config.get("runtime", {}).get("amp", True)))
        start_epoch = 0
        best_val = float("inf")
        if resume:
            ckpt = torch.load(resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "ema_model" in ckpt:
                ema.load_state_dict(ckpt["ema_model"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))
            if "scaler" in ckpt and ckpt["scaler"] is not None and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler"])

        epochs = int(self.config["train"].get("epochs", 1))
        history_path = self.roots.log_dir / "train.jsonl"
        target_mode = str(self.config.get("target", {}).get("builder", "analytic_path"))
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            train_stats = self._run_epoch(model, ema, dataloaders["train"], optimizer, scaler, path, target_builder, device, train=True)
            eval_model = ema.shadow if ema is not None else model
            val_stats = self._run_epoch(eval_model, None, dataloaders["val"], optimizer, scaler, path, target_builder, device, train=False)
            elapsed = time.time() - t0
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "train_t_mean": train_stats["t_mean"],
                "train_s_mean": train_stats["s_mean"],
                "train_delta_mean": train_stats["delta_mean"],
                "val_t_mean": val_stats["t_mean"],
                "val_s_mean": val_stats["s_mean"],
                "val_delta_mean": val_stats["delta_mean"],
                "target_builder": target_mode,
                "elapsed_sec": elapsed,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "best_val": best_val,
                "config": self.config,
            }
            torch.save(ckpt, self.roots.checkpoint_dir / "last.pt")
            if val_stats["loss"] <= best_val:
                best_val = val_stats["loss"]
                ckpt["best_val"] = best_val
                torch.save(ckpt, self.roots.checkpoint_dir / "best.pt")
            print(
                f"epoch={epoch + 1}/{epochs} train_loss={train_stats['loss']:.6f} "
                f"val_loss={val_stats['loss']:.6f} target={target_mode} "
                f"t_mean={train_stats['t_mean']:.4f} s_mean={train_stats['s_mean']:.4f} "
                f"delta_mean={train_stats['delta_mean']:.4f} elapsed_sec={elapsed:.2f}",
                flush=True,
            )
