from __future__ import annotations

from dataclasses import dataclass
import json
import time

import torch
from torch import nn
import yaml

from dgfm.config import RunRoots
from dgfm.datasets import build_image_dataloaders
from dgfm.models import build_velocity_model
from dgfm.paths import build_path, ensure_flow_matching_on_path


def _device_from_config(config: dict) -> torch.device:
    requested = config.get("runtime", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@dataclass(slots=True)
class BaselineTrainer:
    config: dict
    roots: RunRoots

    def prepare(self) -> None:
        self.roots.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.roots.sample_dir.mkdir(parents=True, exist_ok=True)
        self.roots.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.roots.log_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=False)

    def _run_epoch(self, model: nn.Module, loader, optimizer, path, device: torch.device, train: bool) -> float:
        model.train(train)
        total = 0.0
        count = 0
        train_cfg = self.config.get("train", {})
        batch_limit_key = "max_train_batches" if train else "max_val_batches"
        batch_limit = int(train_cfg.get(batch_limit_key, 0) or 0)
        ctx = torch.enable_grad if train else torch.no_grad
        with ctx():
            for batch_idx, (images, _labels) in enumerate(loader):
                if batch_limit > 0 and batch_idx >= batch_limit:
                    break
                images = images.to(device)
                images = images * 2.0 - 1.0
                noise = torch.randn_like(images)
                t = torch.rand(images.shape[0], device=device)
                path_sample = path.sample(x_0=noise, x_1=images, t=t)
                pred = model(path_sample.x_t, path_sample.t)
                loss = torch.mean((pred - path_sample.dx_t) ** 2)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                total += float(loss.detach().item())
                count += 1
        return total / max(1, count)

    def run(self, resume: str | None = None) -> None:
        self.prepare()
        ensure_flow_matching_on_path()
        device = _device_from_config(self.config)
        dataloaders = build_image_dataloaders(self.config)
        path = build_path(self.config)
        model = build_velocity_model(self.config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["train"].get("lr", 1.0e-4)),
            weight_decay=float(self.config["train"].get("weight_decay", 0.0)),
        )
        start_epoch = 0
        best_val = float("inf")
        if resume:
            ckpt = torch.load(resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))

        epochs = int(self.config["train"].get("epochs", 1))
        history_path = self.roots.log_dir / "train.jsonl"
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            train_loss = self._run_epoch(model, dataloaders["train"], optimizer, path, device, train=True)
            val_loss = self._run_epoch(model, dataloaders["val"], optimizer, path, device, train=False)
            elapsed = time.time() - t0
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_sec": elapsed,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
            last_ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "config": self.config,
            }
            torch.save(last_ckpt, self.roots.checkpoint_dir / "last.pt")
            if val_loss <= best_val:
                best_val = val_loss
                last_ckpt["best_val"] = best_val
                torch.save(last_ckpt, self.roots.checkpoint_dir / "best.pt")
            print(
                f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} elapsed_sec={elapsed:.2f}",
                flush=True,
            )
