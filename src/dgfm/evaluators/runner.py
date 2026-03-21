from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import torch
from torchvision.utils import save_image

from dgfm.models import build_velocity_model
from dgfm.paths import ensure_flow_matching_on_path


def _device_from_config(config: dict) -> torch.device:
    requested = config.get("runtime", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@dataclass(slots=True)
class EvaluationRunner:
    config: dict
    checkpoint: Path
    eval_root: Path

    def _sample_steps(self, model, x, step_count: int):
        ensure_flow_matching_on_path()
        from flow_matching.solver import ODESolver

        solver = ODESolver(velocity_model=model)
        time_grid = torch.linspace(0.0, 1.0, steps=step_count + 1, device=x.device)
        return solver.sample(x_init=x, time_grid=time_grid, step_size=None, method="midpoint")

    def run(self, step_counts: list[int]) -> None:
        self.eval_root.mkdir(parents=True, exist_ok=True)
        device = _device_from_config(self.config)
        ckpt = torch.load(self.checkpoint, map_location=device)
        model = build_velocity_model(self.config).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        report_dir = self.eval_root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        fixed_seed = int(self.config.get("eval", {}).get("fixed_seed", 42))
        torch.manual_seed(fixed_seed)
        results = []
        for step_count in step_counts:
            step_dir = self.eval_root / f"steps{step_count}"
            step_dir.mkdir(parents=True, exist_ok=True)
            noise = torch.randn(16, int(self.config["dataset"]["channels"]), int(self.config["dataset"]["image_size"]), int(self.config["dataset"]["image_size"]), device=device)
            t0 = time.time()
            samples = self._sample_steps(model, noise, step_count=step_count)
            elapsed = time.time() - t0
            samples = torch.clamp(samples * 0.5 + 0.5, 0.0, 1.0)
            torch.save(samples.detach().cpu(), step_dir / "samples.pt")
            save_image(samples, step_dir / "grid.png", nrow=4)
            record = {"step_count": step_count, "elapsed_sec": elapsed, "checkpoint": str(self.checkpoint)}
            with (step_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            results.append(record)
        with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print("dgfm evaluation runner completed")
        print(f"eval_root: {self.eval_root}")
