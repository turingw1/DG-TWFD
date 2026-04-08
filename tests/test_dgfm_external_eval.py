from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dgfm.datasets.factory import build_image_dataloaders
from dgfm.evaluators.defect_evaluator import evaluate_held_out_defect
import dgfm.evaluators.official_metrics as official_metrics
from dgfm.evaluators.official_metrics import NPZImageDataset, evaluate_npz_metrics, save_samples_npz


def test_save_samples_npz_and_dataset_roundtrip(tmp_path: Path) -> None:
    images = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    labels = np.array([3, 1, 2, 0], dtype=np.int64)
    out = save_samples_npz(images, tmp_path / "samples.npz", labels_int64=labels, shuffle=True, seed=7)
    dataset = NPZImageDataset(out)
    image, label = dataset[0]
    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.uint8
    assert 0 <= label <= 3
    assert len(dataset) == 4


def test_evaluate_npz_metrics_uses_torch_fidelity_backend(monkeypatch, tmp_path: Path) -> None:
    samples = save_samples_npz(np.zeros((2, 8, 8, 3), dtype=np.uint8), tmp_path / "samples.npz")
    reference = save_samples_npz(np.zeros((2, 8, 8, 3), dtype=np.uint8), tmp_path / "reference.npz")

    def _fake_calculate_metrics(**kwargs):
        assert kwargs["fid"] is True
        assert kwargs["isc"] is True
        assert kwargs["prc"] is True
        return {
            "frechet_inception_distance": 12.5,
            "inception_score_mean": 3.25,
            "inception_score_std": 0.15,
            "precision": 0.7,
            "recall": 0.55,
        }

    monkeypatch.setattr(official_metrics, "calculate_metrics", _fake_calculate_metrics)
    result = evaluate_npz_metrics(
        samples_path=samples,
        reference_path=reference,
        metrics=["fid", "is", "precision", "recall"],
        cuda=False,
        batch_size=2,
    )
    assert result.fid == 12.5
    assert result.inception_score_mean == 3.25
    assert result.precision == 0.7
    assert result.recall == 0.55


def test_imagenet64_dataloader_supports_preprocessed_folder(tmp_path: Path) -> None:
    train_class = tmp_path / "train" / "n00000001"
    val_class = tmp_path / "val" / "n00000001"
    train_class.mkdir(parents=True)
    val_class.mkdir(parents=True)
    for idx in range(2):
        image = Image.fromarray(np.full((72, 72, 3), idx * 32, dtype=np.uint8))
        image.save(train_class / f"{idx}.png")
    Image.fromarray(np.full((72, 72, 3), 128, dtype=np.uint8)).save(val_class / "0.png")

    cfg = {
        "experiment": {"seed": 42},
        "dataset": {
            "name": "imagenet64",
            "image_size": 64,
            "channels": 3,
            "data_root": str(tmp_path),
            "train_split": "train",
            "val_split": "val",
        },
        "train": {"batch_size": 1, "num_workers": 0, "val_fraction": 0.2, "pin_memory": False},
    }
    loaders = build_image_dataloaders(cfg)
    images, labels = next(iter(loaders["train"]))
    assert images.shape == (1, 3, 64, 64)
    assert int(labels[0]) == 0


def test_held_out_defect_evaluator_runs_with_dummy_model(monkeypatch, tmp_path: Path) -> None:
    class _IdentityMap(torch.nn.Module):
        def forward(self, x_t, t, s, extra=None):
            del t, s, extra
            return x_t

    monkeypatch.setattr("dgfm.evaluators.defect_evaluator.device_from_config", lambda cfg: torch.device("cpu"))
    monkeypatch.setattr("dgfm.evaluators.defect_evaluator.load_model_from_checkpoint", lambda config, checkpoint, device: _IdentityMap())
    monkeypatch.setattr("dgfm.evaluators.defect_evaluator.load_timewarp_from_checkpoint", lambda config, checkpoint, device: None)

    cfg = {
        "train": {"objective": "explicit_map"},
        "dataset": {"channels": 1, "image_size": 4},
        "model": {"num_classes": None},
    }
    out_path = tmp_path / "defect.json"
    report = evaluate_held_out_defect(
        config=cfg,
        checkpoint="dummy.pt",
        out_path=out_path,
        num_samples=8,
        grid_steps=4,
        seed=7,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert report.defect_mean == 0.0
    assert payload["num_triplets"] > 0
