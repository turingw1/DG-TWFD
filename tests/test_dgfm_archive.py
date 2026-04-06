from pathlib import Path

import torch

from dgfm.config import resolve_run_roots
from dgfm.utils import build_experiment_archive


def test_resolve_run_roots_uses_archive_env(monkeypatch, tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("DGFM_ARCHIVE_ROOT", str(archive_root))
    roots = resolve_run_roots(tmp_path / "run")
    assert roots.archive_root == archive_root
    assert roots.archive_checkpoint_dir == archive_root / "checkpoints"
    assert roots.archive_log_dir == archive_root / "logs"


def test_experiment_archive_writes_expected_files(monkeypatch, tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("DGFM_ARCHIVE_ROOT", str(archive_root))
    roots = resolve_run_roots(tmp_path / "run")
    archive = build_experiment_archive(roots)
    archive.dump_yaml("config_resolved.yaml", {"train": {"epochs": 1}})
    archive.append_jsonl("train.jsonl", {"epoch": 0, "train_loss": 1.0})
    archive.save_checkpoint("last.pt", {"epoch": 0, "tensor": torch.ones(1)})

    assert (archive_root / "logs" / "config_resolved.yaml").is_file()
    assert (archive_root / "logs" / "train.jsonl").is_file()
    assert (archive_root / "checkpoints" / "last.pt").is_file()
