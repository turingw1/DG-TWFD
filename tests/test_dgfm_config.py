from pathlib import Path

from dgfm.config import load_experiment_config, resolve_run_roots


def test_load_experiment_config_merges_base_and_includes() -> None:
    cfg = load_experiment_config("configs/experiment/fm_cifar10_baseline.yaml")
    assert cfg["experiment"]["name"] == "fm_cifar10_baseline"
    assert cfg["dataset"]["name"] == "cifar10"
    assert cfg["model"]["family"] == "official_fm_unet"
    assert cfg["teacher"]["type"] == "none"
    assert cfg["eval"]["step_counts"] == [1, 2, 4, 8, 16]


def test_resolve_run_roots() -> None:
    roots = resolve_run_roots("/tmp/dgfm_test")
    assert roots.run_root == Path("/tmp/dgfm_test")
    assert roots.checkpoint_dir == Path("/tmp/dgfm_test/checkpoints")
