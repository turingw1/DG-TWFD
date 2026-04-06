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


def test_load_experiment_config_expands_bash_style_default_env(monkeypatch) -> None:
    monkeypatch.delenv("TRAJ_ROOT", raising=False)
    cfg = load_experiment_config("configs/experiment/fm_cifar10_map_branch.yaml")
    assert cfg["target"]["shard_root"] == "/cache/Zhengwei/dgfm_teacher_traj/cifar10_ddpm128_p33"

    monkeypatch.setenv("TRAJ_ROOT", "/custom/traj")
    cfg = load_experiment_config("configs/experiment/fm_cifar10_map_branch.yaml")
    assert cfg["target"]["shard_root"] == "/custom/traj"
