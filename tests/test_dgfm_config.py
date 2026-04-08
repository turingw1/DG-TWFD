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
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("TRAJ_ROOT", raising=False)
    cfg = load_experiment_config("configs/experiment/fm_cifar10_map_branch.yaml")
    assert cfg["target"]["builder"] == "teacher_sampler"
    assert cfg["target"]["target_construction"] == "ctm_consistency"
    assert cfg["target"]["target_source"] == "ema_model"
    assert cfg["target"]["bridge_source"] == "ema_model_rollout"
    assert cfg["teacher"]["name_or_path"] == "google/ddpm-cifar10-32"

    fallback_cfg = load_experiment_config("configs/target/teacher_trajectory.yaml")
    assert fallback_cfg["target"]["shard_root"] == "/cache/Zhengwei/dgfm_teacher_traj/cifar10_ddpm128_p33"
    monkeypatch.setenv("TRAJ_ROOT", "/custom/traj")
    fallback_cfg = load_experiment_config("configs/target/teacher_trajectory.yaml")
    assert fallback_cfg["target"]["shard_root"] == "/custom/traj"


def test_load_quick_map_branch_config_overrides_teacher_runtime() -> None:
    cfg = load_experiment_config("configs/experiment/fm_cifar10_map_branch_quick.yaml")
    assert cfg["dataset"]["name"] == "cifar10"
    assert cfg["model"]["family"] == "official_map_unet"
    assert cfg["teacher"]["num_inference_steps"] == 32
    assert cfg["target"]["sampling_mode"] == "ctm_discrete"
    assert cfg["target"]["start_scales"] == 18
    assert cfg["target"]["num_heun_step"] == 8
    assert cfg["loss"]["endpoint_every"] == 16
    assert cfg["train"]["epochs"] == 30


def test_load_timewarp_probe_config_enables_learnable_warp() -> None:
    cfg = load_experiment_config("configs/experiment/fm_cifar10_map_branch_timewarp_probe.yaml")
    assert cfg["experiment"]["name"] == "fm_cifar10_map_branch_timewarp_probe"
    assert cfg["scheduler"]["timewarp"]["enabled"] is True
    assert cfg["scheduler"]["timewarp"]["type"] == "learnable_monotone"
    assert cfg["loss"]["timewarp_weight"] == 1.0
    assert cfg["train"]["epochs"] == 30


def test_load_timewarp_smoke_config_reduces_runtime_scale() -> None:
    cfg = load_experiment_config("configs/experiment/fm_cifar10_map_branch_timewarp_smoke.yaml")
    assert cfg["experiment"]["name"] == "fm_cifar10_map_branch_timewarp_smoke"
    assert cfg["scheduler"]["timewarp"]["enabled"] is True
    assert cfg["teacher"]["num_inference_steps"] == 16
    assert cfg["target"]["start_scales"] == 12
    assert cfg["train"]["batch_size"] == 32
    assert cfg["train"]["max_train_batches"] == 128
    assert cfg["eval"]["num_fid_samples"] == 1000


def test_load_stage1_smoke_target_configs() -> None:
    traj = load_experiment_config("configs/experiment/fm_cifar10_map_branch_s1_e1_traj_reg.yaml")
    assert traj["target"]["target_construction"] == "trajectory_regression"
    assert traj["target"]["bridge_source"] == "teacher"
    assert traj["loss"]["endpoint_weight"] == 0.0
    assert traj["loss"]["timewarp_weight"] == 0.0

    ctm = load_experiment_config("configs/experiment/fm_cifar10_map_branch_s1_e1_ctm_ema.yaml")
    assert ctm["target"]["target_construction"] == "ctm_consistency"
    assert ctm["target"]["target_source"] == "ema_model"
    assert ctm["target"]["bridge_source"] == "ema_model_rollout"
    assert ctm["train"]["epochs"] == 8


def test_load_stage1_warp_and_budget_configs() -> None:
    learned = load_experiment_config("configs/experiment/fm_cifar10_map_branch_s1_e5_warp_learned.yaml")
    assert learned["scheduler"]["timewarp"]["enabled"] is True
    assert learned["scheduler"]["timewarp"]["type"] == "learnable_monotone"
    assert learned["loss"]["timewarp_weight"] == 1.0

    quick = load_experiment_config("configs/experiment/fm_cifar10_map_branch_s1_e6_budget_quick.yaml")
    assert quick["target"]["bridge_source"] == "ema_model_rollout"
    assert quick["loss"]["endpoint_weight"] == 0.0
    assert quick["eval"]["step_counts"] == [1, 2, 4, 8, 16, 32, 64, 128, 256]

    full = load_experiment_config("configs/experiment/fm_cifar10_map_branch_s1_e6_budget_full.yaml")
    assert full["experiment"]["name"] == "fm_cifar10_map_branch_s1_e6_budget_full"
    assert full["loss"]["timewarp_weight"] == 0.0


def test_load_stage1_spline_and_stage2_external_configs() -> None:
    spline = load_experiment_config("configs/experiment/fm_cifar10_map_branch_s1_e5_warp_spline.yaml")
    assert spline["scheduler"]["timewarp"]["type"] == "spline_mass"
    assert spline["loss"]["timewarp_weight"] == 1.0

    metrics = load_experiment_config("configs/experiment/fm_cifar10_map_branch_s2_official_metrics.yaml")
    assert metrics["eval"]["official_metrics"] == ["fid", "is", "precision", "recall"]
    assert metrics["eval"]["official_num_samples"] == 50000

    defect = load_experiment_config("configs/experiment/fm_cifar10_map_branch_s2_defect_eval.yaml")
    assert defect["eval"]["defect_num_samples"] == 4096
    assert defect["eval"]["defect_grid_steps"] == 16

    imagenet = load_experiment_config("configs/experiment/fm_imagenet64_baseline_smoke.yaml")
    assert imagenet["dataset"]["name"] == "imagenet64"
    assert imagenet["model"]["num_classes"] == 1000
