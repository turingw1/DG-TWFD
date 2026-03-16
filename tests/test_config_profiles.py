from __future__ import annotations

import os

from dg_twfd.config import load_config


def test_profile_extends_and_env_expansion() -> None:
    original = {key: os.environ.get(key) for key in ("EXP_NAME", "SHARD_ROOT", "CKPT_DIR", "TEACHER_ID")}
    os.environ["EXP_NAME"] = "unit_profile_exp"
    os.environ["SHARD_ROOT"] = "/tmp/unit_shards"
    os.environ["CKPT_DIR"] = "/tmp/unit_ckpts"
    os.environ["TEACHER_ID"] = "unit/teacher"
    try:
        cfg = load_config("train_a100_stable")
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert cfg.experiment.name == "unit_profile_exp"
    assert cfg.data.trajectory_shard_dir == "/tmp/unit_shards"
    assert cfg.train.checkpoint_dir == "/tmp/unit_ckpts"
    assert cfg.teacher.pretrained_model_name_or_path == "unit/teacher"
    assert cfg.loss.composition_weight == 0.25
    assert cfg.model.residual_tanh_scale == 0.75
