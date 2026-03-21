from dgfm.config import load_experiment_config


def test_experiment_config_overrides() -> None:
    cfg = load_experiment_config(
        "configs/experiment/fm_cifar10_baseline.yaml",
        overrides=["train.epochs=1", "train.batch_size=8", "experiment.name='fm_test'"]
    )
    assert cfg["train"]["epochs"] == 1
    assert cfg["train"]["batch_size"] == 8
    assert cfg["experiment"]["name"] == "fm_test"
