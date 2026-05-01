#!/usr/bin/env python3
"""Generate CIFAR-10 qualitative samples from OpenAI JAX consistency models.

The public CIFAR-10 JAX checkpoints are unconditional with respect to class
labels. The generated rows are therefore seed-locked across CD/CT variants,
but they are not class-locked to the conditional EDM/DG-TWFD/CTM CIFAR rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import haiku as hk
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
JCM_ROOT = ROOT / "refs" / "consistency_models_cifar10"
if str(JCM_ROOT) not in sys.path:
    sys.path.insert(0, str(JCM_ROOT))

from configs import cifar10_ve_cd, cifar10_ve_ct_adaptive  # noqa: E402
from jcm import checkpoints, losses, sde_lib  # noqa: E402
from jcm.models import ncsnpp  # noqa: F401,E402 - registers the model
from jcm.models import utils as mutils  # noqa: E402


ROWS = {
    "cd_lpips_cifar10_jax": {
        "display_name": "CD-LPIPS CIFAR-10 JAX",
        "config": "cifar10_ve_cd",
        "config_overrides": {"training.loss_norm": "lpips", "optim.lr": 4e-4},
        "workdir": "/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/consistency_models_cifar10/cd-lpips",
        "checkpoint_step": 80,
    },
    "cd_l2_cifar10_jax": {
        "display_name": "CD-L2 CIFAR-10 JAX",
        "config": "cifar10_ve_cd",
        "config_overrides": {"training.loss_norm": "l2", "optim.lr": 8e-5},
        "workdir": "/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/consistency_models_cifar10/cd-l2",
        "checkpoint_step": 80,
    },
    "ct_lpips_cifar10_jax": {
        "display_name": "CT-LPIPS CIFAR-10 JAX",
        "config": "cifar10_ve_ct_adaptive",
        "config_overrides": {
            "training.loss_norm": "lpips",
            "training.dsm_target": True,
            "training.start_scales": 2,
            "training.end_scales": 150,
            "training.start_ema": 0.9,
            "optim.lr": 4e-4,
        },
        "workdir": "/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/consistency_models_cifar10/ct-lpips",
        "checkpoint_step": 74,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260501",
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--sample-seeds", nargs="+", type=int, default=[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007])
    parser.add_argument("--rows", nargs="+", default=list(ROWS))
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _get_config(name: str):
    if name == "cifar10_ve_cd":
        return cifar10_ve_cd.get_config()
    if name == "cifar10_ve_ct_adaptive":
        return cifar10_ve_ct_adaptive.get_config()
    raise ValueError(f"unknown config: {name}")


def _set_config_value(config: Any, dotted_key: str, value: Any) -> None:
    cur = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


def _init_model(rng, config: Any):
    """Initialize a JCM model with a Flax-version-tolerant state split."""
    rng = hk.PRNGSequence(rng)
    model_def = mutils.get_model(config.model.name)
    input_shape = (
        jax.local_device_count(),
        config.data.image_size,
        config.data.image_size,
        config.data.num_channels,
    )
    fake_input = jnp.zeros(input_shape)
    fake_label = jnp.zeros(input_shape[:1], dtype=jnp.int32)
    model = model_def(config=config)
    variables = model.init({"params": next(rng), "dropout": next(rng)}, fake_input, fake_label)
    variables_mut = unfreeze(variables)
    initial_params = freeze(variables_mut.pop("params"))
    init_model_state = freeze(variables_mut)
    return model, init_model_state, initial_params


def _load_state(config: Any, workdir: Path, step: int):
    rng = hk.PRNGSequence(config.seed + 1)
    model, init_model_state, initial_params = _init_model(next(rng), config)
    optimizer, _optimize_fn = losses.get_optimizer(config)
    state_cls = (
        mutils.StateWithTarget
        if config.training.loss.lower().endswith(("ema", "adaptive", "progressive_distillation"))
        else mutils.State
    )
    common = {
        "step": 0,
        "lr": config.optim.lr,
        "ema_rate": config.model.ema_rate,
        "params": initial_params,
        "params_ema": initial_params,
        "model_state": init_model_state,
        "opt_state": optimizer.init(initial_params),
        "rng_state": rng.internal_state,
    }
    if state_cls is mutils.StateWithTarget:
        common["target_params"] = initial_params
    state = state_cls(**common)
    state = checkpoints.restore_checkpoint(str(workdir / "checkpoints"), state, step=step)
    return model, state


def _time_indices(step_count: int, train_n: int) -> list[int]:
    if step_count < 1:
        raise ValueError(f"step_count must be positive, got {step_count}")
    raw = np.linspace(0, train_n - 1, step_count + 1)
    indices = [int(round(item)) for item in raw]
    indices[0] = 0
    indices[-1] = train_n - 1
    for idx in range(1, len(indices)):
        if indices[idx] <= indices[idx - 1]:
            indices[idx] = indices[idx - 1] + 1
    if indices[-1] != train_n - 1 or max(indices) > train_n - 1:
        raise ValueError(f"invalid time index grid for {step_count} steps and train_n={train_n}: {indices}")
    return indices


def _sigmas_from_indices(config: Any, indices: list[int]) -> list[float]:
    t_max_rho = float(config.model.t_max) ** (1.0 / float(config.model.rho))
    t_min_rho = float(config.model.t_min) ** (1.0 / float(config.model.rho))
    denom = float(config.model.num_scales - 1)
    return [
        float((t_max_rho + idx / denom * (t_min_rho - t_max_rho)) ** float(config.model.rho))
        for idx in indices
    ]


def _latents(seeds: list[int], shape: tuple[int, int, int]) -> np.ndarray:
    values = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        values.append(rng.standard_normal(shape, dtype=np.float32))
    return np.stack(values, axis=0)


def _save_images(samples: np.ndarray, sample_dir: Path, seeds: list[int]) -> list[str]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(samples)
    if samples.min() < -0.01:
        samples = (samples + 1.0) / 2.0
    samples = np.clip(samples, 0.0, 1.0)
    written = []
    for image, seed in zip(samples, seeds):
        arr = (image * 255.0 + 0.5).astype(np.uint8)
        path = sample_dir / f"{int(seed):06d}.png"
        Image.fromarray(arr, mode="RGB").save(path)
        written.append(str(path.relative_to(ROOT)))
    return written


def _make_sampler(config: Any, model: Any, state: Any):
    sde = sde_lib.get_sde(config)
    model_fn = mutils.get_distiller_fn(
        sde,
        model,
        state.params_ema,
        state.model_state,
        train=False,
        return_state=False,
    )

    @jax.jit
    def transition(x, noise, t, next_t):
        vec_t = jnp.ones((x.shape[0],), dtype=x.dtype) * t
        x0 = jnp.clip(model_fn(x, vec_t), -1.0, 1.0)
        next_t = jnp.clip(next_t, sde.t_min, sde.t_max)
        noise_scale = jnp.sqrt(jnp.maximum(next_t**2 - sde.t_min**2, 0.0))
        return x0 + noise * noise_scale

    return transition


def _sample_row(
    *,
    config: Any,
    model: Any,
    state: Any,
    step_count: int,
    seeds: list[int],
) -> tuple[np.ndarray, list[int], list[float]]:
    transition = _make_sampler(config, model, state)
    indices = _time_indices(step_count, int(config.model.num_scales))
    sigmas = _sigmas_from_indices(config, indices)
    x = jnp.asarray(_latents(seeds, (config.data.image_size, config.data.image_size, config.data.num_channels)))
    x = x * float(config.sampling.std)
    for hop, (t, next_t) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        noise = jnp.asarray(_latents([seed + 100000 * (hop + 1) for seed in seeds], x.shape[1:]))
        x = transition(x, noise, jnp.asarray(t, dtype=x.dtype), jnp.asarray(next_t, dtype=x.dtype))
    return np.asarray(jnp.clip(x, -1.0, 1.0)), indices, sigmas


def main() -> None:
    args = _parse_args()
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in args.sample_seeds]
    steps = [int(step) for step in args.steps]

    manifest: dict[str, Any] = {
        "note": "OpenAI CIFAR-10 JAX consistency rows are seed-locked only; the released models are not class-label conditional.",
        "backend": str(jax.default_backend()),
        "devices": [str(device) for device in jax.devices()],
        "steps": steps,
        "sample_seeds": seeds,
        "rows": [],
    }

    for row_name in args.rows:
        row = ROWS[row_name]
        print(f"[load] {row_name}: {row['workdir']} checkpoint_{row['checkpoint_step']}", flush=True)
        config = _get_config(row["config"])
        for dotted_key, value in row["config_overrides"].items():
            _set_config_value(config, dotted_key, value)
        model, state = _load_state(config, _resolve(row["workdir"]), int(row["checkpoint_step"]))
        for step_count in steps:
            print(f"[sample] {row_name} steps={step_count}", flush=True)
            samples, time_indices, sigmas = _sample_row(
                config=config,
                model=model,
                state=state,
                step_count=step_count,
                seeds=seeds,
            )
            sample_dir = output_root / row_name / f"steps{step_count}"
            written = _save_images(samples, sample_dir, seeds)
            manifest["rows"].append(
                {
                    "row": row_name,
                    "display_name": row["display_name"],
                    "step_count": int(step_count),
                    "sampling": "official JCM stochastic iterative sampler adapted from editing_multistep_sampling.ipynb",
                    "class_conditioning": "none; seed-locked only",
                    "config": row["config"],
                    "config_overrides": row["config_overrides"],
                    "workdir": str(_resolve(row["workdir"])),
                    "checkpoint_step": int(row["checkpoint_step"]),
                    "time_indices": time_indices,
                    "sigma_grid": sigmas,
                    "sample_dir": str(sample_dir.relative_to(ROOT)),
                    "files": written,
                }
            )

    manifest_path = output_root / "consistency_cifar10_jax_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
