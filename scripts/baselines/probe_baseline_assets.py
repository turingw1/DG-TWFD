from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path(os.environ.get("DG_TWFD_RUNTIME_ROOT", "/cache/Zhengwei/DG-TWFD-runtime"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe external baseline code/checkpoint assets.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "baselines" / "asset_probe.json"),
        help="JSON report path.",
    )
    return parser.parse_args()


def _exists(path: Path) -> dict:
    return {"path": str(path), "exists": path.exists()}


def _glob_many(patterns: list[str], roots: list[Path], *, limit: int = 40) -> list[str]:
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches.extend(root.glob(pattern))
    unique = sorted({path.resolve() for path in matches if path.is_file()})
    return [str(path) for path in unique[:limit]]


def main() -> None:
    args = parse_args()
    checkpoint_roots = [
        ROOT / "author_ckpt",
        ROOT / "checkpoints",
        RUNTIME / "author_ckpt",
        RUNTIME / "checkpoints",
        RUNTIME / ".torch" / "dnnlib" / "downloads",
    ]
    report = {
        "roots": {
            "repo": str(ROOT),
            "runtime": str(RUNTIME),
            "checkpoint_roots": [str(path) for path in checkpoint_roots],
        },
        "repos": {
            "edm": _exists(ROOT / "refs" / "edm"),
            "ctm": _exists(ROOT / "refs" / "ctm"),
            "ctm_cifar10": _exists(ROOT / "refs" / "ctm-cifar10"),
            "consistency_models": _exists(ROOT / "refs" / "consistency_models"),
            "optimalsteps": _exists(ROOT / "refs" / "optimalsteps"),
            "entropic_time_schedulers": _exists(ROOT / "refs" / "entropic_time_schedulers"),
            "tcm": _exists(ROOT / "refs" / "tcm"),
        },
        "edm": {
            "cifar10_cached": _glob_many(["*edm-cifar10-32x32-cond-vp.pkl", "*cifar10-32x32.npz"], checkpoint_roots),
            "imagenet64_cached": _glob_many(
                ["*edm-imagenet-64x64-cond-adm.pkl", "*imagenet-64x64.npz"], checkpoint_roots
            ),
        },
        "ctm": {
            "cifar10_candidates": _glob_many(["*ctm*cifar*.pt", "*cifar*ctm*.pt", "*ctm*cifar*.pkl"], checkpoint_roots),
            "imagenet64_candidates": _glob_many(
                ["*ctm*imagenet*.pt", "*imagenet*ctm*.pt", "ema_0.999_*.pt"], checkpoint_roots
            ),
        },
        "consistency_models": {
            "imagenet64_candidates": _glob_many(
                [
                    "edm_imagenet64_ema.pt",
                    "cd_imagenet64_l2.pt",
                    "cd_imagenet64_lpips.pt",
                    "ct_imagenet64.pt",
                ],
                checkpoint_roots,
            ),
            "official_urls": {
                "edm_imagenet64_ema": "https://openaipublic.blob.core.windows.net/consistency/edm_imagenet64_ema.pt",
                "cd_imagenet64_l2": "https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_l2.pt",
                "cd_imagenet64_lpips": "https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_lpips.pt",
                "ct_imagenet64": "https://openaipublic.blob.core.windows.net/consistency/ct_imagenet64.pt",
            },
        },
        "schedule": {
            "entropic_schedules": _glob_many(["EDM/Schedules/*.pt"], [ROOT / "refs" / "entropic_time_schedulers"]),
            "optimalsteps_repo": str(ROOT / "refs" / "optimalsteps"),
        },
        "tcm": {
            "cifar10_candidates": _glob_many(["*cifar*stage*.pkl", "*tcm*cifar*.pkl"], checkpoint_roots),
            "imagenet64_candidates": _glob_many(["*imgnet*stage*.pkl", "*imagenet*stage*.pkl"], checkpoint_roots),
            "checkpoint_source": "https://drive.google.com/drive/folders/1gw6OMKCKaEe3LxSSlJKNhwG-M92u9DsW?usp=sharing",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(output)


if __name__ == "__main__":
    main()
