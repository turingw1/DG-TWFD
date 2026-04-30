#!/usr/bin/env python3
"""Prune redundant training checkpoints while preserving decision nodes.

This is intentionally conservative:
- only checkpoint-like files are touched;
- logs, eval reports, configs, samples, and analysis files are left in place;
- each key checkpoint is copied to the project-stable key-node archive before
  any pruning is applied.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_RUNS = Path("/cache/Zhengwei/DG-TWFD-runtime/runs")
TEMP_PROJECT = Path("/temp/Zhengwei/projects/DG-TWFD")
TEMP_RUNS = TEMP_PROJECT / "critical" / "runs"
TEMP_CKPT = TEMP_PROJECT / "critical" / "ckpt"
KEY_NODE_ROOT = TEMP_CKPT / "key_nodes"
MANIFEST_DIR = REPO_ROOT / "docs" / "experiments" / "DG_TWFD_v3" / "cleanup_manifests"


@dataclass(frozen=True)
class KeepNode:
    rel: str
    reason: str


MAIN_KEEP_NODES: tuple[KeepNode, ...] = (
    KeepNode(
        "edm_first_cifar10_warp_e501a/checkpoints/best.pt",
        "e501a first EDM-map learned-warp viability checkpoint",
    ),
    KeepNode(
        "edm_first_cifar10_warp_e502a/checkpoints/last.pt",
        "e502a stopped continuation checkpoint used for early EDM-map diagnosis",
    ),
    KeepNode(
        "edm_first_cifar10_onestep_prior_e503c/checkpoints/step2000.pt",
        "e503c prior one-step endpoint reference",
    ),
    KeepNode(
        "edm_first_cifar10_onestep_msdefect_e504a/checkpoints/step250.pt",
        "e504a diagnostic threshold reference",
    ),
    KeepNode(
        "edm_first_cifar10_onestep_msdefect_e504a/checkpoints/step1250.pt",
        "e504a restored endpoint checkpoint before resume",
    ),
    KeepNode(
        "edm_first_cifar10_onestep_msdefect_e504a_resume_from1250/checkpoints/step1750.pt",
        "handoff checkpoint from endpoint-only training to full-stack v11a",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v11a_from_step1750/checkpoints/step6750.pt",
        "v11a best composition checkpoint and v12a branch point",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v11a_from_step1750/checkpoints/step8750.pt",
        "v11a endpoint-specialized final evaluated checkpoint",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v12a_from_step6750/checkpoints/step10500.pt",
        "v12a best endpoint and budget-policy checkpoint; v13 branch point",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v13_preserve2_from_step10500/checkpoints/step3500.pt",
        "v13 first all-budget-improving diagnostic checkpoint",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v13_preserve2_from_step10500/checkpoints/step4250.pt",
        "v13 first strong all-budget-improving checkpoint",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v13_preserve2_from_step10500/checkpoints/step7500.pt",
        "v13 best isolated FID@4 checkpoint",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v13_preserve2_from_step10500/checkpoints/step7750.pt",
        "v13 best all-around checkpoint and v14 branch point",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v14_guarded_from_step7750/checkpoints/step10000.pt",
        "v14 best FID@1/FID@2 checkpoint",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v14_guarded_from_step7750/checkpoints/step10500.pt",
        "v14 best FID@4 checkpoint",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v14_guarded_from_step7750/checkpoints/step10750.pt",
        "v14 best overall multi-step checkpoint and v15 branch point",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v14_guarded_from_step7750/checkpoints/step10801.pt",
        "v14 final-checkpoint guard artifact",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v15_multimid_from_step10750/checkpoints/step10000.pt",
        "v15 positive signal checkpoint",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v15_multimid_from_step10750/checkpoints/step11500.pt",
        "v15 best balanced checkpoint",
    ),
    KeepNode(
        "edm_first_cifar10_prior_fullstack_timewarp_v15_multimid_from_step10750/checkpoints/step11855.pt",
        "v15 best high-budget and final evaluated checkpoint",
    ),
)


SIDE_KEEP_NODES: tuple[KeepNode, ...] = (
    KeepNode("fm_cifar10_map_branch_v1/checkpoints/best.pt", "early FM map branch best checkpoint"),
    KeepNode("fm_cifar10_map_branch_timewarp_smoke_tws02/checkpoints/best.pt", "timewarp smoke checkpoint"),
    KeepNode("fm_cifar10_map_branch_timewarp_smoke_tws03/checkpoints/best.pt", "timewarp smoke checkpoint"),
    KeepNode("fm_cifar10_map_branch_timewarp_probe_tw001/checkpoints/best.pt", "timewarp probe checkpoint"),
    KeepNode("fm_cifar10_map_branch_quick_v1/checkpoints/best.pt", "quick branch best checkpoint"),
    KeepNode(
        "ctm_nogan_20260429/cifar10_nogan_dsm_10k_mb4_gb16_resume_from8000/ema_0.999_010000.pt",
        "evaluated CTM no-GAN diagnostic EMA checkpoint",
    ),
)


KEEP_NODES = MAIN_KEEP_NODES + SIDE_KEEP_NODES
KEEP_RELS = {node.rel for node in KEEP_NODES}
RUN_PREFIXES = {node.rel.split("/", 1)[0] for node in KEEP_NODES}
RUN_PREFIXES.add("ctm_nogan_20260429")


def bytes_to_gib(value: int) -> float:
    return value / float(1024**3)


def is_checkpoint_like(path: Path, rel: Path) -> bool:
    if path.suffix not in {".pt", ".pth", ".ckpt"}:
        return False
    parts = set(rel.parts)
    if "checkpoints" in parts:
        return True
    if rel.parts and rel.parts[0] == "ctm_nogan_20260429":
        return True
    return False


def iter_prunable(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for prefix in sorted(RUN_PREFIXES):
        base = root / prefix
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if is_checkpoint_like(path, rel) and str(rel) not in KEEP_RELS:
                candidates.append(path)
    return candidates


def source_for(rel: str) -> Path | None:
    candidates = [
        RUNTIME_RUNS / rel,
        TEMP_RUNS / rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_key_nodes(apply: bool) -> tuple[list[str], list[str]]:
    copied: list[str] = []
    missing: list[str] = []
    for node in KEEP_NODES:
        src = source_for(node.rel)
        dest = KEY_NODE_ROOT / node.rel
        if src is None:
            missing.append(node.rel)
            continue
        needs_copy = not dest.exists() or dest.stat().st_size != src.stat().st_size
        if needs_copy:
            copied.append(node.rel)
            if apply:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                if tmp.exists():
                    tmp.unlink()
                if dest.exists():
                    dest.unlink()
                # The shared /temp filesystem can reject os.replace() even
                # within the same directory, so copy directly after unlinking.
                shutil.copy2(src, dest, follow_symlinks=True)
    return copied, missing


def iter_temp_ckpt_prunable() -> list[Path]:
    protected_prefix = KEY_NODE_ROOT
    keep_specific = {
        TEMP_CKPT / "ctm_nogan_20260429" / "cifar10_nogan_dsm_10k_mb4_gb16_resume_from8000" / "ema_0.999_010000.pt",
        TEMP_CKPT / "edm_first_cifar10_onestep_msdefect_e504a_resume_from1250_step1750.pt",
    }
    candidates: list[Path] = []
    if not TEMP_CKPT.exists():
        return candidates
    for path in TEMP_CKPT.rglob("*"):
        if not path.is_file() or path.suffix not in {".pt", ".pth", ".ckpt"}:
            continue
        if protected_prefix in path.parents or path in keep_specific:
            continue
        if path.parts and "ctm_nogan_20260429" in path.parts:
            candidates.append(path)
    return candidates


def delete_files(paths: list[Path], apply: bool) -> tuple[int, int]:
    total = 0
    count = 0
    for path in paths:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        total += size
        count += 1
        if apply:
            path.unlink()
    return count, total


def summarize_paths(paths: list[Path], root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(paths):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = 0
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        rows.append({"path": rel, "bytes": size, "gib": round(bytes_to_gib(size), 4)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually copy key nodes and delete redundant checkpoint files.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest path.")
    args = parser.parse_args()

    copied, missing = ensure_key_nodes(apply=args.apply)
    runtime_delete = iter_prunable(RUNTIME_RUNS)
    temp_runs_delete = iter_prunable(TEMP_RUNS)
    temp_ckpt_delete = iter_temp_ckpt_prunable()

    runtime_count, runtime_bytes = delete_files(runtime_delete, apply=args.apply)
    temp_runs_count, temp_runs_bytes = delete_files(temp_runs_delete, apply=args.apply)
    temp_ckpt_count, temp_ckpt_bytes = delete_files(temp_ckpt_delete, apply=args.apply)
    deleted_bytes = runtime_bytes + temp_runs_bytes + temp_ckpt_bytes

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "apply" if args.apply else "dry-run",
        "key_nodes": [{"path": node.rel, "reason": node.reason} for node in KEEP_NODES],
        "key_nodes_to_copy": copied,
        "missing_key_nodes": missing,
        "delete_summary": {
            "runtime_runs": {"count": runtime_count, "gib": round(bytes_to_gib(runtime_bytes), 3)},
            "temp_critical_runs": {"count": temp_runs_count, "gib": round(bytes_to_gib(temp_runs_bytes), 3)},
            "temp_critical_ckpt": {"count": temp_ckpt_count, "gib": round(bytes_to_gib(temp_ckpt_bytes), 3)},
            "total": {"count": runtime_count + temp_runs_count + temp_ckpt_count, "gib": round(bytes_to_gib(deleted_bytes), 3)},
        },
        "runtime_delete": summarize_paths(runtime_delete, RUNTIME_RUNS),
        "temp_critical_runs_delete": summarize_paths(temp_runs_delete, TEMP_RUNS),
        "temp_critical_ckpt_delete": summarize_paths(temp_ckpt_delete, TEMP_CKPT),
    }

    manifest_path = args.manifest
    if manifest_path is None:
        suffix = "apply" if args.apply else "dry_run"
        manifest_path = MANIFEST_DIR / f"checkpoint_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}.json"
    if args.apply or not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest["delete_summary"], indent=2))
    print(f"manifest: {manifest_path}")
    if missing:
        print("missing key nodes:")
        for item in missing:
            print(f"  {item}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
