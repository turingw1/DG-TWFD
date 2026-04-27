from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

FIELDNAMES = [
    "dataset",
    "method",
    "step",
    "fid",
    "is",
    "recall",
    "checkpoint",
    "eval_script",
    "notes",
]

REQUESTED_OUTPUTS = [
    "baseline_edm_cifar10.csv",
    "baseline_edm_imagenet64.csv",
    "baseline_ctm_imagenet64.csv",
    "baseline_ctm_cifar10.csv",
    "baseline_cd_imagenet64.csv",
    "schedule_ays_cifar10.csv",
    "schedule_ays_imagenet64.csv",
    "schedule_optimalsteps_cifar10.csv",
    "schedule_optimalsteps_imagenet64.csv",
    "schedule_entropic_cifar10.csv",
    "schedule_entropic_imagenet64.csv",
    "baseline_tcm_cifar10.csv",
    "baseline_tcm_imagenet64.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export available baseline reports to the paper-table CSV schema."
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "results" / "baselines"),
        help="Directory for unified baseline CSV files.",
    )
    parser.add_argument(
        "--write-empty",
        action="store_true",
        help="Also write header-only CSV files for requested baselines that are not available yet.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _records_from_summary(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, dict):
        records = payload.get("records", [])
    else:
        records = payload
    if not isinstance(records, list):
        raise TypeError(f"summary is not a list: {path}")
    return [dict(item) for item in records]


def _format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def _edm_cifar10_rows() -> list[dict[str, str]]:
    summary = _best_existing_summary(
        [
            ROOT / "eval" / "edm_cifar10_public_eval_5k" / "reports" / "summary.json",
            ROOT / "eval" / "edm_cifar10_public_eval_full" / "reports" / "summary.json",
            ROOT / "eval" / "edm_cifar10_public_eval_e501full" / "reports" / "summary.json",
            ROOT / "eval" / "edm_cifar10_public_eval_e501ref" / "reports" / "summary.json",
        ]
    )
    if not summary.exists():
        return []
    return _edm_rows(
        summary=summary,
        dataset="cifar10",
        eval_script="scripts/run_edm_cifar10_eval.py",
    )


def _edm_imagenet64_rows() -> list[dict[str, str]]:
    summary = _best_existing_summary(
        [
            ROOT / "eval" / "edm_imagenet64_public_eval_5k" / "reports" / "summary.json",
            ROOT / "eval" / "edm_imagenet64_public_eval_full" / "reports" / "summary.json",
            ROOT / "eval" / "edm_imagenet64_public_eval_e501full" / "reports" / "summary.json",
            ROOT / "eval" / "edm_imagenet64_public_eval_e501ref" / "reports" / "summary.json",
        ]
    )
    if not summary.exists():
        return []
    return _edm_rows(
        summary=summary,
        dataset="imagenet64",
        eval_script="scripts/run_edm_cifar10_eval.py",
    )


def _edm_rows(*, summary: Path, dataset: str, eval_script: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in _records_from_summary(summary):
        notes = [
            "official EDM sampler",
            f"num_fid_samples={record.get('num_fid_samples', '')}",
            f"nfe={record.get('nfe', '')}",
            f"fid_ref={record.get('fid_ref', '')}",
        ]
        rows.append(
            {
                "dataset": dataset,
                "method": "EDM-official",
                "step": str(record.get("step_count", "")),
                "fid": _format_float(record.get("fid")),
                "is": "",
                "recall": "",
                "checkpoint": str(record.get("network", "")),
                "eval_script": eval_script,
                "notes": "; ".join(notes),
            }
        )
    return rows


def _best_existing_summary(candidates: list[Path]) -> Path:
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return candidates[0]

    def score(path: Path) -> tuple[int, int]:
        try:
            records = _records_from_summary(path)
        except Exception:
            return (0, 0)
        sample_count = 0
        steps = 0
        for record in records:
            if record.get("fid") is not None:
                steps += 1
            try:
                sample_count = max(sample_count, int(record.get("num_fid_samples", 0)))
            except Exception:
                pass
        return (sample_count, steps)

    return max(existing, key=score)


def _optimalsteps_cifar10_rows() -> list[dict[str, str]]:
    comparison = (
        ROOT
        / "results"
        / "time_coordinate_ablation"
        / "e405b_optimalsteps_like_vs_timewarp_20260426"
        / "comparison.csv"
    )
    if not comparison.exists():
        return []
    rows: list[dict[str, str]] = []
    with comparison.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            notes = [
                "preliminary OSS-like schedule adapted from refs/optimalsteps",
                "not paper-ready external reproduction",
                "checkpoint=e405b DDPM/DGTD failed sample-quality gate",
                f"num_fid_samples={record.get('num_fid_samples', '')}",
                f"schedule_json={record.get('schedule_json', '')}",
            ]
            rows.append(
                {
                    "dataset": "cifar10",
                    "method": "OptimalSteps-like",
                    "step": str(record.get("step_count", "")),
                    "fid": _format_float(record.get("optimalsteps_fid")),
                    "is": "",
                    "recall": "",
                    "checkpoint": "runs/dgtd_cifar10_v3_probe_fast_teacher_e405b/checkpoints/best.pt",
                    "eval_script": "scripts/run_time_coordinate_ablation.py",
                    "notes": "; ".join(notes),
                }
            )
    return rows


def _cd_imagenet64_rows() -> list[dict[str, str]]:
    summary = _best_existing_summary(
        [
            ROOT / "eval" / "cd_imagenet64_lpips_5k" / "reports" / "summary.json",
            ROOT / "eval" / "cd_imagenet64_l2_5k" / "reports" / "summary.json",
            ROOT / "eval" / "cd_imagenet64_lpips_full" / "reports" / "summary.json",
            ROOT / "eval" / "cd_imagenet64_l2_full" / "reports" / "summary.json",
            ROOT / "eval" / "cd_imagenet64_full" / "reports" / "summary.json",
        ]
    )
    if not summary.exists():
        return []
    rows: list[dict[str, str]] = []
    for record in _records_from_summary(summary):
        notes = str(record.get("notes", ""))
        rows.append(
            {
                "dataset": "imagenet64",
                "method": str(record.get("method", "CD-LPIPS-official")),
                "step": str(record.get("step_count", "")),
                "fid": _format_float(record.get("fid")),
                "is": "",
                "recall": "",
                "checkpoint": str(record.get("checkpoint", "")),
                "eval_script": "scripts/baselines/run_cd_imagenet64_eval.py",
                "notes": notes,
            }
        )
    return rows


def build_outputs() -> dict[str, list[dict[str, str]]]:
    return {
        "baseline_edm_cifar10.csv": _edm_cifar10_rows(),
        "baseline_edm_imagenet64.csv": _edm_imagenet64_rows(),
        "baseline_cd_imagenet64.csv": _cd_imagenet64_rows(),
        "schedule_optimalsteps_cifar10.csv": _optimalsteps_cifar10_rows(),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    outputs = build_outputs()
    names = REQUESTED_OUTPUTS if args.write_empty else sorted(outputs)
    manifest: dict[str, dict[str, Any]] = {}

    for name in names:
        rows = outputs.get(name, [])
        if rows or args.write_empty:
            path = output_root / name
            _write_csv(path, rows)
            manifest[name] = {"rows": len(rows), "path": str(path)}

    manifest_path = output_root / "MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    for name, item in manifest.items():
        print(f"{name}: rows={item['rows']} path={item['path']}")


if __name__ == "__main__":
    main()
