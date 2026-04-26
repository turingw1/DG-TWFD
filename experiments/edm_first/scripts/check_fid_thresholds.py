from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_records(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["step_count"]): item for item in payload["records"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare an EDM-first eval summary against a FID reduction target.")
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target-ratio", type=float, default=0.5)
    parser.add_argument("--primary-step", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = _load_records(args.baseline_summary)
    current = _load_records(args.summary)
    rows = []
    for step, base_row in sorted(baseline.items()):
        if step not in current:
            continue
        baseline_fid = float(base_row["fid"])
        current_fid = float(current[step]["fid"])
        target_fid = baseline_fid * float(args.target_ratio)
        rows.append(
            {
                "step_count": step,
                "baseline_fid": baseline_fid,
                "current_fid": current_fid,
                "target_fid": target_fid,
                "ratio_current_over_baseline": current_fid / baseline_fid if baseline_fid else None,
                "target_met": current_fid <= target_fid,
                "primary_step": step == int(args.primary_step),
            }
        )
    primary = next((row for row in rows if row["primary_step"]), None)
    payload = {
        "baseline_summary": str(args.baseline_summary),
        "summary": str(args.summary),
        "target_ratio": float(args.target_ratio),
        "primary_step": int(args.primary_step),
        "primary_target_met": bool(primary and primary["target_met"]),
        "any_target_met": any(bool(row["target_met"]) for row in rows),
        "records": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "threshold_verdict.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.out_dir / "threshold_verdict.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["step_count"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        "fid_threshold "
        f"primary_step={args.primary_step} "
        f"primary_target_met={payload['primary_target_met']} "
        f"any_target_met={payload['any_target_met']}"
    )


if __name__ == "__main__":
    main()
