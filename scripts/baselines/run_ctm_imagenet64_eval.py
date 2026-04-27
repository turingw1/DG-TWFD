from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
CTM_ROOT = ROOT / "refs" / "ctm" / "code"
EDM_ROOT = ROOT / "refs" / "edm"
FID_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*$")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official CTM ImageNet64 samples and EDM FID eval.")
    parser.add_argument(
        "--checkpoint",
        default="/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/ctm/ctm_imagenet64_ema_0.999.pt",
    )
    parser.add_argument("--method", default="CTM-official")
    parser.add_argument("--sample-root", default="runs/ctm_imagenet64_5k/samples")
    parser.add_argument("--eval-root", default="eval/ctm_imagenet64_5k")
    parser.add_argument("--csv-out", default="results/baselines/baseline_ctm_imagenet64.csv")
    parser.add_argument("--fid-ref", default="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--fid-batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--skip-fid", action="store_true")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def _run_and_tee(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    with log_path.open("w", encoding="utf-8") as handle:
        for line in proc.stdout:
            captured.append(line)
            handle.write(line)
            handle.flush()
            print(line, end="")
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, command, output="".join(captured))
    return "".join(captured)


def _count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*.png") if item.is_file())


def _load_completed_metric(metrics_path: Path, image_dir: Path, *, expected: int) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if row.get("fid") is None:
        return None
    if int(row.get("num_fid_samples", 0) or 0) != expected:
        return None
    if _count_pngs(image_dir) < expected:
        return None
    return row


def _ctm_npz_dirs(raw_root: Path) -> list[Path]:
    dirs: dict[Path, float] = {}
    for path in raw_root.rglob("sample_*.npz"):
        dirs[path.parent] = max(dirs.get(path.parent, 0.0), path.stat().st_mtime)
    return [path for path, _mtime in sorted(dirs.items(), key=lambda item: item[1], reverse=True)]


def _count_npz_images(raw_dir: Path) -> int:
    count = 0
    for path in sorted(raw_dir.glob("sample_*.npz")):
        with np.load(path) as data:
            count += int(data[data.files[0]].shape[0])
    return count


def _convert_npz_chunks_to_pngs(raw_dir: Path, image_dir: Path, *, expected: int) -> int:
    if _count_pngs(image_dir) >= expected:
        return expected
    image_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in sorted(raw_dir.glob("sample_*.npz")):
        with np.load(path) as data:
            arr = data[data.files[0]]
            if arr.ndim != 4 or arr.shape[-1] != 3:
                raise ValueError(f"Expected NHWC image array, got {arr.shape} from {path}")
            for image in arr:
                if written >= expected:
                    return _count_pngs(image_dir)
                subdir = image_dir / f"{written - written % 1000:06d}"
                subdir.mkdir(parents=True, exist_ok=True)
                out = subdir / f"{written:06d}.png"
                if not out.exists():
                    Image.fromarray(image.astype(np.uint8), mode="RGB").save(out)
                written += 1
    return _count_pngs(image_dir)


def _parse_fid(stdout: str) -> float | None:
    for line in reversed(stdout.splitlines()):
        match = FID_RE.match(line)
        if match:
            return float(match.group(1))
    return None


def _sample_command(*, checkpoint: Path, step_count: int, num_samples: int, batch: int, seed: int, raw_root: Path) -> list[str]:
    return [
        "mpiexec",
        "-n",
        "1",
        sys.executable,
        str(CTM_ROOT / "image_sample.py"),
        "--data_name=imagenet64",
        "--attention_type=legacy",
        "--class_cond=True",
        "--num_classes=1000",
        "--eval_batch",
        str(batch),
        "--eval_fid=True",
        "--eval_similarity=False",
        "--check_dm_performance=False",
        "--out_dir",
        str(raw_root),
        "--model_path",
        str(checkpoint),
        "--training_mode=ctm",
        "--eval_num_samples",
        str(num_samples),
        "--batch_size",
        str(batch),
        "--device_id=0",
        "--sampler=exact",
        "--sampling_steps",
        str(step_count),
        "--save_format=npz",
        "--stochastic_seed=False",
        "--use_MPI=True",
        "--generator=determ",
        "--eval_seed",
        str(seed),
    ]


def _write_outputs(*, rows: list[dict[str, Any]], eval_root: Path, csv_out: Path) -> None:
    report_dir = eval_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"records": rows}, handle, indent=2)
    with (report_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": "imagenet64",
                    "method": row["method"],
                    "step": row["step_count"],
                    "fid": "" if row["fid"] is None else f"{float(row['fid']):.6f}",
                    "is": "",
                    "recall": "",
                    "checkpoint": row["checkpoint"],
                    "eval_script": "scripts/baselines/run_ctm_imagenet64_eval.py",
                    "notes": row["notes"],
                }
            )


def main() -> None:
    args = parse_args()
    checkpoint = _resolve(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not CTM_ROOT.exists():
        raise FileNotFoundError(CTM_ROOT)
    if not EDM_ROOT.exists():
        raise FileNotFoundError(EDM_ROOT)

    sample_root = _resolve(args.sample_root)
    eval_root = _resolve(args.eval_root)
    csv_out = _resolve(args.csv_out)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(CTM_ROOT) + os.pathsep + str(EDM_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("DNNLIB_CACHE_DIR", str(ROOT / ".torch" / "dnnlib"))
    env.setdefault("OMPI_MCA_btl", "^openib")
    env.setdefault("OMPI_MCA_btl_openib_warn_no_device_params_found", "0")

    rows: list[dict[str, Any]] = []
    for step_count in args.steps:
        step_dir = eval_root / f"steps{step_count}"
        raw_root = step_dir / "ctm_raw"
        image_dir = sample_root / f"steps{step_count}" / "images"
        step_dir.mkdir(parents=True, exist_ok=True)
        raw_root.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        metrics_path = step_dir / "metrics.json"
        completed_row = _load_completed_metric(metrics_path, image_dir, expected=args.num_samples)
        if completed_row is not None and not args.skip_fid:
            rows.append(completed_row)
            _write_outputs(rows=rows, eval_root=eval_root, csv_out=csv_out)
            print(f"reuse ctm imagenet64 step_count={step_count} fid={completed_row['fid']}", flush=True)
            continue

        raw_dirs = _ctm_npz_dirs(raw_root)
        raw_dir = raw_dirs[0] if raw_dirs else None
        if not args.skip_sample and (raw_dir is None or _count_npz_images(raw_dir) < args.num_samples):
            _run_and_tee(
                _sample_command(
                    checkpoint=checkpoint,
                    step_count=step_count,
                    num_samples=args.num_samples,
                    batch=args.batch,
                    seed=args.seed + 1000 * step_count,
                    raw_root=raw_root,
                ),
                cwd=CTM_ROOT,
                env=env,
                log_path=step_dir / "sample.stdout_stderr.txt",
            )
            raw_dirs = _ctm_npz_dirs(raw_root)
            raw_dir = raw_dirs[0] if raw_dirs else None
        if raw_dir is None:
            raise FileNotFoundError(f"No CTM sample_*.npz found in {raw_root}")

        png_count = _convert_npz_chunks_to_pngs(raw_dir, image_dir, expected=args.num_samples)
        if png_count < args.num_samples:
            raise RuntimeError(f"Only converted {png_count}/{args.num_samples} images for step_count={step_count}")

        fid = None
        if not args.skip_fid:
            fid_stdout = _run_and_tee(
                [
                    sys.executable,
                    str(EDM_ROOT / "fid.py"),
                    "calc",
                    f"--images={image_dir}",
                    f"--ref={args.fid_ref}",
                    f"--num={args.num_samples}",
                    f"--batch={args.fid_batch}",
                ],
                cwd=EDM_ROOT,
                env=env,
                log_path=step_dir / "fid.stdout_stderr.txt",
            )
            fid = _parse_fid(fid_stdout)

        row = {
            "step_count": int(step_count),
            "fid": fid,
            "num_fid_samples": int(args.num_samples),
            "checkpoint": str(checkpoint),
            "method": args.method,
            "raw_sample_dir": str(raw_dir),
            "image_dir": str(image_dir),
            "elapsed_sec": time.time() - t0,
            "notes": "; ".join(
                [
                    "official CTM ImageNet64 checkpoint",
                    "sampler=exact",
                    f"sampling_steps={step_count}",
                    f"num_fid_samples={args.num_samples}",
                    "recall pending ImageNet64 reference sample batch",
                ]
            ),
        }
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2)
        rows.append(row)
        _write_outputs(rows=rows, eval_root=eval_root, csv_out=csv_out)
        print(f"ctm imagenet64 step_count={step_count} fid={fid} elapsed_sec={row['elapsed_sec']:.2f}", flush=True)


if __name__ == "__main__":
    main()
