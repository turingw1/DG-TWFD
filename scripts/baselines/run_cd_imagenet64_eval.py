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
CM_ROOT = ROOT / "refs" / "consistency_models"
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
    parser = argparse.ArgumentParser(description="Run OpenAI CD ImageNet64 samples and EDM FID eval.")
    parser.add_argument(
        "--checkpoint",
        default="/cache/Zhengwei/DG-TWFD-runtime/checkpoints/baselines/consistency_models/cd_imagenet64_lpips.pt",
    )
    parser.add_argument("--method", default="CD-LPIPS-official")
    parser.add_argument("--sample-root", default="runs/cd_imagenet64_lpips_full/samples")
    parser.add_argument("--eval-root", default="eval/cd_imagenet64_lpips_full")
    parser.add_argument("--csv-out", default="results/baselines/baseline_cd_imagenet64.csv")
    parser.add_argument(
        "--fid-ref",
        default="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz",
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fid-batch", type=int, default=16)
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


def _ts_for_step_count(step_count: int) -> str:
    if step_count <= 1:
        return ""
    if step_count == 2:
        return "0,22,39"
    values = np.rint(np.linspace(0, 39, step_count + 1)).astype(int).tolist()
    deduped: list[int] = []
    for value in values:
        if not deduped or value != deduped[-1]:
            deduped.append(value)
    deduped[0] = 0
    deduped[-1] = 39
    return ",".join(str(value) for value in deduped)


def _count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*.png") if item.is_file())


def _find_sample_npz(log_dir: Path) -> Path | None:
    candidates = sorted(log_dir.glob("samples_*.npz"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _convert_npz_to_pngs(npz_path: Path, image_dir: Path, *, expected: int) -> int:
    if _count_pngs(image_dir) >= expected:
        return expected
    image_dir.mkdir(parents=True, exist_ok=True)
    arr = np.load(npz_path)["arr_0"]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected NHWC image array, got {arr.shape} from {npz_path}")
    count = min(expected, int(arr.shape[0]))
    for idx in range(count):
        subdir = image_dir / f"{idx - idx % 1000:06d}"
        subdir.mkdir(parents=True, exist_ok=True)
        out = subdir / f"{idx:06d}.png"
        if out.exists():
            continue
        Image.fromarray(arr[idx].astype(np.uint8), mode="RGB").save(out)
    return _count_pngs(image_dir)


def _parse_fid(stdout: str) -> float | None:
    for line in reversed(stdout.splitlines()):
        match = FID_RE.match(line)
        if match:
            return float(match.group(1))
    return None


def _sample_command(*, checkpoint: Path, step_count: int, num_samples: int, batch: int, seed: int) -> list[str]:
    sampler = "onestep" if step_count <= 1 else "multistep"
    args = [
        sys.executable,
        str(CM_ROOT / "scripts" / "image_sample.py"),
        "--training_mode",
        "consistency_distillation",
        "--sampler",
        sampler,
        "--model_path",
        str(checkpoint),
        "--attention_resolutions",
        "32,16,8",
        "--class_cond",
        "True",
        "--use_scale_shift_norm",
        "True",
        "--dropout",
        "0.0",
        "--image_size",
        "64",
        "--num_channels",
        "192",
        "--num_head_channels",
        "64",
        "--num_res_blocks",
        "3",
        "--num_samples",
        str(num_samples),
        "--batch_size",
        str(batch),
        "--resblock_updown",
        "True",
        "--use_fp16",
        "True",
        "--weight_schedule",
        "uniform",
        "--seed",
        str(seed),
    ]
    if step_count > 1:
        args.extend(["--steps", "40", "--ts", _ts_for_step_count(step_count)])
    return args


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
                    "eval_script": "scripts/baselines/run_cd_imagenet64_eval.py",
                    "notes": row["notes"],
                }
            )


def main() -> None:
    args = parse_args()
    checkpoint = _resolve(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not CM_ROOT.exists():
        raise FileNotFoundError(CM_ROOT)
    if not EDM_ROOT.exists():
        raise FileNotFoundError(EDM_ROOT)

    sample_root = _resolve(args.sample_root)
    eval_root = _resolve(args.eval_root)
    csv_out = _resolve(args.csv_out)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(CM_ROOT) + os.pathsep + str(EDM_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("DNNLIB_CACHE_DIR", str(ROOT / ".torch" / "dnnlib"))
    env.setdefault("OPENAI_LOG_FORMAT", "stdout,log")
    env.setdefault("OMPI_MCA_btl", "^openib")
    env.setdefault("OMPI_MCA_btl_openib_warn_no_device_params_found", "0")

    rows: list[dict[str, Any]] = []
    for step_count in args.steps:
        step_dir = eval_root / f"steps{step_count}"
        log_dir = step_dir / "openai_log"
        image_dir = sample_root / f"steps{step_count}" / "images"
        step_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        env["OPENAI_LOGDIR"] = str(log_dir)
        t0 = time.time()

        sample_npz = _find_sample_npz(log_dir)
        if not args.skip_sample and sample_npz is None:
            _run_and_tee(
                _sample_command(
                    checkpoint=checkpoint,
                    step_count=step_count,
                    num_samples=args.num_samples,
                    batch=args.batch,
                    seed=args.seed + 1000 * step_count,
                ),
                cwd=CM_ROOT,
                env=env,
                log_path=step_dir / "sample.stdout_stderr.txt",
            )
            sample_npz = _find_sample_npz(log_dir)
        if sample_npz is None:
            raise FileNotFoundError(f"No samples_*.npz found in {log_dir}")

        png_count = _convert_npz_to_pngs(sample_npz, image_dir, expected=args.num_samples)
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

        notes = [
            "official OpenAI consistency_models checkpoint",
            f"sampler={'onestep' if step_count <= 1 else 'multistep'}",
            f"ts={_ts_for_step_count(step_count) or 'onestep'}",
            f"num_fid_samples={args.num_samples}",
            "recall pending ImageNet64 reference sample batch",
        ]
        row = {
            "step_count": int(step_count),
            "fid": fid,
            "num_fid_samples": int(args.num_samples),
            "checkpoint": str(checkpoint),
            "method": args.method,
            "sample_npz": str(sample_npz),
            "image_dir": str(image_dir),
            "elapsed_sec": time.time() - t0,
            "notes": "; ".join(notes),
        }
        with (step_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2)
        rows.append(row)
        _write_outputs(rows=rows, eval_root=eval_root, csv_out=csv_out)
        print(f"cd imagenet64 step_count={step_count} fid={fid} elapsed_sec={row['elapsed_sec']:.2f}", flush=True)


if __name__ == "__main__":
    main()
