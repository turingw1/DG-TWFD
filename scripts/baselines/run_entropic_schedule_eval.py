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
ENTROPIC_ROOT = ROOT / "refs" / "entropic_time_schedulers" / "EDM"
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

DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "cifar10": {
        "data_code": "CIFAR10",
        "time_path": "refs/entropic_time_schedulers/EDM/Schedules/RE_function_CIFAR10_uncond_vp_80_128_FREQ.pt",
        "method": "Entropic-RE-official",
        "sample_root": "runs/entropic_cifar10_5k/samples",
        "eval_root": "eval/entropic_cifar10_5k",
        "csv_out": "results/baselines/schedule_entropic_cifar10.csv",
        "fid_ref": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz",
        "checkpoint": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl",
        "batch": 256,
    },
    "imagenet64": {
        "data_code": "Imagenet_64_s",
        "time_path": "refs/entropic_time_schedulers/EDM/Schedules/Rescaled_entropic_time_64.pt",
        "method": "Entropic-RE-official",
        "sample_root": "runs/entropic_imagenet64_5k/samples",
        "eval_root": "eval/entropic_imagenet64_5k",
        "csv_out": "results/baselines/schedule_entropic_imagenet64.csv",
        "fid_ref": "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz",
        "checkpoint": "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl",
        "batch": 128,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Entropic Time Scheduler baseline and EDM FID.")
    subparsers = parser.add_subparsers(dest="command")

    main = subparsers.add_parser("run")
    main.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    main.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
    main.add_argument("--num-samples", type=int, default=5000)
    main.add_argument("--batch", type=int, default=None)
    main.add_argument("--fid-batch", type=int, default=512)
    main.add_argument("--solver", choices=["SDDIM", "DDDIM", "Heun"], default="SDDIM")
    main.add_argument("--seed", type=int, default=42)
    main.add_argument("--skip-sample", action="store_true")
    main.add_argument("--skip-fid", action="store_true")
    main.add_argument("--force", action="store_true")
    main.add_argument("--sample-root", default=None)
    main.add_argument("--eval-root", default=None)
    main.add_argument("--csv-out", default=None)
    main.add_argument("--time-path", default=None)
    main.add_argument("--fid-ref", default=None)
    main.add_argument(
        "--stable-report-root",
        default="/temp/Zhengwei/projects/DG-TWFD/critical/analysis/baselines/reports",
    )

    worker = subparsers.add_parser("_generate")
    worker.add_argument("--data-code", required=True)
    worker.add_argument("--time-path", required=True)
    worker.add_argument("--solver", required=True)
    worker.add_argument("--batch", type=int, required=True)
    worker.add_argument("--num-samples", type=int, required=True)
    worker.add_argument("--num-steps", type=int, required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--outdir", required=True)
    worker.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def _dataset_defaults(args: argparse.Namespace) -> dict[str, Any]:
    defaults = dict(DATASET_DEFAULTS[args.dataset])
    defaults["sample_root"] = args.sample_root or defaults["sample_root"]
    defaults["eval_root"] = args.eval_root or defaults["eval_root"]
    defaults["csv_out"] = args.csv_out or defaults["csv_out"]
    defaults["time_path"] = args.time_path or defaults["time_path"]
    defaults["fid_ref"] = args.fid_ref or defaults["fid_ref"]
    defaults["batch"] = args.batch or defaults["batch"]
    return defaults


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
    return sum(1 for item in path.glob("*.png") if item.is_file())


def _parse_fid(stdout: str) -> float | None:
    for line in reversed(stdout.splitlines()):
        match = FID_RE.match(line)
        if match:
            return float(match.group(1))
    return None


def _load_completed_metric(metrics_path: Path, image_dir: Path, *, expected: int) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    try:
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if row.get("fid") is None:
        return None
    if int(row.get("num_fid_samples", 0) or 0) != expected:
        return None
    if _count_pngs(image_dir) < expected:
        return None
    return row


def _write_preview(image_dir: Path, preview_path: Path, *, limit: int = 64) -> None:
    images = sorted(image_dir.glob("*.png"))[:limit]
    if not images:
        return
    thumbs = []
    for path in images:
        image = Image.open(path).convert("RGB")
        image.thumbnail((64, 64))
        thumbs.append(image.copy())
    side = int(np.ceil(np.sqrt(len(thumbs))))
    canvas = Image.new("RGB", (side * 64, side * 64), "white")
    for idx, image in enumerate(thumbs):
        canvas.paste(image, ((idx % side) * 64, (idx // side) * 64))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_path)


def _write_outputs(*, rows: list[dict[str, Any]], eval_root: Path, csv_out: Path, stable_report_root: Path) -> None:
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
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "step": row["step_count"],
                    "fid": "" if row["fid"] is None else f"{float(row['fid']):.6f}",
                    "is": "",
                    "recall": "",
                    "checkpoint": row["checkpoint"],
                    "eval_script": "scripts/baselines/run_entropic_schedule_eval.py",
                    "notes": row["notes"],
                }
            )

    stable_report_root.mkdir(parents=True, exist_ok=True)
    stem = csv_out.stem
    for source in [report_dir / "summary.json", report_dir / "summary.csv", csv_out]:
        target = stable_report_root / f"{stem}_{source.name}"
        if source.resolve() == target.resolve():
            continue
        target.write_bytes(source.read_bytes())


def _run_worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(ENTROPIC_ROOT))

    import torch
    from Generate_image import generate_image

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    checkpoint = torch.load(args.time_path, map_location=torch.device("cpu"))
    time_values = checkpoint["time"]
    time_func = checkpoint["time_func"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    generate_image(
        data=args.data_code,
        time=time_values,
        time_function=time_func,
        solver=args.solver,
        batch_size=int(args.batch),
        num_images=int(args.num_samples),
        num_steps=int(args.num_steps),
        device=args.device,
        save_path=str(outdir),
    )


def _run_main(args: argparse.Namespace) -> None:
    defaults = _dataset_defaults(args)
    if not ENTROPIC_ROOT.exists():
        raise FileNotFoundError(ENTROPIC_ROOT)
    if not EDM_ROOT.exists():
        raise FileNotFoundError(EDM_ROOT)

    time_path = _resolve(defaults["time_path"])
    if not time_path.exists():
        raise FileNotFoundError(time_path)
    sample_root = _resolve(defaults["sample_root"])
    eval_root = _resolve(defaults["eval_root"])
    csv_out = _resolve(defaults["csv_out"])
    stable_report_root = _resolve(args.stable_report_root)

    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ENTROPIC_ROOT) + os.pathsep + str(EDM_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    )
    env.setdefault("DNNLIB_CACHE_DIR", str(ROOT / ".torch" / "dnnlib"))
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("OMPI_MCA_btl", "^openib")
    env.setdefault("OMPI_MCA_btl_openib_warn_no_device_params_found", "0")

    rows: list[dict[str, Any]] = []
    for step_count in args.steps:
        step_dir = eval_root / f"steps{step_count}"
        image_dir = sample_root / f"steps{step_count}"
        metrics_path = step_dir / "metrics.json"
        step_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        completed = None if args.force else _load_completed_metric(
            metrics_path,
            image_dir,
            expected=args.num_samples,
        )
        if completed is not None and not args.skip_fid:
            rows.append(completed)
            _write_outputs(
                rows=rows,
                eval_root=eval_root,
                csv_out=csv_out,
                stable_report_root=stable_report_root,
            )
            print(f"reuse entropic {args.dataset} step_count={step_count} fid={completed['fid']}", flush=True)
            continue

        if not args.skip_sample and _count_pngs(image_dir) < args.num_samples:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_generate",
                "--data-code",
                str(defaults["data_code"]),
                "--time-path",
                str(time_path),
                "--solver",
                args.solver,
                "--batch",
                str(defaults["batch"]),
                "--num-samples",
                str(args.num_samples),
                "--num-steps",
                str(step_count),
                "--seed",
                str(args.seed + 1000 * int(step_count)),
                "--outdir",
                str(image_dir),
                "--device",
                "cuda:0",
            ]
            _run_and_tee(command, cwd=ROOT, env=env, log_path=step_dir / "sample.stdout_stderr.txt")

        png_count = _count_pngs(image_dir)
        if png_count < args.num_samples:
            raise RuntimeError(f"Only found {png_count}/{args.num_samples} images for step_count={step_count}")

        _write_preview(image_dir, step_dir / "preview_first64.png")
        fid = None
        if not args.skip_fid:
            fid_stdout = _run_and_tee(
                [
                    sys.executable,
                    str(EDM_ROOT / "fid.py"),
                    "calc",
                    f"--images={image_dir}",
                    f"--ref={defaults['fid_ref']}",
                    f"--num={args.num_samples}",
                    f"--batch={args.fid_batch}",
                ],
                cwd=EDM_ROOT,
                env=env,
                log_path=step_dir / "fid.stdout_stderr.txt",
            )
            fid = _parse_fid(fid_stdout)

        row = {
            "dataset": args.dataset,
            "step_count": int(step_count),
            "fid": fid,
            "num_fid_samples": int(args.num_samples),
            "checkpoint": str(defaults["checkpoint"]),
            "method": defaults["method"],
            "image_dir": str(image_dir),
            "time_path": str(time_path),
            "solver": args.solver,
            "elapsed_sec": time.time() - t0,
            "notes": "; ".join(
                [
                    "official Entropic Time Scheduler precomputed schedule",
                    f"data_code={defaults['data_code']}",
                    f"solver={args.solver}",
                    f"time_path={time_path}",
                    f"num_fid_samples={args.num_samples}",
                    f"png_count={png_count}",
                    "recall pending reference sample batch",
                ]
            ),
        }
        metrics_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        _write_outputs(
            rows=rows,
            eval_root=eval_root,
            csv_out=csv_out,
            stable_report_root=stable_report_root,
        )
        print(f"entropic {args.dataset} step_count={step_count} fid={fid} elapsed_sec={row['elapsed_sec']:.2f}", flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "_generate":
        _run_worker(args)
    elif args.command == "run":
        _run_main(args)
    else:
        raise SystemExit("Choose a command: run or _generate")


if __name__ == "__main__":
    main()
