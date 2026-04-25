#!/usr/bin/env python
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main() -> int:
    proj = Path(os.environ.get("PROJ", Path(__file__).resolve().parents[2]))
    edm = proj / "refs" / "edm"

    print("PROJ:", proj)
    print("branch:", git(["rev-parse", "--abbrev-ref", "HEAD"], proj))
    print("root_commit:", git(["rev-parse", "HEAD"], proj))
    print("submodules:")
    try:
        print(git(["submodule", "status"], proj))
    except subprocess.CalledProcessError as exc:
        print("submodule_status_error:", exc)

    print("edm_exists:", edm.exists())
    if edm.exists():
        print("edm_commit:", git(["rev-parse", "HEAD"], edm))

    print("python:", sys.executable)
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))
    print("DATA_ROOT:", os.environ.get("DATA_ROOT", ""))
    print("RUNS_ROOT:", os.environ.get("RUNS_ROOT", ""))
    print("EVAL_ROOT:", os.environ.get("EVAL_ROOT", ""))
    print("DG_TWFD_BACKUP_ROOT:", os.environ.get("DG_TWFD_BACKUP_ROOT", ""))
    print("DNNLIB_CACHE_DIR:", os.environ.get("DNNLIB_CACHE_DIR", ""))
    print("TMPDIR:", os.environ.get("TMPDIR", ""))

    import torch

    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count:", torch.cuda.device_count())
        print("device0:", torch.cuda.get_device_name(0))

    sys.path.insert(0, str(proj / "src"))
    if edm.exists():
        sys.path.insert(0, str(edm))

    from dgfm.config import load_experiment_config
    import dgfm  # noqa: F401
    import dgtd  # noqa: F401

    cfg = load_experiment_config("configs/experiment/dgtd_cifar10_v3_smoke.yaml")
    print("config_name:", cfg["experiment"]["name"])
    print("config_data_root:", cfg["dataset"]["data_root"])
    print("config_traj_root:", cfg["target"]["shard_root"])
    print("root_imports_ok")

    if edm.exists():
        import dnnlib  # noqa: F401

        print("edm_imports_ok")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
