#!/usr/bin/env python3
"""Build per-sample qualitative panels.

Each output panel corresponds to one seed/class. Rows are model variants and
columns are the 1/2/4/8 display steps. Samples are nearest-neighbor upscaled
before grid composition so the embedded raster has enough pixels for paper
layout. The PDFs intentionally contain images only and use lossless
FlateDecode embedding.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
QUAL_DIR = ROOT / "docs" / "experiments" / "DG_TWFD_v3" / "figures" / "qualitative"
OUTDIR = QUAL_DIR / "paper_panels_20260502"
STEPS = [1, 2, 4, 8]
CELL_GAP = 0
ROW_GAP = 0
BACKGROUND = (255, 255, 255)
SAMPLE_PIXEL_SIZE = 256
TARGET_DPI = 300
CIFAR_SAMPLE_ROOT = QUAL_DIR / "class_locked_samples" / "cifar10_20260502_paper"
IMAGENET_SAMPLE_ROOT = QUAL_DIR / "class_locked_samples" / "imagenet64_20260502_paper"


def _save_lossless_pdf(image: Image.Image, path: Path) -> None:
    image = image.convert("RGB")
    width, height = image.size
    compressed = zlib.compress(image.tobytes(), level=9)
    content = f"q\n{width} 0 0 {height} 0 0 cm\n/Im0 Do\nQ\n".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("ascii"),
        (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(compressed)} >>\nstream\n"
        ).encode("ascii")
        + compressed
        + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
        + content
        + b"endstream",
    ]

    data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode("ascii"))
        data.extend(obj)
        data.extend(b"\nendobj\n")
    xref_offset = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    data.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(data))


def _load_sample(pattern: str, step: int, sample_id: int) -> Image.Image:
    path = ROOT / pattern.format(step=step) / f"{sample_id:06d}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    return image.resize((SAMPLE_PIXEL_SIZE, SAMPLE_PIXEL_SIZE), Image.Resampling.NEAREST)


def _compose_grid(rows: list[dict], sample_id: int) -> Image.Image:
    cells = [
        [_load_sample(row["pattern"], step, sample_id) for step in STEPS]
        for row in rows
    ]
    col_widths = [max(row[col].width for row in cells) for col in range(len(STEPS))]
    row_heights = [max(cell.height for cell in row) for row in cells]
    width = sum(col_widths) + CELL_GAP * (len(STEPS) - 1)
    height = sum(row_heights) + ROW_GAP * (len(rows) - 1)
    panel = Image.new("RGB", (width, height), BACKGROUND)

    y = 0
    for row_idx, row in enumerate(cells):
        x = 0
        for col_idx, cell in enumerate(row):
            panel.paste(cell, (x, y))
            x += col_widths[col_idx] + CELL_GAP
        y += row_heights[row_idx] + ROW_GAP
    return panel


def _write_panel(panel: Image.Image, dataset: str, stem: str) -> dict:
    dataset_dir = OUTDIR / dataset
    paths = {
        "png": dataset_dir / "png" / f"{stem}.png",
        "pdf": dataset_dir / "pdf" / f"{stem}.pdf",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    panel.save(paths["png"], compress_level=6)
    _save_lossless_pdf(panel, paths["pdf"])

    return {
        name: str(path.relative_to(ROOT))
        for name, path in paths.items()
    } | {
        "pixel_size": [panel.width, panel.height],
        "recommended_max_width_in_at_300dpi": round(panel.width / TARGET_DPI, 3),
    }


def _cifar_rows() -> list[dict]:
    base = "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/cifar10_20260502_paper"
    return [
        {
            "label": "EDM CIFAR-10 cond-VP 32/48/64/128",
            "pattern": f"{base}/edm_cifar10_cond_vp_32_48_64_128/steps{{step}}",
        },
        {
            "label": "DG-TWFD identity",
            "pattern": f"{base}/dg_twfd_identity_same_checkpoint/steps{{step}}",
        },
        {
            "label": "CTM official CIFAR-10 conditional",
            "pattern": f"{base}/ctm_official_cond/steps{{step}}",
        },
        {
            "label": "CTM no-GAN DSM 10k",
            "pattern": f"{base}/ctm_nogan_dsm_10k/steps{{step}}",
        },
        {
            "label": "CD-LPIPS CIFAR-10 JAX",
            "pattern": f"{base}/cd_lpips_cifar10_jax/steps{{step}}",
            "lock": "seed-only; released JAX checkpoint is not class-conditional",
        },
        {
            "label": "CD-L2 CIFAR-10 JAX",
            "pattern": f"{base}/cd_l2_cifar10_jax/steps{{step}}",
            "lock": "seed-only; released JAX checkpoint is not class-conditional",
        },
        {
            "label": "CT-LPIPS CIFAR-10 JAX",
            "pattern": f"{base}/ct_lpips_cifar10_jax/steps{{step}}",
            "lock": "seed-only; released JAX checkpoint is not class-conditional",
        },
    ]


def _imagenet_rows() -> list[dict]:
    base = "docs/experiments/DG_TWFD_v3/figures/qualitative/class_locked_samples/imagenet64_20260502_paper"
    return [
        {
            "label": "EDM ImageNet64 cond-ADM 32/48/64/128",
            "pattern": f"{base}/edm_imagenet64_cond_adm_32_48_64_128/steps{{step}}",
        },
        {
            "label": "EDM ImageNet64 identity proxy 8/16/24/30",
            "pattern": f"{base}/edm_imagenet64_identity_8_16_24_30/steps{{step}}",
        },
        {
            "label": "CD-LPIPS ImageNet64",
            "pattern": f"{base}/cd_lpips_imagenet64/steps{{step}}",
        },
        {
            "label": "CD-L2 ImageNet64",
            "pattern": f"{base}/cd_l2_imagenet64/steps{{step}}",
        },
        {
            "label": "CT ImageNet64",
            "pattern": f"{base}/ct_imagenet64/steps{{step}}",
        },
        {
            "label": "CTM ImageNet64 official",
            "pattern": f"{base}/ctm_imagenet64_official/steps{{step}}",
        },
    ]


def main() -> None:
    cifar_rows = _cifar_rows()
    imagenet_rows = _imagenet_rows()
    cifar_columns = json.loads((CIFAR_SAMPLE_ROOT / "manifest.json").read_text(encoding="utf-8"))["columns"]
    imagenet_columns = json.loads((IMAGENET_SAMPLE_ROOT / "manifest.json").read_text(encoding="utf-8"))["columns"]

    panels: dict[str, list[dict]] = {"cifar10": [], "imagenet64": []}
    for item in cifar_columns:
        sample_id = int(item["latent_seed"])
        class_id = int(item["class_id"])
        class_name = str(item["class_name"])
        panel = _compose_grid(cifar_rows, sample_id)
        stem = f"cifar10_seed{sample_id:04d}_class{class_id:02d}_{class_name}"
        panels["cifar10"].append(
            {
                "stem": stem,
                "sample_id": sample_id,
                "class_id": class_id,
                "class_name": class_name,
                "outputs": _write_panel(panel, "cifar10", stem),
            }
        )

    for item in imagenet_columns:
        sample_id = int(item["index"])
        class_id = int(item["class_id"])
        panel = _compose_grid(imagenet_rows, sample_id)
        stem = f"imagenet64_idx{sample_id:02d}_class{class_id:04d}_seed{int(item['seed']):02d}"
        panels["imagenet64"].append(
            {
                "stem": stem,
                "sample_id": sample_id,
                "class_id": class_id,
                "seed": int(item["seed"]),
                "outputs": _write_panel(panel, "imagenet64", stem),
            }
        )

    manifest = {
        "note": (
            "Per-sample image-only panels. Each file is one seed/class. "
            "Rows are models and columns are display steps 1/2/4/8. "
            "Samples are nearest-neighbor upscaled to 256x256 before tiling. "
            "No internal whitespace or gaps are inserted."
        ),
        "columns": [f"NFE {step}" for step in STEPS],
        "sample_pixel_size": SAMPLE_PIXEL_SIZE,
        "cell_gap": CELL_GAP,
        "row_gap": ROW_GAP,
        "target_dpi": TARGET_DPI,
        "pdf_encoding": "lossless FlateDecode RGB image XObject",
        "datasets": {
            "cifar10": {
                "rows": [row["label"] for row in cifar_rows],
                "row_notes": {
                    row["label"]: row.get("lock", "class-locked")
                    for row in cifar_rows
                },
                "panels": panels["cifar10"],
            },
            "imagenet64": {
                "rows": [row["label"] for row in imagenet_rows],
                "row_notes": {
                    row["label"]: row.get("lock", "class-locked")
                    for row in imagenet_rows
                },
                "panels": panels["imagenet64"],
            },
        },
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    readme = (
        "# Paper-Ready Per-Sample Qualitative Panels\n\n"
        "Each file contains one seed/class. Rows are model variants and columns are "
        "`1 / 2 / 4 / 8` display steps. The panels are image-only; row/column labels "
        "are recorded in `manifest.json` for LaTeX or vector-editor labeling.\n\n"
        "Samples are nearest-neighbor upscaled to 256 x 256 before tiling. No gaps "
        "or borders are inserted. The PDFs use lossless `FlateDecode`; text labels "
        "should be added in LaTeX.\n"
    )
    (OUTDIR / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
