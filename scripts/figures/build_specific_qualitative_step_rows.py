#!/usr/bin/env python3
"""Build compact step-row qualitative panels from existing per-sample grids.

The source grids are image-only panels generated earlier with model rows and
step columns. This script crops those existing cells and transposes the layout:
rows become inference steps and columns become methods. It does not regenerate
or alter samples.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dgtwfd")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figs" / "qualitative_main"
STEP_LABELS = ["1-step", "2-step", "4-step", "8-step"]


@dataclass(frozen=True)
class PanelSpec:
    name: str
    source_png: Path
    output_stem: str
    methods: tuple[str, ...]
    caption_note: str
    cell_px: int = 256


PANELS = [
    PanelSpec(
        name="cifar10_truck_seed3018",
        source_png=ROOT
        / "docs"
        / "experiments"
        / "DG_TWFD_v3"
        / "figures"
        / "qualitative"
        / "neurips_sample_bank_20260503"
        / "per_sample"
        / "cifar10"
        / "cifar10_018_class0009_truck_seed3018.png",
        output_stem="cifar10_truck_seed3018_step_rows",
        methods=("DG-TWFD", "Identity", "CTM", "CTM no-GAN", "CD-LPIPS", "CD-L2", "CT-LPIPS"),
        caption_note=(
            "CIFAR-10 class-conditional comparison for truck, seed 3018. "
            "Rows fix the inference budget and columns fix the method. "
            "CIFAR-10 JAX consistency-model baselines are seed-only references "
            "because the released checkpoints are not class-conditional."
        ),
    ),
    PanelSpec(
        name="imagenet64_class0407_seed3109",
        source_png=ROOT
        / "docs"
        / "experiments"
        / "DG_TWFD_v3"
        / "figures"
        / "qualitative"
        / "neurips_sample_bank_20260503"
        / "per_sample"
        / "imagenet64"
        / "imagenet64_009_class0407_id0407_seed3109.png",
        output_stem="imagenet64_class0407_seed3109_step_rows",
        methods=("DG-TWFD", "Identity", "CD-LPIPS", "CD-L2", "CT", "CTM"),
        caption_note=(
            "ImageNet64 class-locked comparison for class 0407, seed 3109. "
            "Rows fix the inference budget and columns fix the method. "
            "DG-TWFD preserves the target vehicle class more consistently under "
            "low-step inference."
        ),
    ),
]


def _method_header(label: str) -> str:
    if label == "CTM no-GAN":
        return "CTM\nno-GAN"
    return label


def _crop_cells(source: Image.Image, *, rows: int, cols: int, cell_px: int) -> list[list[Image.Image]]:
    expected = (cols * cell_px, rows * cell_px)
    if source.size != expected:
        raise ValueError(f"Expected source size {expected}, got {source.size}")
    cells: list[list[Image.Image]] = []
    for row in range(rows):
        row_cells = []
        for col in range(cols):
            box = (col * cell_px, row * cell_px, (col + 1) * cell_px, (row + 1) * cell_px)
            row_cells.append(source.crop(box))
        cells.append(row_cells)
    return cells


def _render_panel(spec: PanelSpec) -> dict[str, object]:
    source = Image.open(spec.source_png).convert("RGB")
    n_methods = len(spec.methods)
    n_steps = len(STEP_LABELS)
    cells = _crop_cells(source, rows=n_methods, cols=n_steps, cell_px=spec.cell_px)

    # Compact full-width layout with square cells and vector text.
    cell_in = 0.68
    gutter = 0.035
    left_label_w = 0.60
    right_margin = 0.06
    top_header_h = 0.34
    bottom_margin = 0.08
    width = left_label_w + n_methods * cell_in + (n_methods - 1) * gutter + right_margin
    height = top_header_h + n_steps * cell_in + (n_steps - 1) * gutter + bottom_margin

    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig.subplots_adjust(0, 0, 1, 1)

    x0 = left_label_w / width
    y_top = 1.0 - top_header_h / height
    cell_w = cell_in / width
    cell_h = cell_in / height
    gutter_x = gutter / width
    gutter_y = gutter / height

    for col, method in enumerate(spec.methods):
        cx = x0 + col * (cell_w + gutter_x) + 0.5 * cell_w
        fig.text(
            cx,
            1.0 - 0.50 * top_header_h / height,
            _method_header(method),
            ha="center",
            va="center",
            fontsize=7.2 if method != "CTM no-GAN" else 6.6,
            fontfamily="serif",
            fontweight="bold" if col == 0 else "normal",
            linespacing=0.92,
        )

    for step_idx, step_label in enumerate(STEP_LABELS):
        y = y_top - (step_idx + 1) * cell_h - step_idx * gutter_y
        fig.text(
            0.50 * left_label_w / width,
            y + 0.5 * cell_h,
            step_label,
            ha="center",
            va="center",
            fontsize=7.0,
            fontfamily="serif",
        )
        for method_idx in range(n_methods):
            x = x0 + method_idx * (cell_w + gutter_x)
            ax = fig.add_axes([x, y, cell_w, cell_h])
            # Transpose original source: original row = method, original col = step.
            ax.imshow(cells[method_idx][step_idx], interpolation="none")
            ax.set_axis_off()
            if method_idx == 0:
                ax.add_patch(
                    Rectangle(
                        (0, 0),
                        spec.cell_px - 1,
                        spec.cell_px - 1,
                        fill=False,
                        edgecolor="#222222",
                        linewidth=1.0,
                    )
                )

    pdf_path = OUT_DIR / f"{spec.output_stem}.pdf"
    png_path = OUT_DIR / f"{spec.output_stem}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    return {
        "name": spec.name,
        "source_png": str(spec.source_png.relative_to(ROOT)),
        "pdf": str(pdf_path.relative_to(ROOT)),
        "png_preview_600dpi": str(png_path.relative_to(ROOT)),
        "row_order": STEP_LABELS,
        "column_order": list(spec.methods),
        "cell_crop_px": spec.cell_px,
        "layout": "transposed step rows; DG-TWFD column highlighted with a thin dark border",
        "caption_note": spec.caption_note,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [_render_panel(spec) for spec in PANELS]
    manifest = {
        "description": "Compact main-text qualitative panels transposed from existing per-sample grids.",
        "sampling": "No new sampling; cells are exact crops from existing image-only per-sample grids.",
        "outputs": outputs,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT_DIR / "caption_notes.txt").write_text(
        "\n\n".join(item["caption_note"] for item in outputs) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
