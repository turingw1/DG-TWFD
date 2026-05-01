# Paper-Ready Per-Sample Qualitative Panels

Each file contains one seed/class. Rows are model variants and columns are `1 / 2 / 4 / 8` display steps. The panels are image-only; row/column labels are recorded in `manifest.json` for LaTeX or vector-editor labeling. PDFs are stored in the per-seed PDF folders under each dataset.

Samples are nearest-neighbor upscaled to 256 x 256 before tiling. No gaps or borders are inserted. The PDFs use lossless `FlateDecode`; text labels should be added in LaTeX.
