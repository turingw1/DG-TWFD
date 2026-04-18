from __future__ import annotations

from pathlib import Path
import sys

from torch import nn


def _flow_matching_image_roots() -> list[Path]:
    root = Path(__file__).resolve().parents[3]
    return [
        root / "flow_matching" / "examples" / "image",
        root / "public_repos" / "DGTW-code-base" / "flow_matching" / "examples" / "image",
    ]


def ensure_flow_matching_image_models_on_path() -> None:
    for image_root in _flow_matching_image_roots():
        if not image_root.exists():
            continue
        if str(image_root) not in sys.path:
            sys.path.insert(0, str(image_root))
        return
    candidates = "\n".join(str(path) for path in _flow_matching_image_roots())
    raise ModuleNotFoundError(
        "Could not locate flow_matching example image models. "
        f"Tried:\n{candidates}"
    )


class OfficialVelocityUNet(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        ensure_flow_matching_image_models_on_path()
        from models.unet import UNetModel

        self.model = UNetModel(**kwargs)

    def forward(self, x, t, extra: dict | None = None):
        return self.model(x, t, extra or {})
