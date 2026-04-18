from __future__ import annotations

from torch import nn

def ensure_flow_matching_image_models_on_path() -> None:
    raise ModuleNotFoundError(
        "The official flow_matching image UNet is not vendored in this branch. "
        "Use the tracked local explicit-map backbone instead of official_map_unet."
    )


class OfficialVelocityUNet(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        ensure_flow_matching_image_models_on_path()
        from models.unet import UNetModel

        self.model = UNetModel(**kwargs)

    def forward(self, x, t, extra: dict | None = None):
        return self.model(x, t, extra or {})
