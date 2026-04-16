from .baseline import BaselineTrainer
from .map import MapTrainer
from dgtd.train_dgtd import DGTDTrainer


def build_trainer(config, roots, dist_ctx):
    objective = str(config.get("train", {}).get("objective", "flow_matching_velocity"))
    if objective in {"flow_matching_velocity", "velocity_fm"}:
        return BaselineTrainer(config=config, roots=roots, dist_ctx=dist_ctx)
    if objective in {"explicit_map", "map_branch"}:
        return MapTrainer(config=config, roots=roots, dist_ctx=dist_ctx)
    if objective in {"dgtd_map"}:
        return DGTDTrainer(config=config, roots=roots, dist_ctx=dist_ctx)
    raise ValueError(f"Unsupported train.objective: {objective}")


__all__ = ["BaselineTrainer", "MapTrainer", "DGTDTrainer", "build_trainer"]
