def build_evaluator(config, checkpoint, eval_root):
    objective = str(config.get("train", {}).get("objective", "flow_matching_velocity"))
    if objective in {"explicit_map", "map_branch", "dgtd_map"}:
        from .map_eval import MapEvaluationRunner

        return MapEvaluationRunner(config=config, checkpoint=checkpoint, eval_root=eval_root)
    from .runner import EvaluationRunner

    return EvaluationRunner(config=config, checkpoint=checkpoint, eval_root=eval_root)


def __getattr__(name):
    if name == "EvaluationRunner":
        from .runner import EvaluationRunner

        return EvaluationRunner
    if name == "MapEvaluationRunner":
        from .map_eval import MapEvaluationRunner

        return MapEvaluationRunner
    if name == "save_multistep_qualitative_panel":
        from .qualitative import save_multistep_qualitative_panel

        return save_multistep_qualitative_panel
    if name == "TimewarpSamplingRunner":
        from .timewarp_sampling import TimewarpSamplingRunner

        return TimewarpSamplingRunner
    if name == "TimewarpSearchRunner":
        from .timewarp_search import TimewarpSearchRunner

        return TimewarpSearchRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EvaluationRunner",
    "MapEvaluationRunner",
    "save_multistep_qualitative_panel",
    "TimewarpSamplingRunner",
    "TimewarpSearchRunner",
    "build_evaluator",
]
