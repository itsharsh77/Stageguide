"""Explicit backend selection; importing this module never loads a model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .backend import DisabledJudgeBackend, JudgeBackend


@dataclass(frozen=True)
class JudgeConfig:
    backend: str = "disabled"
    model: str = "Qwen3-4B-Instruct-2507"
    model_path: Optional[Union[str, Path]] = None
    context_size: int = 4096
    acceleration: str = "auto"


def create_judge_backend(config: Optional[JudgeConfig] = None) -> JudgeBackend:
    config = config if config is not None else JudgeConfig()
    if config.backend == "disabled":
        return DisabledJudgeBackend()
    if config.backend == "local":
        from .local_backend import LocalDevelopmentJudgeBackend

        return LocalDevelopmentJudgeBackend(config.model_path, config.model, config.context_size,
                                            acceleration=config.acceleration)
    if config.backend == "qualcomm":
        from .qualcomm_backend import QualcommJudgeBackend

        return QualcommJudgeBackend(config.model)
    raise ValueError(f"Unknown judge backend {config.backend!r}; choose disabled, local or qualcomm")
