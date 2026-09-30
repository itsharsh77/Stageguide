"""Explicit backend selection, independent of application business logic."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .backend import BackendUnavailableError, SpeechBackend


@dataclass(frozen=True)
class BackendConfig:
    backend: str = "cpu_whisper"
    model_path: Union[str, Path] = Path("models/whisper-tiny.en")
    language: str = "en"
    model_name: Optional[str] = None


def create_backend(config: Optional[BackendConfig] = None) -> SpeechBackend:
    """Construct the explicitly selected backend, without loading any model.

    Selection never guesses from the OS and never silently falls back.
    """
    config = config if config is not None else BackendConfig()
    if config.backend == "cpu_whisper":
        from .backend import WhisperCPUBackend

        return WhisperCPUBackend(config.model_path, config.language, model_name=config.model_name)
    if config.backend == "qnn_whisper":
        from .qnn_backend import QualcommQNNBackend

        return QualcommQNNBackend()
    raise BackendUnavailableError(
        f"Unknown speech backend '{config.backend}'; choose cpu_whisper "
        "(implemented) or qnn_whisper (planned, unavailable)"
    )
