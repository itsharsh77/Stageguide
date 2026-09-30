"""Local speech transcription with replaceable inference backends."""

from .backend import BackendMetadata, BackendUnavailableError, SpeechBackend
from .diagnostics import platform_diagnostics
from .factory import BackendConfig, create_backend

__all__ = [
    "BackendConfig", "BackendMetadata", "BackendUnavailableError", "SpeechBackend",
    "create_backend", "platform_diagnostics",
]
