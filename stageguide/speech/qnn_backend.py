"""Planned Windows ARM64 Qualcomm QNN integration point. No inference exists."""

from pathlib import Path

from .backend import BackendMetadata, BackendUnavailableError
from .models import EngineTranscript


QNN_UNAVAILABLE = (
    "qnn_whisper is planned, not implemented or tested on any platform. "
    "The target is Windows ARM64 on Snapdragon with Qualcomm QNN/NPU. "
    "No Qualcomm inference was run; select cpu_whisper for the development backend."
)


class QualcommQNNBackend:
    """Fail-closed placeholder; never silently falls back to CPU.

    A real adapter must implement metadata, audio decoding, model execution and
    original-audio timestamps before the factory/diagnostics enable this backend.
    """

    def __init__(self) -> None:
        raise BackendUnavailableError(QNN_UNAVAILABLE)

    @property
    def metadata(self) -> BackendMetadata:
        raise BackendUnavailableError(QNN_UNAVAILABLE)

    def transcribe(self, path: Path) -> EngineTranscript:
        raise BackendUnavailableError(QNN_UNAVAILABLE)
