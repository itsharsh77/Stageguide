"""Common interface and the local development/fallback CPU implementation."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

from .models import EngineTranscript, SpeechError, TranscriptSegment


class BackendUnavailableError(SpeechError):
    """A selected backend is unknown or has no working implementation."""


@dataclass(frozen=True)
class BackendMetadata:
    """Configured execution identity, separate from transcript data.

    Metadata is not a runtime health check or proof that a model has loaded.
    """

    backend_name: str
    model_name: str
    execution_provider: str
    device_type: str
    accelerator: str
    is_hardware_accelerated: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SpeechBackend(Protocol):
    """A future QNN adapter implements this method and returns these data types.

    Engines own decoding, resampling, inference and timestamp generation.
    Segments must use original recording offsets, even when VAD removes silence.
    """

    @property
    def metadata(self) -> BackendMetadata:
        ...

    def transcribe(self, path: Path) -> EngineTranscript:
        ...


class WhisperCPUBackend:
    """Development/fallback backend; always CPU, never QNN, GPU or NPU.

    Load a pre-downloaded CTranslate2 Whisper model only when used.
    model_name is a caller-supplied label, defaulting to the directory name
    without a leading 'whisper-'; it is not inferred from model weights.
    """

    def __init__(self, model_path: Union[str, Path], language: str = "en", *,
                 model_name: Optional[str] = None) -> None:
        self.model_path = Path(model_path)
        self.language = language
        self.model_name = model_name or self.model_path.name.removeprefix("whisper-") or "unknown"
        self._model = None

    @property
    def metadata(self) -> BackendMetadata:
        return BackendMetadata(
            backend_name="WhisperCPUBackend", model_name=self.model_name,
            execution_provider="CPU", device_type="CPU", accelerator="none",
            is_hardware_accelerated=False,
        )

    def _load_model(self):
        if self._model is not None:
            return self._model
        # Requiring local tokenizer files also prevents the library's fallback
        # tokenizer download. No remote model identifier is accepted here.
        required = ("model.bin", "config.json", "tokenizer.json")
        has_vocabulary = any((self.model_path / name).is_file()
                             for name in ("vocabulary.txt", "vocabulary.json"))
        if not has_vocabulary or not self.model_path.is_dir() or any(
            not (self.model_path / name).is_file() for name in required
        ):
            raise SpeechError(
                f"Local model is missing or incomplete: {self.model_path}. "
                "Provide a downloaded faster-whisper model directory (see README)."
            )
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise SpeechError("Speech dependencies are missing; install stageguide[speech]") from exc
        self._model = WhisperModel(
            str(self.model_path.resolve()), device="cpu", compute_type="int8",
            local_files_only=True, cpu_threads=4,
        )
        return self._model

    def transcribe(self, path: Path) -> EngineTranscript:
        try:
            # Validate/decode before loading the expensive model. PyAV handles
            # local WAV, MP3, FLAC, M4A, OGG and AIFF; no subprocess or URL input.
            from faster_whisper.audio import decode_audio

            audio = decode_audio(str(path), sampling_rate=16000)
            duration = len(audio) / 16000
            if duration == 0:
                raise SpeechError("Audio contains no samples")
            model = self._load_model()
            segments, _ = model.transcribe(
                audio, language=self.language, beam_size=5, vad_filter=True,
                condition_on_previous_text=False,
            )
            result = []
            # Inference is lazy; consume the generator inside the error boundary.
            for segment in segments:
                text = " ".join(segment.text.split())
                if text:
                    # Whisper rounds to timestamp tokens; bound the last token
                    # to the actual recording rather than inventing extra audio.
                    start = min(max(float(segment.start), 0.0), duration)
                    end = min(max(float(segment.end), start), duration)
                    result.append(TranscriptSegment(start, end, text))
            return EngineTranscript(duration, result)
        except SpeechError:
            raise
        except ImportError as exc:
            raise SpeechError("Speech dependencies are missing; install stageguide[speech]") from exc
        except Exception as exc:
            raise SpeechError(
                f"Local transcription failed for '{path.name}'; check the audio and local model files"
            ) from exc


# Compatibility for Milestone 2 callers. New application code uses the factory.
FasterWhisperBackend = WhisperCPUBackend
