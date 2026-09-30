"""Backend-neutral speech structures; timestamps are seconds in the input audio."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List


class SpeechError(ValueError):
    """Audio, model configuration, or local transcription could not be processed."""


class UnsupportedAudioError(SpeechError):
    """The audio filename has an unsupported extension."""


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise SpeechError("Segment timestamps must be finite")
        if self.start < 0 or self.end < self.start:
            raise SpeechError("Segment timestamps must satisfy 0 <= start <= end")
        if not isinstance(self.text, str):
            raise SpeechError("Segment text must be a string")


@dataclass
class EngineTranscript:
    """Contract returned by an engine, independent of its model/library types."""

    audio_duration_seconds: float
    segments: List[TranscriptSegment]

    def __post_init__(self) -> None:
        if not math.isfinite(self.audio_duration_seconds) or self.audio_duration_seconds < 0:
            raise SpeechError("Audio duration must be finite and nonnegative")
        previous_start = 0.0
        for segment in self.segments:
            if segment.start < previous_start or segment.end > self.audio_duration_seconds + 0.001:
                raise SpeechError("Segments must be chronological and within the audio duration")
            previous_start = segment.start


@dataclass
class TranscriptionResult:
    duration_seconds: float
    audio_duration_seconds: float
    word_count: int
    words_per_minute: float
    filler_words: Dict[str, int]
    segments: List[TranscriptSegment]
    full_transcript: str

    def to_dict(self) -> Dict[str, Any]:
        """Return an independent dictionary suitable for JSON serialization."""
        return asdict(self)
