"""Deterministic transcript statistics with no inference dependencies."""

from collections import Counter
import math
import re
from typing import Dict, Sequence

from .models import EngineTranscript, TranscriptionResult


# Apostrophes and hyphens inside words stay together; punctuation is not a word.
_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
DEFAULT_FILLERS = ("um", "uh", "erm", "er", "hmm", "like", "you know", "i mean")


def _tokens(text: str):
    return _WORD.findall(text.casefold())


def count_words(text: str) -> int:
    return len(_tokens(text))


def words_per_minute(word_count: int, duration_seconds: float) -> float:
    if word_count < 0 or duration_seconds < 0 or not math.isfinite(duration_seconds):
        raise ValueError("Word count and duration must be nonnegative; duration must be finite")
    return round(word_count * 60 / duration_seconds, 1) if duration_seconds else 0.0


def detect_fillers(text: str, fillers: Sequence[str] = DEFAULT_FILLERS) -> Dict[str, int]:
    """Case-insensitive whole-token/phrase counts; longest match wins.

    Only observed fillers are returned. These are lexical counts, not a claim
    that every 'like' or 'you know' is a disfluency in context.
    """
    tokens = _tokens(text)
    phrases = sorted({tuple(_tokens(filler)) for filler in fillers} - {()},
                     key=lambda phrase: (-len(phrase), phrase))
    counts: Counter = Counter()
    index = 0
    while index < len(tokens):
        for phrase in phrases:
            if tuple(tokens[index:index + len(phrase)]) == phrase:
                counts[" ".join(phrase)] += 1
                index += len(phrase)
                break
        else:
            index += 1
    return dict(counts)


def summarize_transcript(transcript: EngineTranscript) -> TranscriptionResult:
    # Empty segments are not recognized speech. Preserve all nonempty timestamps.
    segments = [segment for segment in transcript.segments if segment.text.strip()]
    full_text = " ".join(" ".join(segment.text.split()) for segment in segments)
    duration = 0.0
    covered_end = 0.0
    for segment in segments:
        # Union of intervals prevents overlap from counting speech time twice.
        duration += max(0.0, segment.end - max(covered_end, segment.start))
        covered_end = max(covered_end, segment.end)
    word_count = count_words(full_text)
    return TranscriptionResult(
        duration_seconds=round(duration, 6),
        audio_duration_seconds=transcript.audio_duration_seconds,
        word_count=word_count,
        words_per_minute=words_per_minute(word_count, duration),
        filler_words=detect_fillers(full_text),
        segments=segments,
        full_transcript=full_text,
    )
