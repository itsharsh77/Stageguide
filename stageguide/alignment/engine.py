"""Deterministic slide coverage using neutral presentation/transcript data."""

from dataclasses import dataclass
import re
from typing import FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

from stageguide.presentation.models import PresentationPage
from stageguide.speech.models import EngineTranscript, TranscriptionResult, TranscriptSegment

from .models import ItemAlignment, SlideAlignmentResult
from .normalization import NEGATIONS, content_tokens, literal_values, numeric_mentions


TranscriptInput = Union[str, TranscriptionResult, EngineTranscript, TranscriptSegment, Sequence[TranscriptSegment]]
EXPLAINED_THRESHOLD = 0.8
PARTIAL_THRESHOLD = 0.4
_CREDIT = {"explained": 1.0, "partial": 0.5, "missing": 0.0}


@dataclass(frozen=True)
class _Item:
    text: str
    kind: str
    context: Optional[str]
    words: FrozenSet[str]
    numbers: FrozenSet[Tuple[str, str]]
    weight: int


def _sentences(text: str) -> List[str]:
    # Decimal points remain inside numbers. Newlines, semicolons and bullets
    # retain slide boundaries within a page; abbreviations are not NLP-parsed.
    parts = re.split(r"(?<=[.!?])\s+|[\n\r;•]+", text)
    cleaned = [re.sub(r"^[-*]\s+", "", " ".join(part.split())) for part in parts]
    return [part for part in cleaned if part]


def _transcript_text(transcript: TranscriptInput) -> str:
    if isinstance(transcript, str):
        return transcript
    if isinstance(transcript, TranscriptionResult):
        return transcript.full_transcript
    if isinstance(transcript, EngineTranscript):
        return " ".join(segment.text for segment in transcript.segments)
    if isinstance(transcript, TranscriptSegment):
        return transcript.text
    if isinstance(transcript, Sequence) and all(isinstance(item, TranscriptSegment) for item in transcript):
        return " ".join(segment.text for segment in transcript)
    raise TypeError("Transcript must be text, a speech result, or transcript segments")


def _slide_items(slide: PresentationPage) -> List[_Item]:
    items = []
    seen = set()
    sources = ([('title', slide.title)] if slide.title and slide.title.strip() else [])
    sources += [("body", line) for line in _sentences(slide.body_text)]
    source_text = "\n".join(text for _, text in sources)
    source_mentions = numeric_mentions(source_text)
    # The ingestion parser includes the '5' in '5 million' and the '30' in
    # '30%' in numbers. Do not create duplicate or weakened numeric claims.
    represented_values = set(literal_values(source_text)) | {mention.value for mention in source_mentions}
    represented_claims = {mention.key for mention in source_mentions}

    def add(item: _Item) -> None:
        # Text duplicates across title/body count once. The same numeric value
        # in different contexts remains a separate claim.
        group = "text" if item.kind in {"title", "body"} else "numeric"
        key = group, item.words, item.numbers
        if key not in seen:
            seen.add(key)
            items.append(item)

    for kind, text in sources:
        text = " ".join(text.split())
        words = content_tokens(text)
        mentions = numeric_mentions(text)
        if words:
            add(_Item(text, kind, None, words, frozenset(mention.key for mention in mentions), 1))
        for mention in mentions:
            add(_Item(mention.text, mention.kind, text, words, frozenset({mention.key}), 2))

    # Respect structured numeric fields even when a caller supplies values
    # absent from the extracted text. No context is invented for these values.
    for annotation in slide.percentages:
        for mention in numeric_mentions(annotation):
            if mention.kind != "number" and mention.key not in represented_claims:
                add(_Item(mention.text, mention.kind, None, frozenset(), frozenset({mention.key}), 2))
            represented_claims.add(mention.key)
            represented_values.add(mention.value)
    for annotation in slide.numbers:
        for mention in numeric_mentions(annotation):
            if mention.value not in represented_values:
                add(_Item(mention.text, mention.kind, None, frozenset(), frozenset({mention.key}), 2))
                represented_values.add(mention.value)
    return items


def _align(slide: PresentationPage, text: str) -> SlideAlignmentResult:
    candidates = [(sentence, content_tokens(sentence), frozenset(m.key for m in numeric_mentions(sentence)))
                  for sentence in _sentences(text)]
    aligned = []
    for item in _slide_items(slide):
        best_status, best_similarity, best_text = "missing", 0.0, None
        for sentence, words, numbers in candidates:
            # A simple polarity guard prevents obvious 'does not reduce' matches.
            # It is deliberately not semantic negation or argument analysis.
            if bool(item.words & NEGATIONS) != bool(words & NEGATIONS):
                continue
            similarity = len(item.words & words) / len(item.words) if item.words else 1.0
            numeric_match = item.numbers <= numbers
            if item.weight == 2 and not numeric_match:
                continue
            if similarity >= EXPLAINED_THRESHOLD and numeric_match:
                status = "explained"
            elif similarity >= PARTIAL_THRESHOLD:
                status = "partial"
            else:
                status = "missing"
            if (_CREDIT[status], similarity) > (_CREDIT[best_status], best_similarity):
                best_status, best_similarity = status, similarity
                best_text = sentence if status != "missing" else None
        aligned.append(ItemAlignment(
            text=item.text, kind=item.kind, context=item.context, status=best_status,
            similarity=round(best_similarity, 4), weight=item.weight, matched_text=best_text,
        ))
    total_weight = sum(item.weight for item in aligned)
    score = sum(item.weight * _CREDIT[item.status] for item in aligned) / total_weight if total_weight else 0.0
    return SlideAlignmentResult(
        slide_number=slide.page_number, title=slide.title, coverage_score=round(score, 4),
        explained_items=[item.text for item in aligned if item.status == "explained"],
        partially_explained_items=[item.text for item in aligned if item.status == "partial"],
        missing_items=[item.text for item in aligned if item.status == "missing"], items=aligned,
    )


def align_slide(slide: PresentationPage, transcript: TranscriptInput) -> SlideAlignmentResult:
    """Align one slide to caller-selected text; no parsing or inference is run."""
    return _align(slide, _transcript_text(transcript))


def align_presentation(
    slides: Sequence[PresentationPage], transcript: TranscriptInput = "", *,
    transcripts_by_slide: Optional[Mapping[int, TranscriptInput]] = None,
) -> List[SlideAlignmentResult]:
    """Compare slides to shared text, or an explicit page-number assignment.

    Missing assignments use empty text. Unknown keys/duplicate slide numbers
    are rejected. Shared text is reused for each slide, not automatically timed.
    """
    numbers = [slide.page_number for slide in slides]
    if len(set(numbers)) != len(numbers):
        raise ValueError("Slide numbers must be unique within an alignment call")
    if transcripts_by_slide is not None:
        if transcript != "":
            raise ValueError("Supply either a shared transcript or transcripts_by_slide, not both")
        if set(transcripts_by_slide) - set(numbers):
            raise ValueError("Transcript assignments contain unknown slide numbers")
        return [align_slide(slide, transcripts_by_slide.get(slide.page_number, "")) for slide in slides]
    text = _transcript_text(transcript)
    return [_align(slide, text) for slide in slides]
