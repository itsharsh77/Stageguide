"""Backend-independent, deterministic slide/transcript alignment."""

from .engine import align_presentation, align_slide
from .engine import _sentences as split_sentences, _transcript_text as transcript_text
from .models import ItemAlignment, SlideAlignmentResult

__all__ = [
    "align_slide", "align_presentation", "ItemAlignment", "SlideAlignmentResult",
    "split_sentences", "transcript_text",
]
