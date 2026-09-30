"""Presentation-oriented argument checks; no conversational Judge Mode."""

from .analyzer import analyze_presentation, analyze_slide
from .models import ArgumentFinding, FindingEvidence, FindingType, Severity, SlideArgumentResult

__all__ = [
    "analyze_slide", "analyze_presentation", "ArgumentFinding", "FindingEvidence",
    "FindingType", "Severity", "SlideArgumentResult",
]
