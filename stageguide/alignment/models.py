"""Deterministic coverage results, independent of parsing and speech engines."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ItemAlignment:
    text: str
    kind: str
    context: Optional[str]
    status: str
    similarity: float
    weight: int
    matched_text: Optional[str]


@dataclass
class SlideAlignmentResult:
    slide_number: int
    title: Optional[str]
    coverage_score: float
    explained_items: List[str]
    partially_explained_items: List[str]
    missing_items: List[str]
    items: List[ItemAlignment]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
