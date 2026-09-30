"""Plain Python data returned by presentation ingestion."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PresentationPage:
    """One slide or PDF page; page_number is always one-based."""

    page_number: int
    title: Optional[str]
    body_text: str
    numbers: List[str]
    percentages: List[str]
    source_filename: str

    def to_dict(self) -> Dict[str, Any]:
        """Return an independent, JSON-serializable dictionary."""
        return asdict(self)
