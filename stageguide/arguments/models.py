"""Traceable argument findings; confidence concerns rule detection, not truth."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class FindingType(str, Enum):
    UNSUPPORTED_NUMERIC_CLAIM = "UNSUPPORTED_NUMERIC_CLAIM"
    UNEXPLAINED_NUMBER = "UNEXPLAINED_NUMBER"
    MISSING_KEY_CLAIM = "MISSING_KEY_CLAIM"
    PARTIALLY_EXPLAINED_CLAIM = "PARTIALLY_EXPLAINED_CLAIM"
    EVIDENCE_GAP = "EVIDENCE_GAP"
    VAGUE_CLAIM = "VAGUE_CLAIM"
    ASSUMPTION_TO_VALIDATE = "ASSUMPTION_TO_VALIDATE"


class Severity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class FindingEvidence:
    slide_text: str
    transcript_match: Optional[str]
    related_transcript: List[str]
    supporting_excerpts: List[str]
    alignment_status: str
    alignment_similarity: float
    numeric_values: List[str]
    unmentioned_values: List[str]
    trigger_phrases: List[str]
    rule_id: str


@dataclass(frozen=True)
class ArgumentFinding:
    type: FindingType
    severity: Severity
    claim: str
    reason: str
    evidence: FindingEvidence
    confidence: str


@dataclass
class SlideArgumentResult:
    slide_number: int
    findings: List[ArgumentFinding]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
