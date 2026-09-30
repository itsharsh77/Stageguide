"""Questions retain their deterministic source findings for later consumers."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List

from stageguide.arguments.models import ArgumentFinding, FindingType, Severity


class QuestionCategory(str, Enum):
    EVIDENCE = "evidence"
    VALIDATION = "validation"
    CLARIFICATION = "clarification"
    MEASUREMENT = "measurement"


@dataclass(frozen=True)
class FindingReference:
    slide_number: int
    finding_id: str
    finding: ArgumentFinding


@dataclass(frozen=True)
class JudgeQuestion:
    slide_number: int
    question_id: str
    priority: Severity
    category: QuestionCategory
    source_finding_type: FindingType
    claim: str
    question: str
    reason: str
    evidence_reference: FindingReference
    additional_evidence_references: List[FindingReference]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
