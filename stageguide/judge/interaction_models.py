"""In-memory single-turn records; Milestone 5A models remain unchanged."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .backend import JudgeBackendMetadata
from .models import JudgeQuestion


@dataclass(frozen=True)
class JudgeResponse:
    question: str
    mode: str
    grounded: bool
    source_question_id: str
    reason: str
    used_fallback: bool
    fallback_reason: Optional[str]
    source_question: JudgeQuestion
    backend_metadata: JudgeBackendMetadata

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.mode == "follow_up":
            result["follow_up_question"] = result.pop("question")
        return result


@dataclass(frozen=True)
class JudgeInteraction:
    original_question: JudgeQuestion
    refined_question: JudgeResponse
    presenter_answer: str
    follow_up: Optional[JudgeResponse]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_question": self.original_question.to_dict(),
            "refined_question": self.refined_question.to_dict(),
            "presenter_answer": self.presenter_answer,
            "follow_up": self.follow_up.to_dict() if self.follow_up is not None else None,
        }
