"""Refine and follow up once, with deterministic questions as the fail-safe."""

from copy import deepcopy
from typing import Optional

from .backend import DisabledJudgeBackend, JudgeBackend
from .grounding import GroundingError, parse_and_validate
from .interaction_models import JudgeInteraction, JudgeResponse
from .models import JudgeQuestion
from .prompts import PromptBudgetError, build_request


class LocalJudge:
    def __init__(self, backend: Optional[JudgeBackend] = None) -> None:
        self.backend = backend if backend is not None else DisabledJudgeBackend()

    def _generate(self, question: JudgeQuestion, answer: Optional[str]) -> JudgeResponse:
        source = deepcopy(question)
        mode = "refine" if answer is None else "follow_up"
        fallback_reason = None
        text = source.question
        try:
            metadata = self.backend.metadata
            if not metadata.available:
                fallback_reason = f"backend_unavailable:{metadata.status}"
            else:
                request = build_request(source, answer)
                raw = self.backend.generate(request)
                text = parse_and_validate(raw, source, answer)
        except GroundingError as exc:
            fallback_reason = f"grounding_rejected:{exc}"
        except PromptBudgetError:
            fallback_reason = "invalid_or_oversized_context"
        except Exception:
            # Vendor exceptions never make deterministic questions unavailable.
            # Do not leak raw model output or presenter data through errors.
            fallback_reason = "local_inference_failed"
        if fallback_reason is not None:
            text = source.question
        try:
            metadata = self.backend.metadata
        except Exception:
            from .backend import JudgeBackendMetadata

            metadata = JudgeBackendMetadata(
                "unknown", "unknown", "unavailable", "unavailable", "none", False, False, "metadata_failed",
            )
        return JudgeResponse(
            question=text, mode=mode, grounded=True, source_question_id=source.question_id,
            reason=source.reason, used_fallback=fallback_reason is not None,
            fallback_reason=fallback_reason, source_question=source, backend_metadata=metadata,
        )

    def refine(self, question: JudgeQuestion) -> JudgeResponse:
        return self._generate(question, None)

    def follow_up(self, question: JudgeQuestion, presenter_answer: str) -> Optional[JudgeResponse]:
        if not isinstance(presenter_answer, str):
            raise TypeError("presenter_answer must be a string")
        if not presenter_answer.strip():
            return None
        return self._generate(question, presenter_answer)

    def interact(self, question: JudgeQuestion, presenter_answer: str) -> JudgeInteraction:
        """One initial question, one answer and at most one follow-up. No loop."""
        if not isinstance(presenter_answer, str):
            raise TypeError("presenter_answer must be a string")
        source = deepcopy(question)
        return JudgeInteraction(
            original_question=source, refined_question=self.refine(source),
            presenter_answer=presenter_answer, follow_up=self.follow_up(source, presenter_answer),
        )
