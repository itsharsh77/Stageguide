"""Phrase existing findings; do not detect claims or assess their validity."""

import re
from typing import Tuple

from stageguide.arguments.models import ArgumentFinding, FindingType

from .models import QuestionCategory


def render_question(finding: ArgumentFinding) -> Tuple[str, QuestionCategory]:
    """Use literal claim text, with a few deliberately narrow grammar rewrites."""
    claim = finding.claim.strip().rstrip(".!?")
    kind = finding.type
    if kind == FindingType.UNSUPPORTED_NUMERIC_CLAIM:
        reduction = re.fullmatch(
            r"([\w-]+) reduces ([A-Za-z -]+) by (\d+(?:\.\d+)?\s*%)",
            claim, flags=re.IGNORECASE,
        )
        if reduction:
            actor, subject, amount = reduction.groups()
            return (
                f"What evidence supports the claimed {amount} reduction in {subject} by {actor}?",
                QuestionCategory.EVIDENCE,
            )
        return f"What evidence supports the quantified claim “{claim}”?", QuestionCategory.EVIDENCE

    if kind == FindingType.UNEXPLAINED_NUMBER:
        market = re.fullmatch(
            r"(?:Our (?:initial )?)?target market(?: is|:)\s+([^.!?;]+)",
            claim, flags=re.IGNORECASE,
        )
        if market:
            return (
                f"How did you arrive at the estimate of {market.group(1)}?",
                QuestionCategory.VALIDATION,
            )
        return f"How did you derive the figure in “{claim}”?", QuestionCategory.VALIDATION

    if kind == FindingType.MISSING_KEY_CLAIM:
        return f"How would you explain “{claim}” to the audience?", QuestionCategory.CLARIFICATION
    if kind == FindingType.PARTIALLY_EXPLAINED_CLAIM:
        return f"What would you add to fully explain “{claim}”?", QuestionCategory.CLARIFICATION
    if kind == FindingType.EVIDENCE_GAP:
        return f"What evidence or comparison supports “{claim}”?", QuestionCategory.EVIDENCE
    if kind == FindingType.VAGUE_CLAIM:
        return f"What does “{claim}” mean in measurable terms?", QuestionCategory.MEASUREMENT
    if kind == FindingType.ASSUMPTION_TO_VALIDATE:
        payment = re.fullmatch(
            r"(Universities|Users|Customers|Students) will pay ([^.!?;]+)",
            claim, flags=re.IGNORECASE,
        )
        if payment:
            audience, amount = payment.groups()
            return (
                f"What evidence suggests {audience.lower()} would be willing to pay {amount}?",
                QuestionCategory.VALIDATION,
            )
        return f"How would you validate the assumption “{claim}”?", QuestionCategory.VALIDATION
    raise ValueError(f"Unsupported finding type: {kind!r}")
