"""Deterministic planning from findings only; no analysis or inference calls."""

import re
import unicodedata
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple

from stageguide.arguments.models import ArgumentFinding, FindingType, Severity, SlideArgumentResult

from .models import FindingReference, JudgeQuestion
from .templates import render_question


_SEVERITY_ORDER = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
_TYPE_ORDER = {kind: index for index, kind in enumerate((
    FindingType.UNSUPPORTED_NUMERIC_CLAIM,
    FindingType.UNEXPLAINED_NUMBER,
    FindingType.MISSING_KEY_CLAIM,
    FindingType.EVIDENCE_GAP,
    FindingType.ASSUMPTION_TO_VALIDATE,
    FindingType.PARTIALLY_EXPLAINED_CLAIM,
    FindingType.VAGUE_CLAIM,
))}


def _claim_key(claim: str) -> Tuple[str, ...]:
    """Ignore typography, preserving numbers, symbols, negations and word order.

    Deliberately avoid fuzzy similarity: nearby wording can change a claim's
    meaning. Currency, percentage, decimal, sign and comparison symbols matter.
    """
    text = unicodedata.normalize("NFKC", claim).casefold()
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "−": "-"}))
    return tuple(re.findall(r"\d+(?:[.,]\d+)*|[^\W_]+(?:'[^\W_]+)*|[^\w\s.,!?;:\"“”'()]", text))


def plan_questions(results: Sequence[SlideArgumentResult]) -> List[JudgeQuestion]:
    """Rank by severity, type, then supplied presentation/finding order.

    Each slide must occur once. IDs refer to original, one-based finding
    positions (not ranking); they are stable for identical input and top-N
    selection. Deduplication is scoped to one slide and retains all provenance.
    """
    candidates = []
    seen_slides = set()
    for slide_index, result in enumerate(results):
        number = result.slide_number
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError("slide_number must be a positive integer")
        if number in seen_slides:
            raise ValueError(f"Duplicate slide_number: {number}; supply one result per slide")
        seen_slides.add(number)
        for finding_index, finding in enumerate(result.findings, start=1):
            if not finding.claim.strip() or not any(char.isalnum() for char in finding.claim):
                raise ValueError("A question requires a non-empty claim")
            if finding.type not in _TYPE_ORDER:
                raise ValueError(f"Unsupported finding type: {finding.type!r}")
            if finding.severity not in _SEVERITY_ORDER:
                raise ValueError(f"Unsupported severity: {finding.severity!r}")
            reference = FindingReference(number, f"slide{number}_f{finding_index}", deepcopy(finding))
            rank = (_SEVERITY_ORDER[finding.severity], _TYPE_ORDER[finding.type], slide_index, finding_index)
            candidates.append((rank, finding_index, reference))
    candidates.sort(key=lambda candidate: candidate[0])

    groups: Dict[Tuple[int, Tuple[str, ...]], List[Tuple[int, FindingReference]]] = {}
    for _, index, reference in candidates:
        key = (reference.slide_number, _claim_key(reference.finding.claim))
        groups.setdefault(key, []).append((index, reference))

    questions = []
    for group in groups.values():
        index, reference = group[0]
        finding = reference.finding
        question, category = render_question(finding)
        questions.append(JudgeQuestion(
            slide_number=reference.slide_number,
            question_id=f"slide{reference.slide_number}_q{index}",
            priority=finding.severity,
            category=category,
            source_finding_type=finding.type,
            claim=finding.claim,
            question=question,
            reason=finding.reason,
            evidence_reference=reference,
            additional_evidence_references=[other for _, other in group[1:]],
        ))
    return questions


def plan_slide_questions(slide_number: int, findings: Sequence[ArgumentFinding]) -> List[JudgeQuestion]:
    """Convenience entry point when the caller already has one slide's findings."""
    return plan_questions([SlideArgumentResult(slide_number, list(findings))])


def get_top_questions(results: Sequence[SlideArgumentResult], limit: int = 5) -> List[JudgeQuestion]:
    """Select after deduplication, without renumbering question or finding IDs."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    return plan_questions(results)[:limit]
