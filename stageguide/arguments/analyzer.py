"""Conservative, traceable rules over slide claims and their rehearsal coverage."""

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

from stageguide.alignment import align_presentation, align_slide, split_sentences, transcript_text
from stageguide.alignment.engine import TranscriptInput
from stageguide.alignment.models import ItemAlignment, SlideAlignmentResult
from stageguide.alignment.normalization import NEGATIONS
from stageguide.presentation.models import PresentationPage

from .models import ArgumentFinding, FindingEvidence, FindingType as Type, Severity, SlideArgumentResult
from .rules import (
    ACKNOWLEDGED, ASSERTED_BENEFIT, ASSERTION, BENEFIT, BUSINESS_ASSUMPTION,
    FOLLOWUP, GOAL, LABEL, MEASURABLE_SUBJECT, VAGUE, claim_numbers, claim_tokens, support_attempt,
)


@dataclass(frozen=True)
class _Claim:
    text: str
    item: ItemAlignment


def _claims(alignment: SlideAlignmentResult) -> List[_Claim]:
    """Reuse alignment's sentence/bullet extraction, grouping numeric subitems."""
    claims, seen = [], set()
    for item in alignment.items:
        text = item.text if item.kind in {"title", "body"} else item.context
        if not text:
            # A contextless metadata value is not enough to infer an argument.
            continue
        key = claim_tokens(text), frozenset(n.key for n in claim_numbers(text))
        if key not in seen:
            seen.add(key)
            claims.append(_Claim(text, item))
    return claims


def _related(claim: str, sentences: List[str], allow_uncertainty: bool = False) -> List[str]:
    words = claim_tokens(claim)
    if not words:
        return []
    indices = []
    for index, sentence in enumerate(sentences):
        other = claim_tokens(sentence)
        shared = words & other
        polarity_agrees = bool(words & NEGATIONS) == bool(other & NEGATIONS)
        acknowledges_uncertainty = allow_uncertainty and ACKNOWLEDGED.search(sentence)
        if ((polarity_agrees or acknowledges_uncertainty)
                and len(shared) >= (1 if len(words) <= 2 else 2) and len(shared) / len(words) >= 0.4):
            indices.append(index)
    # Support may follow the repeated claim in a directly linked sentence.
    # Do not borrow a survey about another topic merely because it is nearby.
    selected = set(indices)
    for index in indices:
        if index + 1 < len(sentences) and FOLLOWUP.search(sentences[index + 1]):
            next_sentence = sentences[index + 1]
            if support_attempt(next_sentence) or ACKNOWLEDGED.search(next_sentence):
                selected.add(index + 1)
    return [sentences[index] for index in sorted(selected)]


def _validate_inputs(slide: PresentationPage, text: str, alignment: SlideAlignmentResult) -> None:
    if alignment.slide_number != slide.page_number or alignment.title != slide.title:
        raise ValueError("Alignment does not belong to this slide number/title")
    source = " ".join(((slide.title or "") + "\n" + slide.body_text).split())
    rehearsal = " ".join(text.split())
    for item in alignment.items:
        claim = item.text if item.kind in {"title", "body"} else item.context
        if claim and " ".join(claim.split()) not in source:
            raise ValueError("Alignment contains claim text absent from this slide")
        if item.matched_text and " ".join(item.matched_text.split()) not in rehearsal:
            raise ValueError("Alignment contains evidence absent from this transcript")


def analyze_slide(
    slide: PresentationPage, transcript: TranscriptInput,
    alignment: Optional[SlideAlignmentResult] = None,
) -> SlideArgumentResult:
    """Analyze caller-selected rehearsal text; never assert a claim is false.

    Supply alignment from the same inputs, or let the existing aligner compute it.
    Header/source/evidence checks catch obvious mismatches, not every stale result.
    One highest-priority finding per claim avoids piling up redundant criticism.
    """
    text = transcript_text(transcript)
    alignment = alignment if alignment is not None else align_slide(slide, transcript)
    _validate_inputs(slide, text, alignment)
    sentences = split_sentences(text)
    findings = []
    for claim in _claims(alignment):
        original, item = claim.text, claim.item
        numbers = claim_numbers(original)
        words = claim_tokens(original)
        assumption = BUSINESS_ASSUMPTION.search(original)
        vague = VAGUE.search(original)
        benefit = BENEFIT.search(original)
        # Questions, honest goals and bare headings are not assertions of success.
        if LABEL.fullmatch(original) or original.rstrip().endswith("?") or (GOAL.search(original) and not assumption):
            continue
        if not (numbers or assumption or vague or ASSERTION.search(original)):
            continue
        if not words:
            continue
        related = _related(original, sentences, allow_uncertainty=bool(assumption))
        supporting = [sentence for sentence in related if support_attempt(sentence)]
        spoken_values = {n.key for sentence in related for n in claim_numbers(sentence)}
        missing = [n.text for n in numbers if n.key not in spoken_values]
        acknowledged = ACKNOWLEDGED.search(original) or any(ACKNOWLEDGED.search(sentence) for sentence in related)

        def emit(kind, severity, reason, rule_id, triggers, confidence="medium"):
            findings.append(ArgumentFinding(
                type=kind, severity=severity, claim=original, reason=reason,
                evidence=FindingEvidence(
                    slide_text=original, transcript_match=item.matched_text,
                    related_transcript=list(related), supporting_excerpts=list(supporting),
                    alignment_status=item.status, alignment_similarity=item.similarity,
                    numeric_values=[n.text for n in numbers], unmentioned_values=list(missing),
                    trigger_phrases=triggers, rule_id=rule_id,
                ), confidence=confidence,
            ))

        # Future business statements are validation tasks, not established outcomes.
        # Acknowledged hypotheses or connected support attempts receive no criticism.
        if assumption:
            if not acknowledged and not supporting:
                emit(Type.ASSUMPTION_TO_VALIDATE, Severity.MEDIUM,
                     "This future payment, adoption or partnership assertion needs validation; "
                     "no related validation attempt or explicit acknowledgment of uncertainty was detected in the supplied rehearsal.",
                     "business-assumption", [assumption.group()], "high")
            continue

        if numbers and missing:
            severity = Severity.HIGH if len(missing) == len(numbers) else Severity.MEDIUM
            kind = Type.UNSUPPORTED_NUMERIC_CLAIM if benefit else Type.UNEXPLAINED_NUMBER
            emit(kind, severity,
                 "The value(s) " + ", ".join(missing) + " were not matched in a related rehearsal sentence. "
                 + ("The quantified benefit needs a verbal explanation and supporting basis."
                    if benefit else "Explain the figure and what it represents; discussing the audience or topic alone does not cover it."),
                 "numeric-omission", missing, "high")
            continue

        if numbers and benefit and not supporting:
            emit(Type.UNSUPPORTED_NUMERIC_CLAIM, Severity.MEDIUM,
                 "The quantified benefit was mentioned, but no related source, measurement, comparison or mechanism was detected. "
                 "Repeating the value alone is not treated as supporting evidence.",
                 "numeric-support", [benefit.group()])
            continue

        if vague:
            measurable = bool(numbers or any(claim_numbers(sentence) for sentence in related))
            if not measurable:
                emit(Type.VAGUE_CLAIM, Severity.LOW,
                     "The phrase '" + vague.group() + "' gives no measurable extent, and no related quantitative detail was detected in the supplied rehearsal.",
                     "vague-unmeasured", [vague.group()])
            continue

        # Connected support also suppresses weak criticisms caused by lexical
        # paraphrase misses. Support is an attempt, not a correctness guarantee.
        if supporting:
            continue
        if item.status == "missing":
            emit(Type.MISSING_KEY_CLAIM, Severity.HIGH if len(words) >= 3 else Severity.LOW,
                 "This substantive slide claim has less than 40% keyword coverage in the alignment, "
                 "and no related supporting explanation was detected in the supplied rehearsal.",
                 "alignment-missing", [item.status])
        elif item.status == "partial":
            emit(Type.PARTIALLY_EXPLAINED_CLAIM, Severity.MEDIUM,
                 "Alignment marks this claim as partial: some content was mentioned, but the full claim did not reach the explained threshold.",
                 "alignment-partial", [item.status])
        elif ASSERTED_BENEFIT.search(original) and MEASURABLE_SUBJECT.search(original) and len(words) >= 3:
            emit(Type.EVIDENCE_GAP, Severity.MEDIUM,
                 "The benefit assertion was covered verbally, but no related source, measurement, comparison or mechanism was detected. "
                 "A supporting explanation would make the argument easier to assess.",
                 "evidence-gap", [benefit.group()] if benefit else [])
    return SlideArgumentResult(slide.page_number, findings)


def analyze_presentation(
    slides: Sequence[PresentationPage], transcript: TranscriptInput = "", *,
    transcripts_by_slide: Optional[Mapping[int, TranscriptInput]] = None,
) -> List[SlideArgumentResult]:
    """Reuse alignment's shared/explicit assignment modes; no slide-change detection."""
    alignments = align_presentation(slides, transcript, transcripts_by_slide=transcripts_by_slide)
    return [analyze_slide(
        slide, transcripts_by_slide.get(slide.page_number, "") if transcripts_by_slide is not None else transcript, result,
    ) for slide, result in zip(slides, alignments)]
