"""Synthetic deterministic planning tests; inference and downloads are unnecessary."""

import json
import re
import subprocess
import sys
from dataclasses import asdict, replace

import pytest

from stageguide.arguments.models import (
    ArgumentFinding, FindingEvidence, FindingType, Severity, SlideArgumentResult,
)
from stageguide.judge import (
    QuestionCategory, get_top_questions, plan_questions, plan_slide_questions,
)
from stageguide.judge.demo import sample_questions


def finding(kind=FindingType.UNSUPPORTED_NUMERIC_CLAIM, severity=Severity.HIGH,
            claim="StageGuide reduces presentation preparation time by 30%."):
    return ArgumentFinding(
        type=kind, severity=severity, claim=claim,
        reason="The finding's original explanation, retained without rewriting.",
        evidence=FindingEvidence(
            slide_text=claim, transcript_match="We help students rehearse.",
            related_transcript=["We help students rehearse."], supporting_excerpts=[],
            alignment_status="partial", alignment_similarity=0.4,
            numeric_values=["30%"], unmentioned_values=["30%"],
            trigger_phrases=["30%"], rule_id="numeric-omission",
        ), confidence="high",
    )


@pytest.mark.parametrize("kind,claim,category,phrase", [
    (FindingType.UNSUPPORTED_NUMERIC_CLAIM, "StageGuide reduces preparation time by 30%.",
     QuestionCategory.EVIDENCE, "claimed 30% reduction in preparation time by StageGuide"),
    (FindingType.UNEXPLAINED_NUMBER, "Target market: 5 million students",
     QuestionCategory.VALIDATION, "arrive at the estimate of 5 million students"),
    (FindingType.MISSING_KEY_CLAIM, "Practice helps students.",
     QuestionCategory.CLARIFICATION, "explain “Practice helps students” to the audience"),
    (FindingType.PARTIALLY_EXPLAINED_CLAIM, "Practice improves delivery and confidence.",
     QuestionCategory.CLARIFICATION, "fully explain “Practice improves delivery and confidence”"),
    (FindingType.EVIDENCE_GAP, "StageGuide is more effective.",
     QuestionCategory.EVIDENCE, "evidence or comparison supports"),
    (FindingType.VAGUE_CLAIM, "StageGuide provides a major improvement.",
     QuestionCategory.MEASUREMENT, "mean in measurable terms"),
    (FindingType.ASSUMPTION_TO_VALIDATE, "Universities will pay ₹999 per user.",
     QuestionCategory.VALIDATION, "universities would be willing to pay ₹999 per user"),
])
def test_all_finding_types(kind, claim, category, phrase):
    source = finding(kind, claim=claim)
    question, = plan_slide_questions(4, [source])
    assert phrase in question.question
    assert question.category == category
    assert question.source_finding_type == kind
    assert question.claim == claim
    assert question.question.endswith("?")
    assert question.slide_number == 4


@pytest.mark.parametrize("severity", list(Severity))
def test_priority_exactly_matches_severity(severity):
    question, = plan_slide_questions(1, [finding(severity=severity)])
    assert question.priority == severity


def test_severity_before_type_before_presentation_order():
    results = [
        SlideArgumentResult(7, [
            finding(FindingType.VAGUE_CLAIM, Severity.HIGH, "A major improvement"),
            finding(FindingType.UNSUPPORTED_NUMERIC_CLAIM, Severity.LOW, "40% faster"),
            finding(FindingType.EVIDENCE_GAP, Severity.MEDIUM, "Faster feedback"),
        ]),
        SlideArgumentResult(2, [finding(claim="30% faster")]),
        SlideArgumentResult(1, [finding(claim="20% faster")]),
    ]
    questions = plan_questions(results)
    assert [q.question_id for q in questions] == [
        "slide2_q1", "slide1_q1", "slide7_q1", "slide7_q3", "slide7_q2",
    ]


def test_recommended_type_tie_break_order():
    order = [FindingType.UNSUPPORTED_NUMERIC_CLAIM, FindingType.UNEXPLAINED_NUMBER,
             FindingType.MISSING_KEY_CLAIM, FindingType.EVIDENCE_GAP,
             FindingType.ASSUMPTION_TO_VALIDATE, FindingType.PARTIALLY_EXPLAINED_CLAIM,
             FindingType.VAGUE_CLAIM]
    questions = plan_slide_questions(1, [finding(kind, claim=kind.value) for kind in reversed(order)])
    assert [q.source_finding_type for q in questions] == order


def test_equal_rank_preserves_input_finding_order():
    questions = plan_slide_questions(1, [finding(claim="50% faster"), finding(claim="30% faster")])
    assert [q.claim for q in questions] == ["50% faster", "30% faster"]


@pytest.mark.parametrize("limit", [0, 1, 2, 5, 20])
def test_top_n_preserves_ids_and_is_prefix_of_full_plan(limit):
    results = [SlideArgumentResult(1, [finding(claim=f"{i}% faster") for i in range(7)])]
    assert get_top_questions(results, limit) == plan_questions(results)[:limit]
    assert len(get_top_questions(results)) == 5


@pytest.mark.parametrize("limit", [-1, 1.5, True, "2", None])
def test_invalid_limit_fails_clearly(limit):
    with pytest.raises(ValueError, match="non-negative integer"):
        get_top_questions([], limit)


def test_duplicate_claims_keep_highest_rank_and_all_provenance():
    gap = finding(FindingType.EVIDENCE_GAP)
    numeric = finding()
    questions = plan_slide_questions(3, [gap, numeric])
    question, = questions
    assert question.question_id == "slide3_q2"
    assert question.evidence_reference.finding_id == "slide3_f2"
    assert question.evidence_reference.finding == numeric
    reference, = question.additional_evidence_references
    assert reference.finding_id == "slide3_f1"
    assert reference.slide_number == 3
    assert reference.finding == gap


def test_dedup_chooses_severity_before_type():
    questions = plan_slide_questions(1, [
        finding(severity=Severity.MEDIUM), finding(FindingType.EVIDENCE_GAP),
    ])
    question, = questions
    assert question.source_finding_type == FindingType.EVIDENCE_GAP
    assert question.priority == Severity.HIGH


@pytest.mark.parametrize("variant", [
    "STAGEGUIDE reduces preparation time by 30%",
    "StageGuide  reduces\npreparation time by 30 %!",
    "“StageGuide reduces preparation time by 30%.”",
])
def test_dedup_case_whitespace_and_punctuation(variant):
    original = "StageGuide reduces preparation time by 30%."
    assert len(plan_slide_questions(1, [finding(claim=original), finding(claim=variant)])) == 1


@pytest.mark.parametrize("first,second", [
    ("30% faster", "40% faster"),
    ("30% faster", "30 faster"),
    ("Costs ₹999", "Costs $999"),
    ("Costs 9.99", "Costs 999"),
    ("Costs -50", "Costs 50"),
    ("It improves accuracy", "It does not improve accuracy"),
    ("Users will pay", "Users won't pay"),
    ("A beats B", "B beats A"),
    ("Target >30", "Target <30"),
])
def test_distinct_claims_are_not_collapsed(first, second):
    assert len(plan_slide_questions(1, [finding(claim=first), finding(claim=second)])) == 2


def test_same_claim_on_different_slides_keeps_separate_questions():
    questions = plan_questions([SlideArgumentResult(1, [finding()]), SlideArgumentResult(2, [finding()])])
    assert len(questions) == 2
    assert questions[0].evidence_reference.finding_id != questions[1].evidence_reference.finding_id


def test_limit_is_applied_after_deduplication():
    results = [SlideArgumentResult(1, [finding(), finding(), finding(claim="5 million users")])]
    assert len(get_top_questions(results, 2)) == 2


def test_full_evidence_reason_and_confidence_are_traceable_in_json():
    source = finding()
    question, = plan_slide_questions(4, [source])
    data = json.loads(json.dumps(question.to_dict()))
    assert data["reason"] == source.reason
    assert data["claim"] == source.claim
    assert data["priority"] == "HIGH"
    reference = data["evidence_reference"]
    assert reference["slide_number"] == 4
    assert reference["finding_id"] == "slide4_f1"
    assert reference["finding"] == asdict(source)
    assert reference["finding"]["evidence"]["slide_text"] == source.claim


def test_planning_does_not_mutate_inputs_and_snapshots_do_not_alias_inputs():
    source = finding()
    before = asdict(source)
    question, = plan_slide_questions(1, [source])
    assert asdict(source) == before
    source.evidence.numeric_values.append("99")
    assert question.evidence_reference.finding.evidence.numeric_values == ["30%"]
    exported = question.to_dict()
    exported["evidence_reference"]["finding"]["evidence"]["numeric_values"].append("100")
    assert question.evidence_reference.finding.evidence.numeric_values == ["30%"]


@pytest.mark.parametrize("results", [[], [SlideArgumentResult(1, [])]])
def test_empty_findings(results):
    assert plan_questions(results) == []
    assert get_top_questions(results) == []


def test_deterministic_output():
    results = [SlideArgumentResult(4, [finding(FindingType.EVIDENCE_GAP), finding()])]
    first = json.dumps([q.to_dict() for q in plan_questions(results)], ensure_ascii=False)
    assert all(json.dumps([q.to_dict() for q in plan_questions(results)], ensure_ascii=False) == first
               for _ in range(5))


@pytest.mark.parametrize("kind", list(FindingType))
def test_templates_do_not_invent_numbers_from_evidence(kind):
    source = finding(kind, claim="Practice improves clarity.")
    question, = plan_slide_questions(1, [source])
    assert "30" not in question.question  # Fixture evidence is not claim text.
    assert "Practice improves clarity" in question.question
    assert not re.search(r"\d", question.question)


@pytest.mark.parametrize("claim", [
    "It is 2x faster on a 45 TOPS device.",
    "StageGuide does not reduce time by 30%.",
    "StageGuide never reduces time by 30%.",
    "StageGuide reduces time by 30% only in a pilot.",
    "StageGuide reduces errors from 40% to 30%.",
])
def test_fallback_preserves_full_qualified_numeric_claim(claim):
    question, = plan_slide_questions(1, [finding(claim=claim)])
    assert claim.rstrip(".") in question.question
    assert re.findall(r"\d+", question.question) == re.findall(r"\d+", claim)


def test_assumption_fallback_preserves_organization_and_modality():
    claim = "HP can bundle StageGuide."
    question, = plan_slide_questions(1, [finding(FindingType.ASSUMPTION_TO_VALIDATE, claim=claim)])
    assert "validate the assumption “HP can bundle StageGuide”" in question.question


@pytest.mark.parametrize("number", [0, -1, True, 1.5])
def test_invalid_slide_number(number):
    with pytest.raises(ValueError, match="positive integer"):
        plan_slide_questions(number, [finding()])


def test_duplicate_slide_numbers_rejected_to_prevent_ambiguous_ids():
    with pytest.raises(ValueError, match="Duplicate slide_number"):
        plan_questions([SlideArgumentResult(1, [finding()]), SlideArgumentResult(1, [finding()])])


@pytest.mark.parametrize("claim", ["", "   ", "...?!"])
def test_empty_claim_rejected(claim):
    with pytest.raises(ValueError, match="non-empty claim"):
        plan_slide_questions(1, [finding(claim=claim)])


@pytest.mark.parametrize("field,value,match", [
    ("type", "UNKNOWN", "Unsupported finding type"),
    ("severity", "CRITICAL", "Unsupported severity"),
])
def test_unknown_values_fail_clearly(field, value, match):
    with pytest.raises(ValueError, match=match):
        plan_slide_questions(1, [replace(finding(), **{field: value})])


def test_demo_uses_real_analysis_and_produces_three_grounded_questions():
    questions = sample_questions()
    assert [q.source_finding_type for q in questions] == [
        FindingType.UNSUPPORTED_NUMERIC_CLAIM, FindingType.UNEXPLAINED_NUMBER,
        FindingType.ASSUMPTION_TO_VALIDATE,
    ]
    assert [q.priority for q in questions] == [Severity.HIGH, Severity.HIGH, Severity.MEDIUM]
    assert [q.question for q in questions] == [
        "What evidence supports the claimed 30% reduction in presentation preparation time by StageGuide?",
        "How did you arrive at the estimate of 5 million university students?",
        "What evidence suggests universities would be willing to pay ₹999 per user each year?",
    ]
    for question in questions:
        source = question.evidence_reference.finding
        assert question.claim == source.claim == source.evidence.slide_text
        assert question.reason == source.reason
        assert set(re.findall(r"\d+", question.question)) <= set(re.findall(r"\d+", source.claim))


def test_import_does_not_load_optional_inference_or_parsing_libraries():
    script = """
import sys
import stageguide.judge
assert not {'faster_whisper', 'ctranslate2', 'av', 'onnxruntime', 'numpy', 'pptx', 'pymupdf'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
