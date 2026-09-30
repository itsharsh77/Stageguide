"""Synthetic argument checks and counterexamples; no inference or downloads."""

import copy
import json
import subprocess
import sys
from unittest.mock import Mock

import pytest

from stageguide.alignment import align_slide
from stageguide.arguments import FindingType as Type, Severity, analyze_presentation, analyze_slide
from stageguide.arguments.rules import claim_numbers, support_attempt
from stageguide.presentation.models import PresentationPage
from stageguide.speech.models import EngineTranscript, TranscriptSegment, TranscriptionResult


def slide(body="", title=None, number=1):
    return PresentationPage(number, title, body, [], [], "synthetic.pptx")


def one(body, transcript):
    result = analyze_slide(slide(body), transcript)
    assert len(result.findings) == 1
    return result.findings[0]


def test_unsupported_percentage_omission():
    finding = one("StageGuide improves presentation quality by 40%.", "StageGuide improves presentation quality.")
    assert finding.type == Type.UNSUPPORTED_NUMERIC_CLAIM
    assert finding.severity == Severity.HIGH
    assert finding.evidence.unmentioned_values == ["40%"]
    assert finding.evidence.rule_id == "numeric-omission"


@pytest.mark.parametrize("text", [
    "According to a student survey, StageGuide improves presentation quality by forty percent.",
    "A pilot measured a 40% improvement in StageGuide presentation quality.",
    "StageGuide improves presentation quality by 40% compared with manual practice.",
    "StageGuide improves presentation quality by 40%. This was measured in a pilot with 20 respondents.",
])
def test_supported_percentage_has_no_findings(text):
    assert analyze_slide(slide("StageGuide improves presentation quality by 40%."), text).findings == []


def test_repeating_percentage_is_not_support():
    claim = "StageGuide improves presentation quality by 40%."
    finding = one(claim, claim)
    assert finding.type == Type.UNSUPPORTED_NUMERIC_CLAIM
    assert finding.severity == Severity.MEDIUM
    assert finding.evidence.unmentioned_values == []
    assert finding.evidence.rule_id == "numeric-support"


def test_unexplained_market_number():
    finding = one("Target market: 5 million university students.", "Our target market is university students.")
    assert finding.type == Type.UNEXPLAINED_NUMBER
    assert finding.severity == Severity.HIGH
    assert finding.evidence.unmentioned_values == ["5 million"]


@pytest.mark.parametrize("claim,text", [
    ("Target market: 5 million students.", "Our target market is five million students."),
    ("Price: ₹500 per month.", "The price is five hundred rupees per month."),
    ("Device performance: 45 TOPS.", "Device performance is forty-five TOPS."),
])
def test_explained_numbers_have_no_findings(claim, text):
    assert analyze_slide(slide(claim), text).findings == []


def test_unexplained_tops_preserves_original_claim():
    finding = one("Device performance: 45 TOPS.", "Device performance matters.")
    assert finding.type == Type.UNEXPLAINED_NUMBER
    assert finding.claim == "Device performance: 45 TOPS."


def test_multiplier_claim_preserved_and_normalized():
    finding = one("2x improvement in inference throughput.", "Inference throughput improves.")
    assert finding.type == Type.UNSUPPORTED_NUMERIC_CLAIM
    assert finding.evidence.unmentioned_values == ["2x"]
    assert claim_numbers("2x")[0].key == claim_numbers("two times")[0].key
    assert analyze_slide(slide("2x improvement in inference throughput."),
                         "A benchmark measured two times improvement in inference throughput.").findings == []


def test_partly_covered_quantities_have_medium_severity():
    finding = one("Target market has 5 million students in 10 cities.",
                  "Our target market has five million students in cities.")
    assert finding.type == Type.UNEXPLAINED_NUMBER
    assert finding.severity == Severity.MEDIUM
    assert finding.evidence.unmentioned_values == ["10"]


def test_missing_major_claim():
    finding = one("StageGuide supports private offline rehearsal.", "The weather is sunny.")
    assert finding.type == Type.MISSING_KEY_CLAIM
    assert finding.severity == Severity.HIGH
    assert finding.evidence.transcript_match is None
    assert finding.evidence.alignment_status == "missing"


def test_minor_omission_is_low_severity():
    finding = one("Works offline.", "")
    assert finding.type == Type.MISSING_KEY_CLAIM
    assert finding.severity == Severity.LOW


def test_partial_claim():
    finding = one("StageGuide supports private offline rehearsal.", "StageGuide supports rehearsal.")
    assert finding.type == Type.PARTIALLY_EXPLAINED_CLAIM
    assert finding.severity == Severity.MEDIUM
    assert finding.evidence.alignment_similarity == 0.6


def test_evidence_gap_for_repeated_asserted_benefit():
    claim = "StageGuide reduces preparation time."
    finding = one(claim, claim)
    assert finding.type == Type.EVIDENCE_GAP
    assert finding.severity == Severity.MEDIUM
    assert finding.evidence.trigger_phrases == ["reduces"]


@pytest.mark.parametrize("text", [
    "StageGuide reduces preparation time because reusable notes eliminate repeated drafting.",
    "We tested how StageGuide reduces preparation time in our pilot.",
    "Based on a classroom study, StageGuide reduces preparation time.",
    "StageGuide reduces preparation time compared with manual slide review.",
])
def test_explanation_or_support_attempt_suppresses_evidence_gap(text):
    assert analyze_slide(slide("StageGuide reduces preparation time."), text).findings == []


@pytest.mark.parametrize("extra", [
    "A survey measured battery life in 100 devices.",
    "We will run a study of StageGuide preparation time.",
    "We have no data for StageGuide preparation time.",
    "StageGuide preparation time needs research.",
    "StageGuide preparation time has data.",
])
def test_unrelated_planned_negative_or_bare_cues_do_not_count_as_support(extra):
    claim = "StageGuide reduces preparation time."
    finding = one(claim, claim + " " + extra)
    assert finding.type == Type.EVIDENCE_GAP
    assert finding.evidence.supporting_excerpts == []


@pytest.mark.parametrize("phrase", ["significantly better", "very effective", "major improvement", "huge market", "highly efficient"])
def test_obvious_vague_claim(phrase):
    finding = one(phrase, phrase)
    assert finding.type == Type.VAGUE_CLAIM
    assert finding.severity == Severity.LOW
    assert finding.evidence.trigger_phrases == [phrase]


def test_measurable_detail_suppresses_vagueness():
    assert analyze_slide(slide("Huge market."), "The market has five million students.").findings == []


@pytest.mark.parametrize("claim", [
    "Users will pay ₹999 per month.", "Universities will adopt the platform.", "HP can bundle StageGuide.",
])
def test_explicit_business_assumption(claim):
    finding = one(claim, "")
    assert finding.type == Type.ASSUMPTION_TO_VALIDATE
    assert finding.severity == Severity.MEDIUM
    assert finding.claim == claim
    assert "needs validation" in finding.reason
    assert "false" not in finding.reason


@pytest.mark.parametrize("transcript", [
    "We assume universities will pay ₹999 per user each year.",
    "Universities will pay ₹999 per user each year. This hypothesis needs validation.",
    "A survey of universities found they will pay ₹999 per user each year.",
    "Universities will pay ₹999 per user each year, but this is not yet validated.",
])
def test_acknowledged_or_supported_assumption_has_no_finding(transcript):
    assert analyze_slide(slide("Universities will pay ₹999 per user each year."), transcript).findings == []


def test_labeled_slide_assumption_has_no_redundant_criticism():
    assert analyze_slide(slide("Hypothesis: users will pay ₹999 per month."), "").findings == []


@pytest.mark.parametrize("claim", [
    "Market Opportunity", "Agenda", "Summary", "Slide 4", "Version 2.0", "How can we improve presentation quality?",
    "Our goal is to reduce preparation time by 30%.",
    "StageGuide is designed to improve presentation quality.",
    "StageGuide may improve presentation quality.",
])
def test_headings_questions_and_explicit_goals_are_not_criticized(claim):
    assert analyze_slide(slide(claim), "").findings == []


def test_supported_past_adoption_is_not_an_assumption():
    claim = "Universities adopted the platform in a pilot."
    assert analyze_slide(slide(claim), claim).findings == []


def test_well_explained_noncomparative_claim():
    claim = "StageGuide supports private offline rehearsal."
    assert analyze_slide(slide(claim), claim).findings == []


@pytest.mark.parametrize("claim", ["Practice improves confidence.", "StageGuide is better.", "Read better books."])
def test_benefit_word_alone_does_not_justify_evidence_gap(claim):
    assert analyze_slide(slide(claim), claim).findings == []


def test_empty_slide_and_unlabeled_numeric_metadata():
    assert analyze_slide(slide(), "Long discussion.").findings == []
    assert analyze_slide(PresentationPage(1, None, "", ["42"], [], "test.pptx"), "").findings == []


def test_empty_transcript_numeric_claim_is_missing():
    finding = one("Target market: 5 million students.", "")
    assert finding.type == Type.UNEXPLAINED_NUMBER
    assert finding.evidence.related_transcript == []


def test_existing_alignment_is_consumed_without_recomputing(monkeypatch):
    page = slide("StageGuide supports offline rehearsal.")
    text = "StageGuide supports offline rehearsal."
    alignment = align_slide(page, text)
    monkeypatch.setattr("stageguide.arguments.analyzer.align_slide", Mock(side_effect=AssertionError("recomputed")))
    assert analyze_slide(page, text, alignment).findings == []


def test_alignment_input_mismatches_are_rejected():
    page = slide("StageGuide supports offline rehearsal.")
    alignment = align_slide(page, page.body_text)
    with pytest.raises(ValueError, match="number/title"):
        analyze_slide(slide(page.body_text, number=2), page.body_text, alignment)
    with pytest.raises(ValueError, match="claim text"):
        analyze_slide(slide("Completely different text."), page.body_text, alignment)
    with pytest.raises(ValueError, match="evidence absent"):
        analyze_slide(page, "Different transcript", alignment)


def test_result_determinism_serialization_traceability_and_no_mutation():
    page = slide("StageGuide reduces preparation time by 30%.")
    segments = [TranscriptSegment(0, 4, "StageGuide reduces preparation time.")]
    text = TranscriptionResult(4, 4, 5, 75, {}, segments, segments[0].text)
    alignment = align_slide(page, text)
    before = copy.deepcopy((page, text, alignment))
    result = analyze_slide(page, text, alignment)
    assert (page, text, alignment) == before
    assert result.to_dict() == analyze_slide(page, text, alignment).to_dict()
    finding = result.findings[0]
    assert finding.evidence.slide_text == page.body_text
    assert finding.evidence.transcript_match in text.full_transcript
    assert all(snippet in text.full_transcript for snippet in finding.evidence.related_transcript)
    assert finding.evidence.numeric_values == ["30%"]
    assert finding.evidence.rule_id
    assert finding.confidence in {"high", "medium"}
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()
    data = result.to_dict()
    data["findings"][0]["evidence"]["numeric_values"].append("99")
    assert finding.evidence.numeric_values == ["30%"]


def test_all_transcript_inputs_and_presentation_assignments():
    page = slide("Target market: 5 million students.")
    segment = TranscriptSegment(0, 4, "Our target market is five million students.")
    for transcript in (segment, [segment], EngineTranscript(4, [segment])):
        assert analyze_slide(page, transcript).findings == []
    pages = [page, slide(page.body_text, number=2)]
    assert [len(result.findings) for result in analyze_presentation(pages, segment.text)] == [0, 0]
    assert [len(result.findings) for result in analyze_presentation(pages, transcripts_by_slide={1: segment})] == [0, 1]
    assert analyze_presentation([], "") == []


def test_requested_demo_exactly_three_findings_without_piling_on():
    page = slide("StageGuide reduces presentation preparation time by 30%.\n\n"
                 "Our initial target market is 5 million university students.\n\n"
                 "Universities will pay ₹999 per user each year.")
    text = ("StageGuide is designed for university students preparing presentations.\n"
            "It helps them practise their delivery and argument before presenting.")
    findings = analyze_slide(page, text).findings
    assert [finding.type for finding in findings] == [
        Type.UNSUPPORTED_NUMERIC_CLAIM, Type.UNEXPLAINED_NUMBER, Type.ASSUMPTION_TO_VALIDATE,
    ]
    assert [finding.severity for finding in findings] == [Severity.HIGH, Severity.HIGH, Severity.MEDIUM]
    assert "₹999" in findings[2].claim


def test_numeric_value_in_unrelated_sentence_does_not_defend_claim():
    finding = one("StageGuide reduces preparation time by 30%.",
                  "StageGuide reduces preparation time. A benchmark measured battery savings of 30%.")
    assert finding.type == Type.UNSUPPORTED_NUMERIC_CLAIM
    assert finding.severity == Severity.HIGH


def test_this_followed_by_unrelated_subject_does_not_supply_evidence():
    finding = one("StageGuide reduces preparation time by 30%.",
                  "StageGuide reduces preparation time. This battery benchmark measured savings of 30%.")
    assert finding.type == Type.UNSUPPORTED_NUMERIC_CLAIM
    assert finding.severity == Severity.HIGH
    assert finding.evidence.supporting_excerpts == []


def test_analysis_imports_no_models_or_parsers():
    script = """
import sys
from stageguide.arguments import analyze_slide
from stageguide.presentation.models import PresentationPage
analyze_slide(PresentationPage(1, None, 'Target market: 5 million students.', [], [], 'test'), '')
for name in ('faster_whisper', 'ctranslate2', 'av', 'onnxruntime', 'numpy', 'pptx', 'pymupdf'):
    assert name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
