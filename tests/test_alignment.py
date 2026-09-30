"""Small deterministic examples; no parsing libraries or models are required."""

import copy
import json
import subprocess
import sys

import pytest

from stageguide.alignment import align_presentation, align_slide
from stageguide.alignment.normalization import content_tokens, numeric_mentions
from stageguide.presentation.models import PresentationPage
from stageguide.speech.models import EngineTranscript, TranscriptionResult, TranscriptSegment


def slide(body="", title=None, number=1, numbers=None, percentages=None):
    return PresentationPage(number, title, body, numbers or [], percentages or [], "sample.pptx")


def test_exact_match_and_result_structure():
    result = align_slide(slide("Practice improves confidence.", title="Benefits", number=3),
                         "Benefits. Practice improves confidence.")
    assert result.slide_number == 3
    assert result.title == "Benefits"
    assert result.coverage_score == 1.0
    assert result.explained_items == ["Benefits", "Practice improves confidence."]
    assert result.partially_explained_items == result.missing_items == []
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()
    data = result.to_dict()
    data["items"][0]["text"] = "changed"
    assert result.items[0].text == "Benefits"


@pytest.mark.parametrize("transcript", [
    "practice improves confidence", "PRACTICE IMPROVES CONFIDENCE!",
    "Practice, improves: confidence.", "  Practice\t improves   confidence  ",
])
def test_case_punctuation_and_whitespace(transcript):
    assert align_slide(slide("Practice improves confidence."), transcript).coverage_score == 1


def test_word_forms_and_minor_rephrasing():
    result = align_slide(slide("Our initial target is university students."),
                         "First, we are targeting students at universities.")
    assert result.coverage_score == 1.0
    assert content_tokens("Reduces preparation") == content_tokens("decrease preparing")


@pytest.mark.parametrize("transcript,status,score", [
    ("alpha beta gamma delta", "explained", 1.0),
    ("alpha beta", "partial", 0.5),
    ("alpha", "missing", 0.0),
    ("unrelated topic", "missing", 0.0),
])
def test_thresholds(transcript, status, score):
    result = align_slide(slide("alpha beta gamma delta epsilon"), transcript)
    assert result.items[0].status == status
    assert result.coverage_score == score


@pytest.mark.parametrize("source,spoken,value", [
    ("5 million", "five million", "5000000"),
    ("5,000,000", "five million", "5000000"),
    ("21", "twenty-one", "21"),
    ("105", "one hundred and five", "105"),
    ("2,500", "two thousand five hundred", "2500"),
    ("1,200,000", "one million two hundred thousand", "1200000"),
    ("2.5 million", "two point five million", "2500000"),
    ("0.25", "zero point two five", "0.25"),
    (".50", "zero point five", "0.5"),
    ("-5.25", "minus five point two five", "-5.25"),
    ("−5", "negative five", "-5"),
    ("+5", "positive five", "5"),
    ("0", "zero", "0"),
])
def test_written_number_equivalence(source, spoken, value):
    assert numeric_mentions(source)[0].value == numeric_mentions(spoken)[0].value == value
    assert align_slide(slide(f"Target {source} students."), f"Target around {spoken} students.").coverage_score == 1


@pytest.mark.parametrize("spoken", ["30%", "30 %", "thirty percent", "thirty per cent", "30.00％"])
def test_percentage_equivalence(spoken):
    result = align_slide(slide("Reduce preparation time by 30%.", numbers=["30"], percentages=["30%"]),
                         f"Reduce preparation time by {spoken}.")
    assert result.coverage_score == 1
    assert [item.kind for item in result.items] == ["body", "percentage"]


@pytest.mark.parametrize("spoken", ["thirty students", "0.3", "20%", "thirty percentage points", "nothing"])
def test_percentage_is_not_plain_number_or_other_value(spoken):
    result = align_slide(slide("30%", numbers=["30"], percentages=["30%"]), spoken)
    assert result.missing_items == ["30%"]
    assert result.coverage_score == 0


def test_missing_number_cannot_be_explained_by_surrounding_words():
    result = align_slide(slide("StageGuide reduces preparation time by 30%."),
                         "StageGuide reduces preparation time.")
    assert result.partially_explained_items == ["StageGuide reduces preparation time by 30%."]
    assert result.missing_items == ["30%"]
    assert result.coverage_score == 0.1667  # 0.5 / (text 1 + percentage 2)


def test_wrong_number_and_number_in_unrelated_sentence():
    result = align_slide(slide("StageGuide reduces preparation time by 30%."),
                         "StageGuide reduces preparation time by 20%. Battery capacity increased 30%.")
    assert "30%" in result.missing_items
    assert not result.explained_items


def test_contextual_numbers_can_be_partial_but_must_match_value():
    result = align_slide(slide("Target 5 million university students."), "Five million students.")
    numeric = next(item for item in result.items if item.kind == "number")
    assert numeric.status == "missing"  # one of three context terms is insufficient
    result = align_slide(slide("Target 5 million university students."), "Five million university students.")
    assert next(item for item in result.items if item.kind == "number").status == "partial"


def test_numeric_claims_in_distinct_contexts_remain_independent():
    result = align_slide(slide("Revenue 5 million. Costs 5 million."), "Revenue five million.")
    numeric = [item for item in result.items if item.kind == "number"]
    assert [item.status for item in numeric] == ["explained", "missing"]
    assert [item.context for item in numeric] == ["Revenue 5 million.", "Costs 5 million."]


def test_duplicate_annotations_and_scaled_raw_numbers_do_not_add_claims():
    page = slide("Target 5 million students. Savings 30%.", numbers=["5", "30", "5"], percentages=["30%"])
    result = align_slide(page, "Target five million students. Savings thirty percent.")
    assert len(result.items) == 4
    assert result.coverage_score == 1


def test_metadata_only_numbers_and_percentages():
    result = align_slide(slide(numbers=["12", "30"], percentages=["30%"]), "Twelve and thirty percent.")
    assert {item.text for item in result.items} == {"12", "30%"}
    assert result.coverage_score == 1


def test_titles_bullets_and_repeated_content():
    page = slide("Practice improves confidence.\n• Target university students; Offer clear feedback.",
                 title="Practice improves confidence.")
    result = align_slide(page, "Practice improves confidence. Target university students. Offer clear feedback.")
    assert len(result.items) == 3
    assert result.coverage_score == 1


@pytest.mark.parametrize("body,title", [("", None), (" \n\t", "  "), ("the and of", None)])
def test_empty_or_uninformative_slide(body, title):
    result = align_slide(slide(body, title=title), "Lots of words.")
    assert result.coverage_score == 0
    assert result.items == []


@pytest.mark.parametrize("transcript", ["", "  \n\t", []])
def test_empty_transcript(transcript):
    result = align_slide(slide("Target 5 million students.", title="Market"), transcript)
    assert result.coverage_score == 0
    assert len(result.missing_items) == 3


def test_simple_negative_statement_does_not_explain_positive_claim():
    page = slide("StageGuide reduces preparation time by 30%.")
    result = align_slide(page, "StageGuide does not reduce preparation time by thirty percent.")
    assert result.coverage_score == 0
    assert len(result.missing_items) == 2
    assert align_slide(page, "StageGuide doesn't reduce preparation time by thirty percent.").coverage_score == 0


def test_no_accumulating_keywords_across_unrelated_sentences():
    result = align_slide(slide("alpha beta gamma delta"), "alpha. beta. gamma. delta.")
    assert result.coverage_score == 0


def test_speech_structures_and_segments_are_supported_without_mutation():
    segments = [TranscriptSegment(0, 1, "Target five million"), TranscriptSegment(1, 2, "students.")]
    engine = EngineTranscript(2, segments)
    transcript = TranscriptionResult(2, 2, 4, 120, {}, segments, "Target five million students.")
    before = copy.deepcopy(transcript)
    page = slide("Target 5 million students.")
    for value in (transcript, engine, segments, TranscriptSegment(0, 2, transcript.full_transcript)):
        assert align_slide(page, value).coverage_score == 1
    assert transcript == before


def test_shared_transcript_multiple_slides_preserves_order():
    pages = [slide("University students", number=2), slide("Battery capacity", number=1)]
    results = align_presentation(pages, "University students need feedback.")
    assert [item.slide_number for item in results] == [2, 1]
    assert [item.coverage_score for item in results] == [1, 0]
    assert align_presentation([], "anything") == []


def test_explicit_assignments_do_not_leak_content_between_slides():
    pages = [slide("University students", number=1), slide("University students", number=2)]
    results = align_presentation(pages, transcripts_by_slide={1: [TranscriptSegment(0, 2, "University students.")]})
    assert [item.coverage_score for item in results] == [1, 0]


def test_invalid_assignment_keys_duplicate_slides_and_mixed_modes():
    pages = [slide("hello")]
    with pytest.raises(ValueError, match="unknown slide"):
        align_presentation(pages, transcripts_by_slide={2: "hello"})
    with pytest.raises(ValueError, match="unique"):
        align_presentation(pages * 2, "hello")
    with pytest.raises(ValueError, match="either"):
        align_presentation(pages, "hello", transcripts_by_slide={1: "hello"})


@pytest.mark.parametrize("value", [None, 42, {"full_transcript": "text"}, ["not a segment"]])
def test_invalid_transcript_type(value):
    with pytest.raises(TypeError, match="Transcript must"):
        align_slide(slide(), value)


def test_weighted_score_is_repeatable_and_counts_titles():
    page = slide("Target 5 million students.\nReduce preparation time.", title="Market")
    text = "Market. Target five million students. Preparation time."
    result = align_slide(page, text)
    # title 1 + body 1 + numeric 2 + partial body 0.5, out of total weight 5
    assert result.coverage_score == 0.9
    assert align_slide(page, text).to_dict() == result.to_dict()


def test_leading_minus_is_preserved_and_bullet_marker_is_removed():
    assert align_slide(slide("-5%"), "minus five percent").coverage_score == 1
    assert align_slide(slide("-5%"), "five percent").coverage_score == 0
    assert align_slide(slide("- Target 5 million students"), "Target five million students").coverage_score == 1


def test_requested_demo_has_missing_numeric_claims_and_partial_concepts():
    page = slide("StageGuide reduces presentation preparation time by 30%.\n"
                 "Our initial target is 5 million university students.",
                 numbers=["30", "5"], percentages=["30%"])
    result = align_slide(page, "Our first users will be university students. "
                        "We want StageGuide to make presentation preparation easier.")
    assert result.missing_items == ["30%", "5 million"]
    assert len(result.partially_explained_items) == 2
    assert result.coverage_score == 0.1667
    assert [item.similarity for item in result.items if item.kind == "body"] == [0.6, 0.75]


def test_alignment_does_not_import_parsers_or_inference_libraries():
    script = """
import sys
from stageguide.alignment import align_slide
from stageguide.presentation.models import PresentationPage
assert align_slide(PresentationPage(1, None, 'Hello', [], [], 'demo'), 'Hello').coverage_score == 1
for name in ('faster_whisper', 'ctranslate2', 'av', 'onnxruntime', 'numpy', 'pptx', 'pymupdf'):
    assert name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
