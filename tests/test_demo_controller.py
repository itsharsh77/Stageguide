"""Orchestration tests need neither Streamlit execution nor model weights."""

import json
from pathlib import Path

import pymupdf
import pytest

from stageguide.demo.controller import DemoController, DemoError, generated_follow_up
from stageguide.judge import JudgeConfig, create_judge_backend
from stageguide.judge.backend import JudgeBackendMetadata
from stageguide.speech import BackendMetadata
from stageguide.speech.models import EngineTranscript, TranscriptSegment


@pytest.fixture
def pdf_bytes():
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 60), "Market opportunity", fontsize=26)
        page.insert_text((50, 120), "StageGuide reduces preparation time by 30%.", fontsize=16)
        return doc.tobytes()


class SpeechDouble:
    def __init__(self, text="StageGuide helps students prepare presentations."):
        self.text = text
        self.paths = []

    @property
    def metadata(self):
        return BackendMetadata("SpeechDouble", "test-model", "mock", "none", "none", False)

    def transcribe(self, path):
        assert path.is_file()
        self.paths.append(path)
        return EngineTranscript(5.0, [TranscriptSegment(0, 5, self.text)])


class JudgeDouble:
    def __init__(self):
        self.calls = []

    @property
    def metadata(self):
        return JudgeBackendMetadata("JudgeDouble", "test-model", "mock", "none", "none", False, True, "mock")

    def generate(self, request):
        self.calls.append(request)
        payload = json.loads(request.context_json)
        context = payload["context"]
        response = {"grounded": True, "source_question_id": context["source_question_id"]}
        if payload["task"] == "refine":
            response["question"] = context["judge_question"]
        else:
            response["follow_up_question"] = "What evidence supports the 30% figure?"
            response["reason"] = context["finding_reason"]
        return json.dumps(response)


@pytest.fixture
def controller():
    return DemoController(SpeechDouble(), create_judge_backend(JudgeConfig()))


def reviewed(controller, pdf_bytes):
    controller.load_presentation("demo.pdf", pdf_bytes)
    controller.transcribe("rehearsal.wav", b"audio passed to test backend")
    return controller.analyze()


def test_real_services_are_composed_and_original_results_retained(controller, pdf_bytes, monkeypatch):
    from stageguide.demo import controller as module
    original = module.analyze_slide
    supplied = []

    def observe(slide, transcript, alignment):
        supplied.append(alignment)
        return original(slide, transcript, alignment)

    monkeypatch.setattr(module, "analyze_slide", observe)
    result = reviewed(controller, pdf_bytes)
    assert supplied == result.coverage
    assert supplied[0] is result.coverage[0]
    assert result.arguments[0].findings
    assert result.questions[0].claim == "StageGuide reduces preparation time by 30%."
    assert result.questions[0].evidence_reference.finding in result.arguments[0].findings
    assert controller.pages[0].source_filename == "demo.pdf"
    assert controller.transcript.word_count > 0
    assert not controller.speech_backend.paths[0].exists()


def test_temporary_presentation_cleaned_and_filename_sanitized(controller, pdf_bytes, monkeypatch):
    from stageguide.demo import controller as module
    original = module.parse_presentation
    paths = []

    def observe(path):
        paths.append(path)
        assert path.is_file()
        return original(path)

    monkeypatch.setattr(module, "parse_presentation", observe)
    controller.load_presentation("../../unsafe/demo.pdf", pdf_bytes)
    assert controller.filename == "demo.pdf"
    assert not paths[0].exists()


@pytest.mark.parametrize("filename,data", [("broken.pdf", b"broken"), ("a.exe", b"x"), ("empty.pdf", b"")])
def test_invalid_presentation_has_safe_error_and_clears_old_state(controller, pdf_bytes, filename, data):
    reviewed(controller, pdf_bytes)
    with pytest.raises(DemoError):
        controller.load_presentation(filename, data)
    assert controller.pages == []
    assert controller.transcript is None
    assert controller.review is None


def test_changed_presentation_clears_answers_and_rehearsal(controller, pdf_bytes):
    review = reviewed(controller, pdf_bytes)
    controller.answer(review.questions[0].question_id, "This is an assumption.")
    controller.load_presentation("new.pdf", pdf_bytes)
    assert controller.transcript is None
    assert controller.review is None
    assert controller.interactions == {}
    assert controller.refinements == {}


def test_changed_audio_clears_review_and_judge(controller, pdf_bytes):
    review = reviewed(controller, pdf_bytes)
    controller.refine(review.questions[0].question_id)
    controller.transcribe("new.wav", b"new audio")
    assert controller.review is None and not controller.refinements
    assert controller.audio_filename == "new.wav"


def test_transcription_failure_cleaned_and_previous_results_removed(controller, pdf_bytes):
    reviewed(controller, pdf_bytes)
    paths = []

    def fail(path):
        paths.append(path)
        raise RuntimeError("private backend internals")

    controller.speech_backend.transcribe = fail
    with pytest.raises(DemoError, match="Transcription failed") as error:
        controller.transcribe("new.wav", b"broken recording")
    assert "private" not in str(error.value)
    assert not paths[0].exists()
    assert controller.transcript is None and controller.review is None
    assert controller.speech_failure


@pytest.mark.parametrize("filename,data", [("no.wav", b""), ("bad.zip", b"some data")])
def test_missing_or_unsupported_audio(controller, pdf_bytes, filename, data):
    controller.load_presentation("demo.pdf", pdf_bytes)
    with pytest.raises(DemoError):
        controller.transcribe(filename, data)
    assert controller.transcript is None


def test_oversized_upload(controller, monkeypatch):
    monkeypatch.setattr("stageguide.demo.controller.MAX_UPLOAD_BYTES", 2)
    with pytest.raises(DemoError, match="50 MB"):
        controller.load_presentation("large.pdf", b"large")


def test_no_presentation(controller):
    with pytest.raises(DemoError, match="Upload a presentation"):
        controller.transcribe("audio.wav", b"audio")
    with pytest.raises(DemoError, match="Upload a presentation"):
        controller.analyze()


def test_no_transcript(controller, pdf_bytes):
    controller.load_presentation("demo.pdf", pdf_bytes)
    with pytest.raises(DemoError, match="Transcribe"):
        controller.analyze()


def test_empty_transcript_does_not_create_findings(controller, pdf_bytes):
    controller.speech_backend.text = ""
    controller.load_presentation("demo.pdf", pdf_bytes)
    controller.transcribe("audio.wav", b"audio")
    with pytest.raises(DemoError, match="No speech"):
        controller.analyze()
    assert controller.review is None


def test_unavailable_speech_backend(controller, pdf_bytes):
    controller.set_speech_backend(None)
    controller.load_presentation("demo.pdf", pdf_bytes)
    with pytest.raises(DemoError, match="speech backend is unavailable"):
        controller.transcribe("audio.wav", b"audio")
    assert controller.status()["speech"] is None


def test_no_findings_produces_no_questions(controller):
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 60), "Agenda", fontsize=26)
        controller.load_presentation("agenda.pdf", doc.tobytes())
    controller.transcribe("audio.wav", b"audio")
    review = controller.analyze()
    assert review.arguments[0].findings == []
    assert review.questions == []


def test_fallback_answer_saved_but_not_shown_as_generated_followup(controller, pdf_bytes):
    review = reviewed(controller, pdf_bytes)
    interaction = controller.answer(review.questions[0].question_id, "We expect automation to help.")
    assert interaction.presenter_answer == "We expect automation to help."
    assert interaction.refined_question.used_fallback
    assert interaction.follow_up.used_fallback
    assert generated_follow_up(interaction) is None
    assert interaction.original_question == review.questions[0]


def test_real_backend_interface_can_generate_follow_up_and_refinement_is_cached(controller, pdf_bytes):
    backend = JudgeDouble()
    controller.set_judge_backend(backend)
    review = reviewed(controller, pdf_bytes)
    question_id = review.questions[0].question_id
    first = controller.refine(question_id)
    assert controller.refine(question_id) is first
    interaction = controller.answer(question_id, "We expect automation to help.")
    assert interaction.refined_question is first
    assert len(backend.calls) == 2
    assert generated_follow_up(interaction) == "What evidence supports the 30% figure?"


def test_empty_answer_does_not_create_interaction(controller, pdf_bytes):
    review = reviewed(controller, pdf_bytes)
    with pytest.raises(DemoError, match="Enter an answer"):
        controller.answer(review.questions[0].question_id, "   ")
    assert controller.interactions == {}


def test_stale_question_rejected(controller, pdf_bytes):
    reviewed(controller, pdf_bytes)
    with pytest.raises(DemoError, match="current review"):
        controller.answer("old_question", "answer")


def test_judge_backend_change_preserves_analysis_but_clears_interactions(controller, pdf_bytes):
    review = reviewed(controller, pdf_bytes)
    controller.answer(review.questions[0].question_id, "We expect automation to help.")
    controller.set_judge_backend(create_judge_backend(JudgeConfig(backend="qualcomm")))
    assert controller.review is review and controller.transcript is not None
    assert not controller.interactions and not controller.refinements
    assert controller.status()["judge"]["status"] == "not_implemented"


def test_speech_backend_change_requires_new_transcription(controller, pdf_bytes):
    reviewed(controller, pdf_bytes)
    controller.set_speech_backend(SpeechDouble())
    assert controller.pages
    assert controller.transcript is None and controller.review is None


def test_report_serializable_and_preserves_trace(controller, pdf_bytes):
    review = reviewed(controller, pdf_bytes)
    controller.answer(review.questions[0].question_id, "An assumption.")
    report = json.loads(json.dumps(controller.report()))
    assert report["source_filename"] == "demo.pdf"
    source = report["interactions"][0]["original_question"]["evidence_reference"]
    assert source["finding"]["claim"] == review.questions[0].claim
    assert source["finding_id"] == review.questions[0].evidence_reference.finding_id


def test_analysis_error_has_safe_message(controller, pdf_bytes, monkeypatch):
    reviewed(controller, pdf_bytes)
    def fail(*args):
        raise RuntimeError("internal details")
    monkeypatch.setattr("stageguide.demo.controller.align_presentation", fail)
    with pytest.raises(DemoError, match="review could not be completed"):
        controller.analyze()
    assert controller.review is None
