"""Connect existing services, manage temporary uploads and invalidate stale state."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

from stageguide.alignment import SlideAlignmentResult, align_presentation
from stageguide.arguments import SlideArgumentResult, analyze_slide
from stageguide.judge import (
    JudgeBackend, JudgeInteraction, JudgeQuestion, JudgeResponse, LocalJudge, get_top_questions,
)
from stageguide.presentation.models import PresentationPage
from stageguide.presentation.parser import parse_presentation
from stageguide.speech import SpeechBackend
from stageguide.speech.models import TranscriptionResult
from stageguide.speech.transcriber import transcribe_audio


MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class DemoError(ValueError):
    """Short display-safe error; backend exceptions remain chained for debugging."""


@contextmanager
def _uploaded_file(filename: str, data: bytes):
    name = Path(filename.replace("\\", "/")).name
    if not name or name in {".", ".."} or not data:
        raise DemoError("Choose a non-empty local file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise DemoError("Choose a file smaller than 50 MB.")
    with TemporaryDirectory(prefix="stageguide-") as folder:
        path = Path(folder) / name
        path.write_bytes(data)
        yield path


@dataclass
class DemoReview:
    coverage: List[SlideAlignmentResult]
    arguments: List[SlideArgumentResult]
    questions: List[JudgeQuestion]

    def to_dict(self):
        return asdict(self)


class DemoController:
    """One browser session, no global user data or persistent storage."""

    def __init__(self, speech_backend: Optional[SpeechBackend], judge_backend: JudgeBackend):
        self.speech_backend = speech_backend
        self.judge_backend = judge_backend
        self.pages: List[PresentationPage] = []
        self.filename = ""
        self.audio_filename = ""
        self.transcript: Optional[TranscriptionResult] = None
        self.review: Optional[DemoReview] = None
        self.refinements: Dict[str, JudgeResponse] = {}
        self.interactions: Dict[str, JudgeInteraction] = {}
        self.speech_failure = False
        self.revision = 0

    def clear_judge(self):
        self.refinements.clear()
        self.interactions.clear()
        self.revision += 1

    def clear_review(self):
        self.review = None
        self.clear_judge()

    def clear_rehearsal(self):
        self.transcript = None
        self.audio_filename = ""
        self.speech_failure = False
        self.clear_review()

    def clear_presentation(self):
        self.pages = []
        self.filename = ""
        self.clear_rehearsal()

    def set_speech_backend(self, backend: Optional[SpeechBackend]):
        self.speech_backend = backend
        self.clear_rehearsal()

    def set_judge_backend(self, backend: JudgeBackend):
        self.judge_backend = backend
        self.clear_judge()

    def load_presentation(self, filename: str, data: bytes):
        self.clear_presentation()
        try:
            with _uploaded_file(filename, data) as path:
                pages = parse_presentation(path)
                if not pages:
                    raise DemoError("No slides or pages were found in this presentation.")
                self.pages = pages
                self.filename = path.name
        except DemoError:
            raise
        except Exception as exc:
            raise DemoError("This presentation could not be read. Choose a valid, unencrypted PPTX or PDF.") from exc

    def transcribe(self, filename: str, data: bytes):
        self.clear_rehearsal()
        if not self.pages:
            raise DemoError("Upload a presentation before starting your rehearsal.")
        if self.speech_backend is None:
            raise DemoError("The speech backend is unavailable. Check the local backend settings.")
        try:
            with _uploaded_file(filename, data) as path:
                result = transcribe_audio(path, self.speech_backend)
                self.transcript = result
                self.audio_filename = path.name
        except DemoError:
            raise
        except Exception as exc:
            self.speech_failure = True
            raise DemoError(
                "Transcription failed. Check the audio and local speech model files in Backend settings."
            ) from exc

    def analyze(self):
        self.clear_review()
        if not self.pages:
            raise DemoError("Upload a presentation first.")
        if self.transcript is None:
            raise DemoError("Transcribe your rehearsal audio first.")
        if not self.transcript.full_transcript.strip():
            raise DemoError("No speech was detected. Try a clearer recording before reviewing coverage.")
        try:
            coverage = align_presentation(self.pages, self.transcript)
            arguments = [analyze_slide(page, self.transcript, aligned)
                         for page, aligned in zip(self.pages, coverage)]
            questions = get_top_questions(arguments, limit=5)
            self.review = DemoReview(coverage, arguments, questions)
            return self.review
        except Exception as exc:
            raise DemoError("The review could not be completed. Try another presentation or recording.") from exc

    def _question(self, question_id: str) -> JudgeQuestion:
        if self.review is not None:
            for question in self.review.questions:
                if question.question_id == question_id:
                    return question
        raise DemoError("Choose a question from the current review.")

    def refine(self, question_id: str) -> JudgeResponse:
        question = self._question(question_id)
        if question_id not in self.refinements:
            self.refinements[question_id] = LocalJudge(self.judge_backend).refine(question)
        return self.refinements[question_id]

    def answer(self, question_id: str, answer: str) -> JudgeInteraction:
        question = self._question(question_id)
        if not answer.strip():
            raise DemoError("Enter an answer before requesting a follow-up.")
        try:
            interaction = JudgeInteraction(
                original_question=deepcopy(question), refined_question=self.refine(question_id),
                presenter_answer=answer,
                follow_up=LocalJudge(self.judge_backend).follow_up(question, answer),
            )
            self.interactions[question_id] = interaction
            return interaction
        except Exception as exc:
            raise DemoError("The judge interaction could not be completed. Your grounded question is still available.") from exc

    def status(self):
        result = {}
        for name, backend in (("speech", self.speech_backend), ("judge", self.judge_backend)):
            try:
                result[name] = backend.metadata.to_dict() if backend is not None else None
            except Exception:
                result[name] = None
        return result

    def report(self):
        return {
            "source_filename": self.filename,
            "audio_filename": self.audio_filename,
            "backend_metadata": self.status(),
            "transcript": self.transcript.to_dict() if self.transcript is not None else None,
            "review": self.review.to_dict() if self.review is not None else None,
            "interactions": [interaction.to_dict() for interaction in self.interactions.values()],
        }


def generated_follow_up(interaction: JudgeInteraction) -> Optional[str]:
    """Do not present the deterministic fallback repeat as a generated follow-up."""
    response = interaction.follow_up
    if response is None or response.used_fallback or not response.grounded:
        return None
    return response.question
