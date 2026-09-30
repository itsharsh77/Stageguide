"""Speech unit tests do not download weights or run a real model."""

import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import wave

import pytest

from stageguide.speech.backend import FasterWhisperBackend
from stageguide.speech.metrics import count_words, detect_fillers, summarize_transcript, words_per_minute
from stageguide.speech.models import EngineTranscript, SpeechError, TranscriptSegment, UnsupportedAudioError
from stageguide.speech.transcriber import main, transcribe_audio


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 16000)
    return path


@pytest.fixture
def fake_backend():
    return Mock(transcribe=Mock(return_value=EngineTranscript(10.0, [
        TranscriptSegment(1.0, 3.0, "Um, today we begin."),
        TranscriptSegment(5.0, 8.0, "You know, like this."),
    ])))


def test_result_structure_and_backend_injection(audio_path, fake_backend):
    result = transcribe_audio(audio_path, fake_backend)
    fake_backend.transcribe.assert_called_once_with(audio_path)
    assert result.to_dict() == {
        "duration_seconds": 5.0,
        "audio_duration_seconds": 10.0,
        "word_count": 8,
        "words_per_minute": 96.0,
        "filler_words": {"um": 1, "you know": 1, "like": 1},
        "segments": [
            {"start": 1.0, "end": 3.0, "text": "Um, today we begin."},
            {"start": 5.0, "end": 8.0, "text": "You know, like this."},
        ],
        "full_transcript": "Um, today we begin. You know, like this.",
    }
    data = result.to_dict()
    data["segments"][0]["text"] = "changed"
    data["filler_words"]["um"] = 99
    assert result.segments[0].text == "Um, today we begin."
    assert result.filler_words["um"] == 1
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()


@pytest.mark.parametrize("text,count", [
    ("", 0), ("  \n\t", 0), ("... — !", 0),
    ("Hello, world!", 2), ("We're ready—it’s real-time.", 4),
    ("café naïve 2026", 3), ("one\n two\tthree", 3),
])
def test_word_count(text, count):
    assert count_words(text) == count


@pytest.mark.parametrize("count,duration,expected", [
    (108, 42.8, 151.4), (10, 60, 10.0), (0, 12, 0.0), (0, 0, 0.0), (5, 0, 0.0),
])
def test_wpm(count, duration, expected):
    assert words_per_minute(count, duration) == expected


@pytest.mark.parametrize("count,duration", [(-1, 2), (1, -1), (1, float("nan")), (1, float("inf"))])
def test_invalid_wpm_inputs(count, duration):
    with pytest.raises(ValueError):
        words_per_minute(count, duration)


def test_fillers_case_punctuation_boundaries_and_phrases():
    text = "UM, um! Uh... like likely unlike umbrella. You know, I mean, erm, er, hmm."
    assert detect_fillers(text) == {
        "um": 2, "uh": 1, "like": 1, "you know": 1,
        "i mean": 1, "erm": 1, "er": 1, "hmm": 1,
    }
    assert detect_fillers("A clear statement.") == {}
    assert detect_fillers("") == {}


def test_custom_fillers_longest_match_without_double_counting():
    assert detect_fillers("You know you know um", ["you", "you know", "UM", "um", ""]) == {
        "you know": 2, "um": 1,
    }


def test_segment_gaps_overlaps_and_empty_text():
    result = summarize_transcript(EngineTranscript(15.0, [
        TranscriptSegment(1, 4, "One"), TranscriptSegment(3, 5, "two"),
        TranscriptSegment(7, 9, "three"), TranscriptSegment(9, 14, "  "),
    ]))
    assert result.duration_seconds == 6.0
    assert result.word_count == 3
    assert result.words_per_minute == 30.0
    assert len(result.segments) == 3


def test_no_speech_and_zero_length_segment():
    silent = summarize_transcript(EngineTranscript(10.0, []))
    assert silent.full_transcript == ""
    assert silent.duration_seconds == silent.words_per_minute == silent.word_count == 0
    assert silent.filler_words == {}
    assert silent.audio_duration_seconds == 10
    instantaneous = summarize_transcript(EngineTranscript(1, [TranscriptSegment(0, 0, "Hello")]))
    assert instantaneous.words_per_minute == 0


@pytest.mark.parametrize("start,end", [(-1, 2), (3, 2), (0, float("inf")), (float("nan"), 2)])
def test_invalid_segment_timestamps(start, end):
    with pytest.raises(SpeechError):
        TranscriptSegment(start, end, "test")


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_invalid_audio_duration(duration):
    with pytest.raises(SpeechError):
        EngineTranscript(duration, [])


def test_segment_contract_rejects_outside_audio_and_out_of_order():
    with pytest.raises(SpeechError):
        EngineTranscript(1, [TranscriptSegment(0, 2, "too long")])
    with pytest.raises(SpeechError):
        EngineTranscript(10, [TranscriptSegment(5, 6, "later"), TranscriptSegment(1, 2, "earlier")])


@pytest.mark.parametrize("name", ["missing.wav", "missing.mp3", "missing.flac"])
def test_missing_paths(tmp_path, name, fake_backend):
    with pytest.raises(SpeechError, match="does not exist"):
        transcribe_audio(tmp_path / name, fake_backend)
    fake_backend.transcribe.assert_not_called()


@pytest.mark.parametrize("name", ["audio.txt", "video.mp4", "audio", "deck.pptx"])
def test_unsupported_formats(tmp_path, name, fake_backend):
    with pytest.raises(UnsupportedAudioError):
        transcribe_audio(tmp_path / name, fake_backend)
    fake_backend.transcribe.assert_not_called()


def test_empty_file_and_directory(tmp_path, fake_backend):
    empty = tmp_path / "empty.wav"
    empty.touch()
    with pytest.raises(SpeechError, match="empty"):
        transcribe_audio(empty, fake_backend)
    directory = tmp_path / "directory.wav"
    directory.mkdir()
    with pytest.raises(SpeechError, match="regular file"):
        transcribe_audio(directory, fake_backend)
    fake_backend.transcribe.assert_not_called()


def test_unreadable_audio(audio_path, fake_backend, monkeypatch):
    monkeypatch.setattr(Path, "open", Mock(side_effect=PermissionError("denied")))
    with pytest.raises(SpeechError, match="Cannot transcribe"):
        transcribe_audio(audio_path, fake_backend)
    fake_backend.transcribe.assert_not_called()


def test_uppercase_extension_and_string_path(audio_path, fake_backend):
    uppercase = audio_path.with_suffix(".WAV")
    audio_path.rename(uppercase)
    assert transcribe_audio(str(uppercase), fake_backend).word_count == 8


def test_engine_failure_is_wrapped(audio_path, fake_backend):
    fake_backend.transcribe.side_effect = RuntimeError("failed")
    with pytest.raises(SpeechError, match="backend failure") as error:
        transcribe_audio(audio_path, fake_backend)
    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.fixture
def mock_whisper(tmp_path, monkeypatch):
    """Fake both optional modules so tests work with only stageguide[test]."""
    module = ModuleType("faster_whisper")
    decoder = ModuleType("faster_whisper.audio")
    decoder.decode_audio = Mock(return_value=[0] * 16000)
    model = Mock()
    model.transcribe.return_value = (iter([
        SimpleNamespace(start=0.1, end=0.9, text="  Um,  hello. "),
    ]), SimpleNamespace(duration=1.0))
    module.WhisperModel = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", decoder)
    folder = tmp_path / "model"
    folder.mkdir()
    for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.json"):
        (folder / name).touch()
    return module, decoder, model, folder


def test_adapter_local_only_and_timestamp_mapping(audio_path, mock_whisper):
    module, decoder, model, folder = mock_whisper
    backend = FasterWhisperBackend(folder)
    transcript = backend.transcribe(audio_path)
    assert transcript == EngineTranscript(1.0, [TranscriptSegment(0.1, 0.9, "Um, hello.")])
    module.WhisperModel.assert_called_once_with(
        str(folder.resolve()), device="cpu", compute_type="int8", local_files_only=True, cpu_threads=4,
    )
    decoder.decode_audio.assert_called_once_with(str(audio_path), sampling_rate=16000)
    assert model.transcribe.call_args.kwargs["vad_filter"] is True
    assert model.transcribe.call_args.kwargs["language"] == "en"
    assert backend._load_model() is model
    assert module.WhisperModel.call_count == 1


def test_missing_tokenizer_cannot_trigger_network_fallback(audio_path, mock_whisper):
    module, _, _, folder = mock_whisper
    (folder / "tokenizer.json").unlink()
    with pytest.raises(SpeechError, match="missing or incomplete"):
        FasterWhisperBackend(folder).transcribe(audio_path)
    module.WhisperModel.assert_not_called()


def test_plain_text_vocabulary_supported(audio_path, mock_whisper):
    _, _, _, folder = mock_whisper
    (folder / "vocabulary.json").rename(folder / "vocabulary.txt")
    assert FasterWhisperBackend(folder).transcribe(audio_path).segments[0].text == "Um, hello."


def test_corrupt_audio_before_model_load(audio_path, mock_whisper):
    module, decoder, _, folder = mock_whisper
    decoder.decode_audio.side_effect = ValueError("invalid audio data")
    with pytest.raises(SpeechError, match="Local transcription failed"):
        FasterWhisperBackend(folder).transcribe(audio_path)
    module.WhisperModel.assert_not_called()


def test_zero_samples_before_model_load(audio_path, mock_whisper):
    module, decoder, _, folder = mock_whisper
    decoder.decode_audio.return_value = []
    with pytest.raises(SpeechError, match="no samples"):
        FasterWhisperBackend(folder).transcribe(audio_path)
    module.WhisperModel.assert_not_called()


def test_lazy_inference_failure(audio_path, mock_whisper):
    _, _, model, folder = mock_whisper
    def broken_segments():
        yield SimpleNamespace(start=0, end=0.5, text="partial")
        raise RuntimeError("inference failed")
    model.transcribe.return_value = (broken_segments(), None)
    with pytest.raises(SpeechError, match="Local transcription failed"):
        FasterWhisperBackend(folder).transcribe(audio_path)


def test_missing_optional_dependency(audio_path, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", None)
    with pytest.raises(SpeechError, match="dependencies are missing"):
        FasterWhisperBackend(tmp_path).transcribe(audio_path)


def test_cli_json_success(audio_path, fake_backend, monkeypatch, capsys):
    monkeypatch.setattr("stageguide.speech.transcriber.create_backend", Mock(return_value=fake_backend))
    assert main([str(audio_path)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["word_count"] == 8


@pytest.mark.parametrize("name", ["missing.wav", "invalid.txt"])
def test_module_cli_error(tmp_path, name):
    result = subprocess.run(
        [sys.executable, "-m", "stageguide.speech.transcriber", str(tmp_path / name)],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"]
    assert "Traceback" not in result.stderr
