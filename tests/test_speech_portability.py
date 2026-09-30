"""Backend selection and truthful diagnostics without SDKs or model inference."""

import ast
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from stageguide.speech import (
    BackendConfig, BackendUnavailableError, create_backend, platform_diagnostics,
)
from stageguide.speech.backend import FasterWhisperBackend, WhisperCPUBackend
from stageguide.speech.models import EngineTranscript, TranscriptSegment
from stageguide.speech.qnn_backend import QualcommQNNBackend
from stageguide.speech.transcriber import main, transcribe_audio


def test_default_backend_and_legacy_alias():
    backend = create_backend()
    assert isinstance(backend, WhisperCPUBackend)
    assert FasterWhisperBackend is WhisperCPUBackend
    assert backend._model is None
    assert backend.metadata.to_dict() == {
        "backend_name": "WhisperCPUBackend", "model_name": "tiny.en",
        "execution_provider": "CPU", "device_type": "CPU",
        "accelerator": "none", "is_hardware_accelerated": False,
    }


def test_configured_model_label_and_language(tmp_path):
    config = BackendConfig(model_path=tmp_path / "custom-weights", language="fr", model_name="small")
    backend = create_backend(config)
    assert backend.model_path == config.model_path
    assert backend.language == "fr"
    assert backend.metadata.model_name == "small"
    assert backend._model is None
    # Never label an arbitrary directory as tiny.en by default.
    assert create_backend(BackendConfig(model_path=tmp_path / "custom")).metadata.model_name == "custom"
    with pytest.raises(FrozenInstanceError):
        config.backend = "qnn_whisper"


def test_metadata_independent_and_immutable():
    metadata = create_backend().metadata
    data = metadata.to_dict()
    data["accelerator"] = "NPU"
    assert metadata.accelerator == "none"
    with pytest.raises(FrozenInstanceError):
        metadata.is_hardware_accelerated = True


@pytest.mark.parametrize("name", ["", "auto", "cuda", "unknown"])
def test_unknown_backend_fails_without_fallback(name, monkeypatch):
    constructor = Mock(side_effect=AssertionError("CPU fallback must not occur"))
    monkeypatch.setattr("stageguide.speech.backend.WhisperCPUBackend", constructor)
    with pytest.raises(BackendUnavailableError, match="Unknown speech backend"):
        create_backend(BackendConfig(backend=name))
    constructor.assert_not_called()


@pytest.mark.parametrize("os_name,arch", [("Darwin", "arm64"), ("Windows", "ARM64"), ("Linux", "x86_64")])
def test_qnn_always_fails_clearly(os_name, arch, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: os_name)
    monkeypatch.setattr("platform.machine", lambda: arch)
    constructor = Mock(side_effect=AssertionError("CPU fallback must not occur"))
    monkeypatch.setattr("stageguide.speech.backend.WhisperCPUBackend", constructor)
    with pytest.raises(BackendUnavailableError, match="planned, not implemented or tested"):
        create_backend(BackendConfig(backend="qnn_whisper"))
    with pytest.raises(BackendUnavailableError, match="No Qualcomm inference was run"):
        QualcommQNNBackend()
    constructor.assert_not_called()


def test_qnn_stub_methods_cannot_return_fake_results():
    # Even bypassing its rejecting constructor cannot obtain results or NPU metadata.
    stub = object.__new__(QualcommQNNBackend)
    with pytest.raises(BackendUnavailableError):
        stub.transcribe(Path("example.wav"))
    with pytest.raises(BackendUnavailableError):
        _ = stub.metadata


@pytest.mark.parametrize("os_name,arch", [("Darwin", "arm64"), ("Windows", "ARM64"), ("Linux", "x86_64")])
def test_diagnostics_do_not_infer_qnn_from_platform(os_name, arch, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: os_name)
    monkeypatch.setattr("platform.machine", lambda: arch)
    monkeypatch.setattr("stageguide.speech.diagnostics.find_spec", lambda name: object())
    report = platform_diagnostics()
    assert report["operating_system"] == os_name
    assert report["machine_architecture"] == arch
    assert report["available_execution_backends"] == ["cpu_whisper"]
    assert report["backends"]["cpu_whisper"]["runtime_verified"] is False
    assert report["backends"]["qnn_whisper"]["implemented"] is False
    assert report["backends"]["qnn_whisper"]["status"] == "planned"


def test_missing_dependencies_are_reported(monkeypatch):
    monkeypatch.setattr("stageguide.speech.diagnostics.find_spec", lambda name: None)
    report = platform_diagnostics()
    assert report["available_execution_backends"] == []
    assert "faster_whisper" in report["backends"]["cpu_whisper"]["missing_dependencies"]
    assert report["backends"]["cpu_whisper"]["dependencies_detected"] is False
    # Configuration/metadata are still usable before optional packages are installed.
    assert create_backend().metadata.execution_provider == "CPU"


@pytest.mark.parametrize("error", [ImportError("missing"), ValueError("no module spec")])
def test_diagnostics_handle_unavailable_specs(monkeypatch, error):
    monkeypatch.setattr("stageguide.speech.diagnostics.find_spec", Mock(side_effect=error))
    assert platform_diagnostics()["available_execution_backends"] == []


def test_factory_cli_selection_and_unchanged_transcript(tmp_path, monkeypatch, capsys):
    path = tmp_path / "example.wav"
    path.write_bytes(b"mock audio")
    engine = Mock(transcribe=Mock(return_value=EngineTranscript(
        3.0, [TranscriptSegment(0.0, 2.0, "Hello world.")],
    )))
    factory = Mock(return_value=engine)
    monkeypatch.setattr("stageguide.speech.transcriber.create_backend", factory)
    assert main([str(path), "--backend", "cpu_whisper", "--model-path", "weights", "--model-name", "base"]) == 0
    factory.assert_called_once_with(BackendConfig(model_path=Path("weights"), model_name="base"))
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == transcribe_audio(path, engine).to_dict()
    assert set(json.loads(output.out)) == {
        "duration_seconds", "audio_duration_seconds", "word_count", "words_per_minute",
        "filler_words", "segments", "full_transcript",
    }


def test_cli_backend_info_needs_no_model(capsys):
    assert main(["--backend-info", "--model-path", "/missing/whisper-tiny.en"]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out)["is_hardware_accelerated"] is False


def test_cli_diagnostics_never_constructs_backend(monkeypatch, capsys):
    factory = Mock(side_effect=AssertionError("diagnostics should not construct a backend"))
    monkeypatch.setattr("stageguide.speech.transcriber.create_backend", factory)
    assert main(["--diagnostics"]) == 0
    assert "machine_architecture" in json.loads(capsys.readouterr().out)
    factory.assert_not_called()


@pytest.mark.parametrize("args", [
    ["--backend", "qnn_whisper", "example.wav"],
    ["--backend", "qnn_whisper", "--backend-info"],
    ["--backend", "unknown", "--backend-info"],
])
def test_cli_selection_failure_is_json(args, capsys):
    assert main(args) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["error"]
    assert "Traceback" not in output.err


def test_cli_still_requires_audio(capsys):
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
    assert "audio path is required" in capsys.readouterr().err


def test_public_api_and_diagnostics_never_import_inference_libraries():
    script = """
import sys
from stageguide.speech import create_backend, platform_diagnostics, BackendConfig, BackendUnavailableError
assert create_backend().metadata.accelerator == 'none'
platform_diagnostics()
try:
    create_backend(BackendConfig(backend='qnn_whisper'))
except BackendUnavailableError:
    pass
else:
    raise AssertionError('QNN succeeded')
for module in ('faster_whisper', 'ctranslate2', 'av', 'onnxruntime', 'numpy', 'tokenizers'):
    assert module not in sys.modules, module
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_application_import_boundary():
    """Keep engine libraries/concrete adapters inside speech as the app grows."""
    root = Path(__file__).resolve().parents[1] / "stageguide"
    libraries = {"faster_whisper", "ctranslate2", "av", "onnxruntime"}
    concrete = {"FasterWhisperBackend", "WhisperCPUBackend", "QualcommQNNBackend"}
    for path in root.rglob("*.py"):
        if path.is_relative_to(root / "speech"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] in libraries for alias in node.names), path
                assert not any(alias.name in {"stageguide.speech.backend", "stageguide.speech.qnn_backend"}
                               for alias in node.names), path
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in libraries, path
                assert not any(alias.name in concrete for alias in node.names), path
                assert not (node.module or "").startswith("stageguide.speech.qnn_backend"), path
