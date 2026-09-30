"""Runtime configuration/diagnostics with a fake native boundary, no weights."""

import json
import sys
from types import SimpleNamespace

import pytest

from stageguide.judge import JudgeConfig, LocalJudge, create_judge_backend
from stageguide.judge.demo import sample_questions
from stageguide.judge.local_backend import LocalDevelopmentJudgeBackend
from stageguide.judge.prompts import build_request


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    path = tmp_path / "test.gguf"
    path.write_bytes(b"test boundary only")
    state = SimpleNamespace(loads=[], closed=0, log=(
        "load_tensors: offloaded 37/37 layers to GPU\n"
        "load_tensors: MTL0 model buffer size = 2380.0 MiB\n"
    ), error=None)
    question = sample_questions()[0]

    class Model:
        def __init__(self, **kwargs):
            state.loads.append(kwargs)
            if kwargs["verbose"]:
                print(state.log, file=sys.stderr)

        def close(self):
            state.closed += 1

        def create_chat_completion(self, **kwargs):
            if state.error:
                raise state.error
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "question": question.question, "grounded": True,
                "source_question_id": question.question_id,
            })}}], "usage": {"prompt_tokens": 512, "completion_tokens": 64}}

    state.module = SimpleNamespace(Llama=Model, llama_supports_gpu_offload=lambda: True)
    monkeypatch.setitem(sys.modules, "llama_cpp", state.module)
    monkeypatch.setattr("stageguide.judge.local_backend.find_spec", lambda _: object())
    monkeypatch.setattr("stageguide.judge.local_backend.platform.system", lambda: "Darwin")
    monkeypatch.setattr("stageguide.judge.local_backend.platform.machine", lambda: "arm64")
    state.path, state.question = path, question
    return state


@pytest.mark.parametrize("acceleration", ["cuda", "qnn", "", None])
def test_invalid_acceleration(acceleration):
    with pytest.raises(ValueError, match="acceleration"):
        create_judge_backend(JudgeConfig(backend="local", acceleration=acceleration))


@pytest.mark.parametrize("buffer_name", ["Metal", "Metal0", "MTL0"])
def test_metal_requires_actual_offload_evidence(runtime, buffer_name):
    runtime.log = runtime.log.replace("MTL0", buffer_name)
    backend = create_judge_backend(JudgeConfig(backend="local", model_path=runtime.path))
    assert backend.acceleration == "auto"
    assert not backend.metadata.is_hardware_accelerated
    assert backend.metadata.status == "ready_to_load"
    result = LocalJudge(backend).refine(runtime.question)
    assert not result.used_fallback
    assert backend.metadata.execution_provider == "llama.cpp/Metal"
    assert backend.metadata.device_type == "Apple Silicon"
    assert backend.metadata.accelerator == "Metal"
    assert backend.metadata.is_hardware_accelerated
    assert runtime.loads[0]["n_gpu_layers"] == -1
    assert runtime.loads[0]["offload_kqv"] is True
    assert runtime.loads[0]["op_offload"] is True
    assert result.source_question == runtime.question


@pytest.mark.parametrize("log", [
    "offloaded 0/37 layers to GPU\nMTL0 model buffer size = 2380.0 MiB",
    "offloaded 37/37 layers to GPU\nCPU model buffer size = 2380.0 MiB",
    "offloaded 37/37 layers to GPU\nMTL0 model buffer size = 0.0 MiB",
    "Unrecognized loader diagnostics",
])
def test_auto_explicitly_reloads_cpu_if_offload_unverified(runtime, log):
    runtime.log = log
    backend = LocalDevelopmentJudgeBackend(runtime.path, "test", acceleration="auto")
    assert not LocalJudge(backend).refine(runtime.question).used_fallback
    assert len(runtime.loads) == 2
    assert runtime.closed == 1
    assert runtime.loads[1]["n_gpu_layers"] == 0
    assert runtime.loads[1]["offload_kqv"] is False
    assert runtime.loads[1]["op_offload"] is False
    assert not backend.metadata.is_hardware_accelerated
    assert backend.metadata.execution_provider == "llama.cpp/CPU"


def test_explicit_metal_without_confirmation_falls_back_to_question(runtime):
    runtime.log = "unrecognized diagnostics"
    backend = LocalDevelopmentJudgeBackend(runtime.path, "test", acceleration="metal")
    result = LocalJudge(backend).refine(runtime.question)
    assert result.used_fallback and result.question == runtime.question.question
    assert runtime.closed == 1 and len(runtime.loads) == 1
    assert not backend.metadata.is_hardware_accelerated
    assert not backend.metadata.available


@pytest.mark.parametrize("system,machine,support", [
    ("Windows", "ARM64", True), ("Darwin", "x86_64", True), ("Darwin", "arm64", False),
])
def test_explicit_metal_unavailable(runtime, monkeypatch, system, machine, support):
    monkeypatch.setattr("stageguide.judge.local_backend.platform.system", lambda: system)
    monkeypatch.setattr("stageguide.judge.local_backend.platform.machine", lambda: machine)
    runtime.module.llama_supports_gpu_offload = lambda: support
    backend = LocalDevelopmentJudgeBackend(runtime.path, "test", acceleration="metal")
    result = LocalJudge(backend).refine(runtime.question)
    assert result.used_fallback
    assert not runtime.loads
    assert not backend.metadata.is_hardware_accelerated
    assert not backend.metadata.available


def test_auto_on_other_platform_uses_cpu(runtime, monkeypatch):
    monkeypatch.setattr("stageguide.judge.local_backend.platform.system", lambda: "Windows")
    backend = LocalDevelopmentJudgeBackend(runtime.path, "test", acceleration="auto")
    assert not LocalJudge(backend).refine(runtime.question).used_fallback
    assert runtime.loads[0]["n_gpu_layers"] == 0
    assert backend.metadata.accelerator == "none"


def test_cpu_overrides_available_gpu(runtime):
    backend = LocalDevelopmentJudgeBackend(runtime.path, "test", acceleration="cpu")
    assert not LocalJudge(backend).refine(runtime.question).used_fallback
    assert runtime.loads[0]["n_gpu_layers"] == 0
    assert runtime.loads[0]["op_offload"] is False
    assert backend.metadata.execution_provider == "llama.cpp/CPU"


def test_metrics_use_runtime_usage_and_clock(runtime, monkeypatch):
    backend = LocalDevelopmentJudgeBackend(runtime.path, "test")
    clock = iter([10.0, 12.0, 15.0])
    monkeypatch.setattr("stageguide.judge.local_backend.perf_counter", lambda: next(clock))
    backend.generate(build_request(runtime.question))
    metrics = backend.last_inference
    assert metrics.total_seconds == 5.0
    assert metrics.load_seconds == 2.0
    assert metrics.generation_seconds == 3.0
    assert metrics.prompt_tokens == 512 and metrics.output_tokens == 64
    assert metrics.decode_tokens_per_second is None
    assert metrics.to_dict()["accelerator"] == "none"


def test_failed_call_clears_previous_metrics(runtime):
    backend = LocalDevelopmentJudgeBackend(runtime.path, "test")
    judge = LocalJudge(backend)
    assert not judge.refine(runtime.question).used_fallback
    assert backend.last_inference is not None
    runtime.error = RuntimeError("test inference failure")
    result = judge.refine(runtime.question)
    assert result.used_fallback and result.question == runtime.question.question
    assert backend.last_inference is None
    assert backend.metadata.status == "inference_failed"


@pytest.mark.parametrize("diagnostics_fail", [False, True])
def test_native_decode_metrics_optional(runtime, diagnostics_fail):
    backend = LocalDevelopmentJudgeBackend(runtime.path, "test")
    model = backend._load()
    model._ctx = SimpleNamespace(ctx=object())
    runtime.module.llama_perf_context_reset = lambda _: None

    def perf(_):
        if diagnostics_fail:
            raise RuntimeError("diagnostics unavailable")
        return SimpleNamespace(n_eval=50, t_eval_ms=2000)

    runtime.module.llama_perf_context = perf
    assert not LocalJudge(backend).refine(runtime.question).used_fallback
    assert backend.last_inference.decode_tokens_per_second == (None if diagnostics_fail else 25.0)


def test_diagnostic_reset_failure_does_not_break_generation(runtime):
    backend = LocalDevelopmentJudgeBackend(runtime.path, "test")
    backend._load()._ctx = SimpleNamespace(ctx=object())

    def fail(_):
        raise RuntimeError("diagnostics unavailable")

    runtime.module.llama_perf_context_reset = fail
    assert not LocalJudge(backend).refine(runtime.question).used_fallback
    assert backend.last_inference.decode_tokens_per_second is None
