"""Runtime-neutral local generation contract. No inference library imports."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Protocol


class JudgeBackendError(RuntimeError):
    """Local generation failed or the selected backend is unavailable."""


@dataclass(frozen=True)
class JudgeInferenceMetrics:
    """Optional diagnostics, independent of question grounding or scoring.

    Total time includes lazy model loading. Decode throughput excludes loading,
    prompt evaluation and JSON validation; None means the runtime did not report it.
    Token counts include the complete JSON response, not just the question text.
    """

    model_name: str
    backend_name: str
    execution_provider: str
    accelerator: str
    total_seconds: float
    load_seconds: float
    generation_seconds: float
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]
    decode_tokens_per_second: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JudgeBackendMetadata:
    backend_name: str
    model_name: str
    execution_provider: str
    device_type: str
    accelerator: str
    is_hardware_accelerated: bool
    available: bool
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JudgeRequest:
    system_prompt: str
    context_json: str
    response_schema: Dict[str, Any]
    temperature: float = 0.0
    max_tokens: int = 384
    seed: int = 0


class JudgeBackend(Protocol):
    @property
    def metadata(self) -> JudgeBackendMetadata:
        ...

    def generate(self, request: JudgeRequest) -> str:
        """Return one JSON object as text, or raise; never download weights."""
        ...


class DisabledJudgeBackend:
    @property
    def metadata(self) -> JudgeBackendMetadata:
        return JudgeBackendMetadata(
            "DisabledJudgeBackend", "none", "none", "none", "none", False, False, "disabled",
        )

    def generate(self, request: JudgeRequest) -> str:
        raise JudgeBackendError("Local LLM is disabled")
