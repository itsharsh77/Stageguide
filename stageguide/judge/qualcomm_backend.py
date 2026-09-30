"""Future Windows Snapdragon adapter. No Qualcomm execution on any platform yet."""

from .backend import JudgeBackendError, JudgeBackendMetadata, JudgeRequest


class QualcommJudgeBackend:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @property
    def metadata(self) -> JudgeBackendMetadata:
        return JudgeBackendMetadata(
            "QualcommJudgeBackend", self.model_name, "unavailable", "unavailable",
            "none", False, False, "not_implemented",
        )

    def generate(self, request: JudgeRequest) -> str:
        raise JudgeBackendError(
            "Qualcomm judge inference is planned, not implemented or tested. "
            "The target is Windows ARM64/Snapdragon using GenieX/QAIRT. No NPU inference was run."
        )
