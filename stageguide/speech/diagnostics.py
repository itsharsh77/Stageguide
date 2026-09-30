"""Read-only platform/dependency discovery; no inference or hardware probing."""

from importlib.util import find_spec
import platform
from typing import Any, Dict


def _discoverable(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def platform_diagnostics() -> Dict[str, Any]:
    """Report candidates based on module discovery, not verified runtime support.

    A discoverable package may still fail to load (ABI, DLL or compute support).
    No SDK, model, driver or accelerator is loaded or installed here.
    """
    required = ("faster_whisper", "ctranslate2", "av", "onnxruntime", "tokenizers", "numpy")
    missing = [name for name in required if not _discoverable(name)]
    return {
        "operating_system": platform.system(),
        "machine_architecture": platform.machine(),
        "available_execution_backends": ["cpu_whisper"] if not missing else [],
        "availability_basis": "Implementation exists and required Python modules are discoverable; runtime and model not verified",
        "backends": {
            "cpu_whisper": {
                "implemented": True,
                "dependencies_detected": not missing,
                "missing_dependencies": missing,
                "runtime_verified": False,
            },
            "qnn_whisper": {
                "implemented": False,
                "status": "planned",
                "runtime_verified": False,
                "reason": "Qualcomm QNN integration is not implemented or tested, including on Windows ARM64",
            },
        },
    }
