"""Optional in-process llama.cpp adapter. Only this file imports llama_cpp."""

from importlib.util import find_spec
from contextlib import redirect_stderr
import io
import logging
from pathlib import Path
import platform
import re
from threading import RLock
from time import perf_counter
from typing import Optional, Union

from .backend import JudgeBackendError, JudgeBackendMetadata, JudgeInferenceMetrics, JudgeRequest


# llama.cpp logging is process-global and its contexts are not thread-safe.
_RUNTIME_LOCK = RLock()


class LocalDevelopmentJudgeBackend:
    """Local GGUF on CPU or verified Apple Metal; never downloads weights.

    model_name is a configuration label, not weight identity verification.
    ready_to_load reports prerequisites only, not successful model execution.
    """

    def __init__(self, model_path: Optional[Union[str, Path]], model_name: str,
                 context_size: int = 4096, *, acceleration: str = "cpu") -> None:
        if isinstance(context_size, bool) or not isinstance(context_size, int) or context_size < 1024:
            raise ValueError("context_size must be an integer of at least 1024")
        if acceleration not in {"cpu", "auto", "metal"}:
            raise ValueError("acceleration must be cpu, auto or metal")
        self.model_path = Path(model_path) if model_path is not None else None
        self.model_name = model_name
        self.context_size = context_size
        self._model = None
        self._failed = False
        self.acceleration = acceleration
        self._metal_active = False
        self.initialization_log = ""
        self.last_inference: Optional[JudgeInferenceMetrics] = None

    @property
    def metadata(self) -> JudgeBackendMetadata:
        if self._failed:
            status = "inference_failed"
        elif self._model is not None:
            status = "loaded"
        elif (self.model_path is None or self.model_path.suffix.lower() != ".gguf"
              or not self.model_path.is_file()):
            status = "model_missing"
        elif find_spec("llama_cpp") is None:
            status = "runtime_missing"
        else:
            status = "ready_to_load"
        return JudgeBackendMetadata(
            "LocalDevelopmentJudgeBackend", self.model_name,
            "llama.cpp/Metal" if self._metal_active else "llama.cpp/CPU",
            "Apple Silicon" if self._metal_active else "CPU",
            "Metal" if self._metal_active else "none", self._metal_active,
            status in {"loaded", "ready_to_load"}, status,
        )

    def _load(self):
        if self._model is not None:
            return self._model
        if self.model_path is None or self.model_path.suffix.lower() != ".gguf" or not self.model_path.is_file():
            raise JudgeBackendError("Provide an existing local .gguf file; automatic downloads are disabled")
        try:
            import llama_cpp
        except ImportError as exc:
            raise JudgeBackendError("Optional runtime missing; install stageguide[judge-local]") from exc
        apple = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
        metal = (self.acceleration != "cpu" and apple
                 and llama_cpp.llama_supports_gpu_offload())
        if self.acceleration == "metal" and not metal:
            self._failed = True
            raise JudgeBackendError("Metal requires Apple Silicon and a Metal-enabled llama.cpp runtime")
        options = dict(model_path=str(self.model_path.resolve()), n_ctx=self.context_size,
                       n_threads=4, seed=0)
        if metal:
            # Capture native loader diagnostics through llama-cpp-python's logging
            # callback. A requested device alone is not proof of layer offload.
            log = io.StringIO()
            logger = logging.getLogger("llama-cpp-python")
            previous_level = logger.level
            try:
                with redirect_stderr(log):
                    model = llama_cpp.Llama(**options, n_gpu_layers=-1,
                                           offload_kqv=True, op_offload=True, verbose=True)
            finally:
                self.initialization_log = log.getvalue()
                logger.setLevel(previous_level)
            offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", self.initialization_log)
            metal_buffer = re.search(r"(?:Metal|MTL)\S*\s+model buffer size\s*=\s*([\d.]+)", self.initialization_log)
            self._metal_active = bool(offload and int(offload[1]) > 0
                                      and metal_buffer and float(metal_buffer[1]) > 0)
            if self._metal_active:
                model.verbose = False
                self._model = model
            else:
                model.close()
                if self.acceleration == "metal":
                    self._failed = True
                    raise JudgeBackendError("Runtime did not confirm Metal model layer offload")
        if self._model is None:
            self._model = llama_cpp.Llama(**options, n_gpu_layers=0, offload_kqv=False,
                                         op_offload=False, verbose=False)
        return self._model

    def generate(self, request: JudgeRequest) -> str:
        with _RUNTIME_LOCK:
            return self._generate_locked(request)

    def _generate_locked(self, request: JudgeRequest) -> str:
        self.last_inference = None
        started = perf_counter()
        try:
            model = self._load()
            loaded = perf_counter()
            # Optional native diagnostics must never affect question availability.
            native = None
            try:
                import llama_cpp
                llama_cpp.llama_perf_context_reset(model._ctx.ctx)
                native = llama_cpp
            except Exception:
                pass
            result = model.create_chat_completion(
                messages=[{"role": "system", "content": request.system_prompt},
                          {"role": "user", "content": request.context_json}],
                response_format={"type": "json_object", "schema": request.response_schema},
                temperature=request.temperature, seed=request.seed, max_tokens=request.max_tokens,
                stream=False,
            )
            finished = perf_counter()
            decode_rate = None
            if native is not None:
                try:
                    perf = native.llama_perf_context(model._ctx.ctx)
                    if perf.n_eval > 0 and perf.t_eval_ms > 0:
                        decode_rate = perf.n_eval * 1000 / perf.t_eval_ms
                except Exception:
                    pass  # Diagnostic failure does not invalidate a usable response.
            usage = result.get("usage") or {}
            info = self.metadata
            self.last_inference = JudgeInferenceMetrics(
                self.model_name, info.backend_name, info.execution_provider, info.accelerator,
                finished - started, loaded - started, finished - loaded,
                usage.get("prompt_tokens"), usage.get("completion_tokens"), decode_rate,
            )
            choice = result["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise JudgeBackendError("Local generation did not finish within its output limit")
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise JudgeBackendError("Local model returned no JSON text")
            self._failed = False
            return content
        except JudgeBackendError:
            raise
        except Exception as exc:
            self._failed = True
            raise JudgeBackendError("Local model loading or inference failed") from exc
