"""JSON demo; defaults to disabled inference and never downloads a model."""

import argparse
import json
import platform

from .demo import sample_questions
from .factory import JudgeConfig, create_judge_backend
from .interaction_models import JudgeInteraction
from .service import LocalJudge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("disabled", "local", "qualcomm"), default="disabled")
    parser.add_argument("--model-path", help="Existing local GGUF file; no downloads")
    parser.add_argument("--model", default=JudgeConfig().model, help="Model identity label")
    parser.add_argument("--acceleration", choices=("auto", "cpu", "metal"), default="auto")
    args = parser.parse_args()
    backend = create_judge_backend(JudgeConfig(args.backend, args.model, args.model_path,
                                             acceleration=args.acceleration))
    question = sample_questions()[0]
    answer = "We expect that because AI automates some preparation work."
    judge = LocalJudge(backend)
    refined = judge.refine(question)
    refine_metrics = getattr(backend, "last_inference", None)
    follow_up = judge.follow_up(question, answer)
    follow_up_metrics = getattr(backend, "last_inference", None)
    interaction = JudgeInteraction(question, refined, answer, follow_up)
    disabled = LocalJudge().refine(question)
    print(json.dumps({
        "platform": {"operating_system": platform.system(), "machine_architecture": platform.machine()},
        "backend_metadata": backend.metadata.to_dict(),
        "interaction": interaction.to_dict(),
        "disabled_fallback": disabled.to_dict(),
        "inference_metrics": {
            "refine": refine_metrics.to_dict() if refine_metrics is not None else None,
            "follow_up": follow_up_metrics.to_dict() if follow_up_metrics is not None else None,
        },
        "runtime_initialization_log": getattr(backend, "initialization_log", None),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
