"""Explicitly opt in with STAGEGUIDE_RUN_LOCAL_JUDGE=1 and a local GGUF path."""

import os
from importlib.util import find_spec
from pathlib import Path

import pytest

from stageguide.judge import JudgeConfig, LocalJudge, create_judge_backend
from stageguide.judge.demo import sample_questions


@pytest.mark.local_model
def test_real_local_refinement_and_follow_up():
    if os.environ.get("STAGEGUIDE_RUN_LOCAL_JUDGE") != "1":
        pytest.skip("Real local inference is opt-in; normal tests never download models")
    path = os.environ.get("STAGEGUIDE_JUDGE_MODEL")
    if not path or not Path(path).is_file() or find_spec("llama_cpp") is None:
        pytest.skip("Supply an existing GGUF and install the optional judge-local runtime")
    backend = create_judge_backend(JudgeConfig(backend="local", model_path=path))
    question = sample_questions()[0]
    result = LocalJudge(backend).interact(
        question, "We expect that because AI automates some preparation work.",
    )
    assert not result.refined_question.used_fallback, result.refined_question.fallback_reason
    assert result.follow_up is not None
    assert not result.follow_up.used_fallback, result.follow_up.fallback_reason
    assert result.refined_question.source_question == question
    assert result.follow_up.source_question == question
    assert backend.metadata.status == "loaded"
