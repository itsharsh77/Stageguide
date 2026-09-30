"""Compose existing analysis and planning: python -m stageguide.judge.demo."""

import json
from typing import List

from stageguide.arguments import analyze_slide
from stageguide.presentation.models import PresentationPage

from .models import JudgeQuestion
from .planner import plan_questions


def sample_questions() -> List[JudgeQuestion]:
    slide = PresentationPage(
        page_number=1, title=None,
        body_text=("StageGuide reduces presentation preparation time by 30%.\n\n"
                   "Our initial target market is 5 million university students.\n\n"
                   "Universities will pay ₹999 per user each year."),
        numbers=["30", "5", "999"], percentages=["30%"], source_filename="argument_demo.pptx",
    )
    transcript = ("StageGuide is designed for university students preparing presentations.\n"
                  "It helps them practise their delivery and argument before presenting.")
    return plan_questions([analyze_slide(slide, transcript)])


def main() -> None:
    print(json.dumps([question.to_dict() for question in sample_questions()], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
