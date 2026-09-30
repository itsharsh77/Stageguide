"""Run with python -m stageguide.alignment.demo; no files or models needed."""

import json

from stageguide.presentation.models import PresentationPage

from .engine import align_slide


def main() -> None:
    slide = PresentationPage(
        page_number=1, title=None,
        body_text=("StageGuide reduces presentation preparation time by 30%.\n"
                   "Our initial target is 5 million university students."),
        numbers=["30", "5"], percentages=["30%"], source_filename="alignment_demo.pptx",
    )
    transcript = ("Our first users will be university students. "
                  "We want StageGuide to make presentation preparation easier.")
    print(json.dumps(align_slide(slide, transcript).to_dict(), indent=2))


if __name__ == "__main__":
    main()
