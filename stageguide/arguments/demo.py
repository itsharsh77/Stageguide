"""Run the requested synthetic example with python -m stageguide.arguments.demo."""

import json

from stageguide.alignment import align_slide
from stageguide.presentation.models import PresentationPage

from .analyzer import analyze_slide


def main() -> None:
    slide = PresentationPage(
        page_number=1, title=None,
        body_text=("StageGuide reduces presentation preparation time by 30%.\n\n"
                   "Our initial target market is 5 million university students.\n\n"
                   "Universities will pay ₹999 per user each year."),
        numbers=["30", "5", "999"], percentages=["30%"], source_filename="argument_demo.pptx",
    )
    transcript = ("StageGuide is designed for university students preparing presentations.\n"
                  "It helps them practise their delivery and argument before presenting.")
    result = analyze_slide(slide, transcript, align_slide(slide, transcript))
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
