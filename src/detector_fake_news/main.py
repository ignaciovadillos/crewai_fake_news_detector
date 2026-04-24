"""CLI entrypoint for a simple article analysis run."""

from __future__ import annotations

from detector_fake_news.service import analyze_article

EXAMPLE_TITLE = "City officials deny claim about emergency water contamination"
EXAMPLE_ARTICLE = """
Several social media posts claimed that a city's drinking water was contaminated
after an industrial spill and that officials told residents to boil all tap
water immediately. The same posts also stated that two hospitals had already
reported dozens of poisoning cases and that the contamination was confirmed by a
federal agency. Later in the article, unnamed residents accuse local authorities
of hiding the truth, while no direct quotations from health authorities are
provided.
""".strip()


def run() -> None:
    report = analyze_article(
        title=EXAMPLE_TITLE,
        article_text=EXAMPLE_ARTICLE,
    )
    print("\n=== FINAL VERDICT ===\n")
    print(report.final_verdict.model_dump_json(indent=2))


if __name__ == "__main__":
    run()
