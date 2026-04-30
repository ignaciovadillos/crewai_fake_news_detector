"""Fake news detector package."""

__all__ = ["analyze_article"]


def __getattr__(name: str):
    if name == "analyze_article":
        from detector_fake_news.service import analyze_article

        return analyze_article
    raise AttributeError(f"module 'detector_fake_news' has no attribute {name!r}")
