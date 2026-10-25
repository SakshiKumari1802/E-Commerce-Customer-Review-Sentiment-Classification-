import html
import re

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def clean_review(text: str) -> str:
    """Normalize markup and whitespace while retaining sentiment-bearing words."""
    text = html.unescape(str(text))
    text = _URL.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()
