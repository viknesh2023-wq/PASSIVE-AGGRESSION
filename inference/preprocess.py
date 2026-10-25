"""
inference/preprocess.py
Shared text preprocessing for standalone inference (no FastAPI dependency).
"""
from __future__ import annotations
import re
import unicodedata


def validate_english(text: str) -> None:
    """Raise ValueError if text is predominantly non-Latin script."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return
    non_latin = 0
    for ch in letters:
        name = unicodedata.name(ch, "")
        if any(name.startswith(s) for s in (
            "CYRILLIC", "ARABIC", "DEVANAGARI", "TAMIL", "TELUGU", "KANNADA",
            "MALAYALAM", "BENGALI", "GUJARATI", "CJK", "HANGUL",
            "HIRAGANA", "KATAKANA", "THAI", "GREEK", "HEBREW",
        )):
            non_latin += 1
    if non_latin / len(letters) > 0.25:
        raise ValueError(
            "This tool analyses English text only. "
            "Please submit a message in English."
        )


def normalize(text: str) -> str:
    """Light normalization preserving original structure."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def model_ready(text: str) -> str:
    """Produce lowercase, URL/mention-stripped text for model input."""
    t = normalize(text)
    t = re.sub(r"https?://\S+", "[URL]", t)
    t = re.sub(r"@\w+", "@user", t)
    return t.lower()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]
