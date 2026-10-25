"""
Text preprocessing pipeline.
Preserves original text for highlighting; produces a cleaned version for model input.
"""

from __future__ import annotations
import re
import unicodedata


def validate_english_language(text: str) -> None:
    """
    Validates that the input text is in English.
    Strict project-wide English-only requirement:
    - Allows Latin alphabetic characters, standard punctuation, numbers, spaces, and emojis.
    - Permits loanwords with Latin accents (e.g. café, résumé).
    - If non-Latin script characters (e.g., Cyrillic, Arabic, Devanagari, Tamil, CJK, etc.)
      form a substantial portion of alphabetic tokens, raises ValueError.
    - Does NOT reject valid English text containing slang, abbreviations, code, or emojis.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return
    
    non_latin_count = 0
    for ch in letters:
        try:
            name = unicodedata.name(ch, "")
            if any(name.startswith(script) for script in (
                "CYRILLIC", "ARABIC", "DEVANAGARI", "TAMIL", "TELUGU", "KANNADA", 
                "MALAYALAM", "BENGALI", "GUJARATI", "GURMUKHI", "CJK", "HANGUL", 
                "HIRAGANA", "KATAKANA", "THAI", "GREEK", "HEBREW"
            )):
                non_latin_count += 1
        except Exception:
            pass
            
    ratio = non_latin_count / len(letters)
    if ratio > 0.25:
        raise ValueError(
            "This platform strictly supports English text analysis. "
            "Detected unsupported non-English script. Please submit English text."
        )


def normalize_text(text: str) -> str:
    """
    Light normalization:
    - Unicode NFC normalization
    - Collapse runs of whitespace (not newlines — keep sentence structure)
    - Strip leading/trailing whitespace
    Does NOT lowercase, strip punctuation, or alter capitalization.
    Original text is preserved for offset-based highlighting.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\r\n', '\n', text)            # normalize Windows line endings
    text = re.sub(r'[ \t]+', ' ', text)          # collapse horizontal whitespace
    text = re.sub(r' ?\n ?', '\n', text)          # strip spaces adjacent to newlines
    text = text.strip()
    return text


def split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter that preserves original text.
    Splits on .!? followed by whitespace or end of string.
    Returns non-empty sentence strings.
    """
    # Split on sentence-ending punctuation followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if s.strip()]


def split_sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """
    Returns list of (sentence_text, start_offset, end_offset) tuples.
    Offsets reference original `text`.
    """
    result: list[tuple[str, int, int]] = []
    cursor = 0
    for sent in split_sentences(text):
        idx = text.find(sent, cursor)
        if idx == -1:
            idx = cursor
        result.append((sent, idx, idx + len(sent)))
        cursor = idx + len(sent)
    return result


def model_ready_text(text: str) -> str:
    """
    Produce a model-ready version of the text.
    Does NOT destroy the original — call normalize_text first, keep that.
    This version: lowercased, URLs replaced, usernames replaced.
    """
    t = normalize_text(text)
    t = re.sub(r'https?://\S+', '[URL]', t)
    t = re.sub(r'@\w+', '@user', t)
    return t.lower()


def compute_text_stats(text: str) -> dict[str, int | float]:
    """
    Compute basic text statistics used as features.
    """
    words = text.split()
    sentences = split_sentences(text)
    return {
        "char_count":     len(text),
        "word_count":     len(words),
        "sentence_count": max(1, len(sentences)),
        "avg_word_len":   (sum(len(w) for w in words) / max(1, len(words))),
        "avg_sent_len":   (len(words) / max(1, len(sentences))),
    }
