"""
Tests for NLP preprocessing, phrase matching, feature extraction, and rewriter.
"""

import pytest
from backend.app.nlp.preprocess import normalize_text, split_sentences, split_sentences_with_offsets, model_ready_text, validate_english_language
from backend.app.nlp.phrase_matcher import find_phrase_matches, compute_style_signals
from backend.app.nlp.sentiment import get_sentiment_scores
from backend.app.nlp.features import extract_all_features
from backend.app.nlp.rewriter import generate_rewrites


def test_validate_english_language():
    # Valid English messages with slang, emojis, etc.
    validate_english_language("Hey bro, this presentation is lit fr fr 💀😂")
    validate_english_language("As per my last email, please confirm.")
    validate_english_language("café résumé touché")
    
    # Non-English messages must be rejected
    with pytest.raises(ValueError, match="strictly supports English"):
        validate_english_language("Привет, как дела? Это русский текст.")
    with pytest.raises(ValueError, match="strictly supports English"):
        validate_english_language("你好，这是一个完全由中文字符构成的句子。")
    with pytest.raises(ValueError, match="strictly supports English"):
        validate_english_language("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?")


def test_normalize_text():
    raw = "  Hello   world!  \r\nThis is a test.  "
    norm = normalize_text(raw)
    assert norm == "Hello world!\nThis is a test."
    assert "  " not in norm


def test_split_sentences():
    text = "As per my last email, the deadline was Friday. Please review it! Is that clear?"
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "As per my last email, the deadline was Friday."
    assert sents[1] == "Please review it!"
    assert sents[2] == "Is that clear?"


def test_split_sentences_with_offsets():
    text = "First sentence. Second sentence!"
    sents = split_sentences_with_offsets(text)
    assert len(sents) == 2
    for s, start, end in sents:
        assert text[start:end] == s


def test_phrase_matcher_detection():
    text = "As per my last email, the deadline was Friday. No worries, whatever you say."
    matches = find_phrase_matches(text)
    tactics = [m.tactic for m in matches]
    assert "blame_shifting" in tactics or "mock_politeness" in tactics
    assert len(matches) >= 2
    for m in matches:
        assert text[m.start:m.end].lower() == m.text.lower()


def test_style_signals():
    text = "WHY ARE YOU IGNORING THIS... ???"
    signals = compute_style_signals(text)
    assert signals["caps_word_count"] >= 4
    assert signals["ellipsis_count"] >= 1


def test_sentiment_scores():
    pos_res = get_sentiment_scores("Great job on delivering the presentation, well done!")
    assert pos_res["compound"] > 0
    neg_res = get_sentiment_scores("I hate working with you, you are useless.")
    assert neg_res["compound"] < 0


def test_feature_extraction():
    text = "Per my last email, thanks in advance for finally doing this."
    feats = extract_all_features(text)
    assert "char_count" in feats
    assert "word_count" in feats
    assert "vader_compound" in feats
    assert feats["word_count"] > 0


def test_rewriter():
    text = "As per my last email, the deadline was Friday. Whatever you say."
    rewrites = generate_rewrites(text)
    assert len(rewrites) >= 1
    for r in rewrites:
        assert len(r) > 0
        assert r != text
