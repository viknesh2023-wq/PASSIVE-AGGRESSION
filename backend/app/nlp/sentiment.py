"""
Sentiment features using VADER (NLTK).
Provides polarity scores and contrast detection.
VADER is specifically designed for social media / short texts.
"""

from __future__ import annotations

_vader = None  # lazy load


def _get_vader():
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except ImportError:
            _vader = None
    return _vader


def get_sentiment_scores(text: str) -> dict[str, float]:
    """
    Returns VADER compound score and component scores.
    compound in [-1, 1]: positive = +, negative = -
    """
    analyzer = _get_vader()
    if analyzer is None:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "available": 0.0}
    scores = analyzer.polarity_scores(text)
    scores["available"] = 1.0
    return scores


def get_sentence_sentiments(sentences: list[str]) -> list[dict[str, float]]:
    """Per-sentence VADER scores."""
    return [get_sentiment_scores(s) for s in sentences]


def detect_polarity_contrast(sentences: list[str]) -> float:
    """
    Detect a positive-framing / negative-implication contrast pattern.
    Returns a contrast score in [0, 1] — higher means stronger contrast.

    Example: "Great job! [Positive] I can't believe it took this long. [Negative]"
    This is a weak signal for backhanded compliments.
    """
    if len(sentences) < 2:
        return 0.0
    sentiments = get_sentence_sentiments(sentences)
    compounds  = [s["compound"] for s in sentiments]
    if not compounds:
        return 0.0
    # Contrast = std dev of compound scores across sentences
    import statistics
    if len(compounds) < 2:
        return 0.0
    return min(1.0, statistics.stdev(compounds))


def politeness_contrast_score(text: str) -> float:
    """
    Detect texts where positive polite words mask a negative intent.
    High score = polite words + negative compound = mock_politeness indicator.
    This is a heuristic, not a trained model.
    """
    analyzer = _get_vader()
    if analyzer is None:
        return 0.0
    scores  = analyzer.polarity_scores(text)
    lower   = text.lower()
    # Polite words that co-occur with negative sentiment are suspicious
    polite_markers = [
        "please", "thank you", "thanks", "kindly", "respectfully",
        "appreciate", "sorry", "apologies", "of course", "certainly",
    ]
    polite_count = sum(1 for m in polite_markers if m in lower)
    if polite_count > 0 and scores["compound"] < -0.1:
        return min(1.0, polite_count * 0.2 * abs(scores["compound"]))
    return 0.0
