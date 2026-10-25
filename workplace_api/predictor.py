"""
workplace_api/predictor.py — Standalone predictor for the workplace passive-aggression model.
Reuses the trained Pipeline artifact from abijitabi47/passive-aggression-detector.
Loads the pipeline once into memory at startup.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "passive_aggression_pipeline.joblib"


def preprocess_text(text: str) -> str:
    """Conservative normalization: case-folding and whitespace trimming while preserving punctuation."""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


class WorkplacePredictor:
    """Singleton service for loading and executing the workplace passive aggression model."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_PATH
        self.pipeline = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.is_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                self.is_loaded = True
            except Exception as e:
                self.is_loaded = False
                print(f"[WorkplacePredictor] Error loading model from {self.model_path}: {e}")
        else:
            self.is_loaded = False
            print(f"[WorkplacePredictor] Model file not found at {self.model_path}")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Execute prediction on single statement.
        Returns genuine model classification, calibrated probability, VADER sentiment, and weighted cues.
        """
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError(
                f"Workplace model is not loaded. Ensure {self.model_path} exists."
            )

        cleaned = preprocess_text(text)
        prediction = int(self.pipeline.predict([cleaned])[0])
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        classes = list(self.pipeline.classes_)
        predicted_index = classes.index(prediction)
        score = float(probabilities[predicted_index])

        # VADER Supporting Sentiment
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        compound = float(sentiment["compound"])
        sentiment_label = (
            "Positive" if compound >= 0.05
            else ("Negative" if compound <= -0.05 else "Neutral")
        )

        # Feature Cues from linear model coefficients
        cues: List[str] = []
        try:
            vectorizer = self.pipeline.named_steps.get("tfidf")
            classifier = self.pipeline.named_steps.get("classifier")
            if vectorizer and classifier and hasattr(classifier, "coef_"):
                vector = vectorizer.transform([cleaned])
                feature_names = vectorizer.get_feature_names_out()
                row = vector.tocoo()
                contributions = []
                for idx, value in zip(row.col, row.data):
                    contribution = float(value * classifier.coef_[0][idx])
                    if prediction == int(classifier.classes_[0]):
                        contribution *= -1
                    if contribution > 0:
                        contributions.append((contribution, feature_names[idx]))
                cues = [term for _, term in sorted(contributions, reverse=True)[:5]]
        except Exception as e:
            # Fallback if cue extraction encounters unexpected sparse vector
            cues = []

        raw_label = "Potentially passive-aggressive" if prediction == 1 else "Not flagged as passive-aggressive"

        return {
            "prediction": prediction,
            "label": raw_label,
            "score": score,
            "sentiment_label": sentiment_label,
            "sentiment_score": compound,
            "cues": cues,
        }


# Global singleton instance
predictor = WorkplacePredictor()
