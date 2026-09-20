import pytest
from src.model import SentimentAnalyzer


def test_sentiment_analyzer_init():
    """Verifica che l'analizzatore di sentiment si inizializzi correttamente."""
    analyzer = SentimentAnalyzer()
    assert analyzer.model is not None
    assert analyzer.tokenizer is not None


def test_predict_single():
    """Verifica che la predizione restituisca la struttura e le chiavi attese."""
    analyzer = SentimentAnalyzer()
    sample_text = "This is a great test sentence!"
    result = analyzer.predict_single(sample_text)

    assert isinstance(result, dict)
    assert "text" in result
    assert "label" in result
    assert "confidence" in result
    assert "scores" in result
    assert result["label"] in ["positive", "neutral", "negative"]
    assert 0.0 <= result["confidence"] <= 1.0
