from moderation_engine.model import ToxicityClassifier


def test_clearly_toxic_input_flagged(classifier: ToxicityClassifier) -> None:
    result = classifier.predict("You are an idiot.")
    assert result["toxic"] > 0.9, f"expected toxic > 0.9, got {result['toxic']:.3f}"
    assert result["insult"] > 0.9, f"expected insult > 0.9, got {result['insult']:.3f}"


def test_clean_input_not_flagged(classifier: ToxicityClassifier) -> None:
    result = classifier.predict("Have a wonderful day!")
    assert result["toxic"] < 0.1, f"expected toxic < 0.1, got {result['toxic']:.3f}"


def test_predict_returns_all_six_labels(classifier: ToxicityClassifier) -> None:
    result = classifier.predict("anything")
    assert set(result.keys()) == {
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    }
