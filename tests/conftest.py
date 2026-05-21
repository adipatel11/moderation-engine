import pytest

from moderation_engine.config import settings
from moderation_engine.model import ToxicityClassifier, build_classifier


@pytest.fixture(scope="session")
def classifier() -> ToxicityClassifier:
    return build_classifier(settings)
