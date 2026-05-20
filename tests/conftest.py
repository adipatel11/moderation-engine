import pytest

from moderation_engine.config import settings
from moderation_engine.model import ToxicityClassifier


@pytest.fixture(scope="session")
def classifier() -> ToxicityClassifier:
    return ToxicityClassifier(settings.model_name)
