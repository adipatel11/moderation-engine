"""Public model API.

Re-exports the backend-agnostic protocol and the factory so callers don't
have to know about the `backends.*` layout.
"""

from .backends import ToxicityClassifier, build_classifier

__all__ = ["ToxicityClassifier", "build_classifier"]
