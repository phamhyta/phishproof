"""Public scaffold for the PhishProof package."""

from .pipeline import PhishProofRunner
from .types import PageRecord, Prediction

__all__ = ["PageRecord", "PhishProofRunner", "Prediction"]
__version__ = "0.1.0"
