"""PhishProof -- trustworthy, explainable, selective phishing detection.

The public repository ships the full package structure and interface of the
PhishProof pipeline; the executable implementation of each component is released
here once the paper is accepted (see :data:`RELEASE_NOTICE`). Importing the
package and inspecting the API works; calling into a withheld component prints
the notice and raises :class:`NotImplementedError`.
"""

from ._release import RELEASE_NOTICE
from .config import PhishProofConfig
from .pipeline import PhishProofRunner
from .types import PageRecord, Prediction

__all__ = [
    "PageRecord",
    "PhishProofConfig",
    "PhishProofRunner",
    "Prediction",
    "RELEASE_NOTICE",
]
__version__ = "1.1.0"
