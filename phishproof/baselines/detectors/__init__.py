"""The dedicated detectors D1-D5.

D1-D3 wrap released code (``released``). D4 and D5 are re-implementations from their papers
(``knowphish`` and ``phishagent``) and are imported directly, since they need optional packages.
"""

from .base import Decision, Detector, DetectorResult, Page

__all__ = ["Decision", "Detector", "DetectorResult", "Page"]
