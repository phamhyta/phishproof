"""PhishProof: selective phishing detection with verified evidence."""

from .pipeline import PhishProof, build_evidence
from .policy import Action, Decision, FrozenPolicy, decide, fit_policy
from .replay import apply_policy, read_evidence, write_evidence
from .schema import Cue, CueType, Label, MemberOutput, PageEvidence, PageRecord

__version__ = "0.2.0"

__all__ = [
    "Action",
    "Cue",
    "CueType",
    "Decision",
    "FrozenPolicy",
    "Label",
    "MemberOutput",
    "PageEvidence",
    "PageRecord",
    "PhishProof",
    "apply_policy",
    "build_evidence",
    "decide",
    "fit_policy",
    "read_evidence",
    "write_evidence",
]
