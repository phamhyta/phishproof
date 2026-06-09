"""PhishProof — Grounded Evidence-Agreement (GEA = A·G) selective phishing detector.

Package layout (see IMPLEMENTATION_PLAN.md):
    schema/       typed cue schema + data models (this is the central contract)
    agents/       evidence agents + panel (Phase 3)
    tools/        grounding tools: dom, cert, redirect, logo, detectors (Phase 2)
    aggregate/    normalize -> consensus -> agreement(A) -> grounding(G) -> gea (Phase 3)
    calibration/  isotonic kappa + selective decision (Phase 4)
    baselines/    B1-B6 reliability baselines (Phase 4)
    eval/         metrics: AURC, SelAcc80, FPR80, Cov99, ECE, flip-rate (Phase 5)
"""

__version__ = "0.1.0"
