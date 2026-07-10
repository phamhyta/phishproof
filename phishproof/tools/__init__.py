"""Grounding verifiers: re-derive each agreed cue from the page."""

from .base import ToolRegistry, Verifier
from .brands import HtmlBrandDetector
from .cert import CertificateVerifier
from .dom import FormActionVerifier
from .logo_brand import LogoBrandMatcher
from .redirect import RedirectVerifier


def default_registry() -> ToolRegistry:
    """Register the verifiers enabled in the deployed configuration."""
    registry = ToolRegistry()
    for tool in (
        HtmlBrandDetector(),
        FormActionVerifier(),
        LogoBrandMatcher(),
        CertificateVerifier(),
        RedirectVerifier(),
    ):
        registry.register(tool)
    return registry


__all__ = [
    "CertificateVerifier",
    "FormActionVerifier",
    "HtmlBrandDetector",
    "LogoBrandMatcher",
    "RedirectVerifier",
    "ToolRegistry",
    "Verifier",
    "default_registry",
]
