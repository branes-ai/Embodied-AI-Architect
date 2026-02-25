"""Agent implementations for the Embodied AI Architect system."""

from .base import BaseAgent  # noqa: F401
from .model_analyzer import ModelAnalyzerAgent  # noqa: F401
from .benchmark import BenchmarkAgent  # noqa: F401
from .hardware_profile import HardwareProfileAgent  # noqa: F401
from .report_synthesis import ReportSynthesisAgent  # noqa: F401

_exports = [
    "BaseAgent",
    "ModelAnalyzerAgent",
    "BenchmarkAgent",
    "HardwareProfileAgent",
    "ReportSynthesisAgent",
]

# Deployment agent (optional: tensorrt)
try:
    from .deployment import DeploymentAgent  # noqa: F401

    _exports.append("DeploymentAgent")
except ImportError:
    pass

__all__ = _exports
