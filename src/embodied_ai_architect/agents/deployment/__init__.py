"""Deployment agent for edge/embedded model deployment."""

from .agent import DeploymentAgent  # noqa: F401
from .models import (  # noqa: F401
    CalibrationConfig,
    DeploymentArtifact,
    DeploymentPrecision,
    DeploymentResult,
    DeploymentStatus,
    ValidationConfig,
    ValidationResult,
)

_exports = [
    "DeploymentAgent",
    "DeploymentPrecision",
    "DeploymentStatus",
    "CalibrationConfig",
    "ValidationConfig",
    "DeploymentArtifact",
    "DeploymentResult",
    "ValidationResult",
]

# Import targets with optional dependencies
try:
    from .targets.jetson import JetsonTarget  # noqa: F401

    _exports.append("JetsonTarget")
except ImportError:
    pass

try:
    from .targets.openvino import OpenVINOTarget  # noqa: F401

    _exports.append("OpenVINOTarget")
except ImportError:
    pass

try:
    from .targets.coral import CoralTarget  # noqa: F401

    _exports.append("CoralTarget")
except ImportError:
    pass

__all__ = _exports
