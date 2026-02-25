"""Deployment targets for different hardware platforms."""

from .base import DeploymentTarget  # noqa: F401

_available_targets = ["DeploymentTarget"]

# Jetson/TensorRT target (optional: tensorrt, pycuda)
try:
    from .jetson import JetsonTarget  # noqa: F401

    _available_targets.append("JetsonTarget")
except ImportError:
    pass

# OpenVINO target (optional: openvino, nncf)
try:
    from .openvino import OpenVINOTarget  # noqa: F401

    _available_targets.append("OpenVINOTarget")
except ImportError:
    pass

# Coral Edge TPU target (optional: tensorflow, onnx2tf)
try:
    from .coral import CoralTarget  # noqa: F401

    _available_targets.append("CoralTarget")
except ImportError:
    pass

# Stillwater KPU target (optional: onnx)
try:
    from .kpu import StillwaterKPUTarget  # noqa: F401

    _available_targets.append("StillwaterKPUTarget")
except ImportError:
    pass

# NVIDIA NVDLA target (optional: onnx)
try:
    from .nvdla import NVDLATarget  # noqa: F401

    _available_targets.append("NVDLATarget")
except ImportError:
    pass

__all__ = _available_targets
