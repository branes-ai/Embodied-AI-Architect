"""Benchmark execution backends."""

from .base import BenchmarkBackend, BenchmarkResult  # noqa: F401
from .local_cpu import LocalCPUBackend  # noqa: F401

# Track what's available
_available_backends = ["BenchmarkBackend", "BenchmarkResult", "LocalCPUBackend"]

# Remote SSH backend (optional dependency: paramiko)
try:
    from .remote_ssh import RemoteSSHBackend  # noqa: F401

    _available_backends.append("RemoteSSHBackend")
except ImportError:
    pass

# Kubernetes backend (optional dependency: kubernetes)
try:
    from .kubernetes import KubernetesBackend  # noqa: F401

    _available_backends.append("KubernetesBackend")
except ImportError:
    pass

__all__ = _available_backends
