"""Benchmark execution backends."""

from .base import BenchmarkBackend, BenchmarkResult
from .local_cpu import LocalCPUBackend

# Track what's available
_available_backends = ["BenchmarkBackend", "BenchmarkResult", "LocalCPUBackend"]

# Remote SSH backend (optional dependency: paramiko)
try:
    from .remote_ssh import RemoteSSHBackend

    _available_backends.append("RemoteSSHBackend")
except ImportError:
    pass

# Kubernetes backend (optional dependency: kubernetes)
try:
    from .kubernetes import KubernetesBackend

    _available_backends.append("KubernetesBackend")
except ImportError:
    pass

__all__ = _available_backends
