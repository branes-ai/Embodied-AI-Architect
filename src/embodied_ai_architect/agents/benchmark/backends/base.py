"""Base interface for benchmark execution backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel
import torch.nn as nn


class BenchmarkResult(BaseModel):
    """Result from a benchmark execution.

    This standardized format allows different backends to return
    comparable results.
    """

    backend_name: str
    device: str
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_samples_per_sec: float | None = None
    memory_allocated_mb: float | None = None
    memory_reserved_mb: float | None = None
    energy_joules: float | None = None
    iterations: int
    warmup_iterations: int
    metadata: Dict[str, Any] = {}


class BenchmarkBackend(ABC):
    """Abstract base class for benchmark execution backends.

    Backends handle the actual execution of benchmarks on different
    platforms (local CPU, GPU, remote cluster, robot, etc.).

    Future backends might include:
    - LocalGPUBackend: Run on local CUDA/ROCm GPU
    - RemoteClusterBackend: Dispatch to Kubernetes/Slurm cluster
    - EdgeDeviceBackend: Deploy to Jetson, RPi, etc.
    - RobotBackend: Deploy to physical robot for real-world testing
    - CloudBackend: Use AWS/GCP/Azure ML services
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system.

        Returns:
            True if backend can be used, False otherwise
        """
        pass

    @abstractmethod
    def execute_benchmark(
        self,
        model: nn.Module,
        input_shape: tuple,
        iterations: int = 100,
        warmup_iterations: int = 10,
        config: Dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Execute a benchmark on this backend.

        Args:
            model: PyTorch model to benchmark
            input_shape: Input tensor shape (batch_size, ...)
            iterations: Number of inference iterations
            warmup_iterations: Number of warmup iterations (not timed)
            config: Optional backend-specific configuration

        Returns:
            BenchmarkResult with timing and resource metrics
        """
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about backend capabilities.

        Returns:
            Dictionary describing what this backend can measure
        """
        return {
            "name": self.name,
            "measures_latency": True,
            "measures_throughput": False,
            "measures_memory": False,
            "measures_energy": False,
            "supports_remote": False,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
