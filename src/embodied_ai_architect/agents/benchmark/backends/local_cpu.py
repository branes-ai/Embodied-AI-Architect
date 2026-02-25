"""Local CPU benchmark backend."""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict

from .base import BenchmarkBackend, BenchmarkResult


class LocalCPUBackend(BenchmarkBackend):
    """Benchmark execution on local CPU.

    This is the simplest backend - runs inference on the local machine's CPU.
    Useful for development and baseline comparisons.
    """

    def __init__(self):
        super().__init__(name="local_cpu")

    def is_available(self) -> bool:
        """CPU is always available.

        Returns:
            True
        """
        return True

    def execute_benchmark(
        self,
        model: nn.Module,
        input_shape: tuple,
        iterations: int = 100,
        warmup_iterations: int = 10,
        config: Dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Execute benchmark on local CPU.

        Args:
            model: PyTorch model to benchmark
            input_shape: Input tensor shape
            iterations: Number of timed iterations
            warmup_iterations: Number of warmup iterations
            config: Optional configuration (unused for CPU backend)

        Returns:
            BenchmarkResult with timing statistics
        """
        # Move model to CPU and set to eval mode
        device = torch.device("cpu")
        model = model.to(device)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(*input_shape, device=device)

        # Warmup iterations
        print(f"  Warming up ({warmup_iterations} iterations)...")
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)

        # Timed iterations
        print(f"  Running benchmark ({iterations} iterations)...")
        latencies = []

        with torch.no_grad():
            for _ in range(iterations):
                start_time = time.perf_counter()
                _ = model(dummy_input)
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

        # Calculate statistics
        latencies_np = np.array(latencies)
        mean_latency = float(np.mean(latencies_np))
        std_latency = float(np.std(latencies_np))
        min_latency = float(np.min(latencies_np))
        max_latency = float(np.max(latencies_np))

        # Calculate throughput (samples per second)
        batch_size = input_shape[0] if len(input_shape) > 0 else 1
        throughput = (1000.0 / mean_latency) * batch_size if mean_latency > 0 else 0.0

        return BenchmarkResult(
            backend_name=self.name,
            device="cpu",
            mean_latency_ms=mean_latency,
            std_latency_ms=std_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            throughput_samples_per_sec=throughput,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            metadata={
                "input_shape": list(input_shape),
                "batch_size": batch_size,
            },
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """Return CPU backend capabilities.

        Returns:
            Capability dictionary
        """
        return {
            "name": self.name,
            "measures_latency": True,
            "measures_throughput": True,
            "measures_memory": False,
            "measures_energy": False,
            "supports_remote": False,
        }
