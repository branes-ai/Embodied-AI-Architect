"""Benchmark Agent - Coordinates performance profiling across backends."""

from typing import Any, Dict, List
from ..base import BaseAgent, AgentResult
from .backends.base import BenchmarkBackend, BenchmarkResult
from .backends.local_cpu import LocalCPUBackend


class BenchmarkAgent(BaseAgent):
    """Agent that benchmarks models across different execution backends.

    The BenchmarkAgent acts as an orchestrator for performance profiling.
    It can dispatch benchmarks to different backends (local, remote, edge devices, etc.)
    and aggregate results.

    This agent demonstrates the plugin architecture - backends can be:
    - Built-in (LocalCPUBackend)
    - External packages (e.g., embodied-ai-architect-benchmark-gpu)
    - Custom implementations for specific hardware
    """

    def __init__(self, backends: List[BenchmarkBackend] | None = None):
        """Initialize the benchmark agent.

        Args:
            backends: List of backends to use. If None, uses LocalCPUBackend
        """
        super().__init__(name="Benchmark")

        # Register backends
        self.backends: Dict[str, BenchmarkBackend] = {}

        if backends is None:
            # Default to local CPU
            self._register_backend(LocalCPUBackend())
        else:
            for backend in backends:
                self._register_backend(backend)

    def _register_backend(self, backend: BenchmarkBackend) -> None:
        """Register a benchmark backend.

        Args:
            backend: Backend to register
        """
        if backend.is_available():
            self.backends[backend.name] = backend
            print(f"  Registered backend: {backend.name}")
        else:
            print(f"  Skipped unavailable backend: {backend.name}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute benchmarks across configured backends.

        Args:
            input_data: Dictionary with keys:
                - 'model': PyTorch model to benchmark
                - 'input_shape': Tuple specifying input shape
                - 'backends': Optional list of backend names to use (default: all)
                - 'iterations': Number of benchmark iterations (default: 100)
                - 'warmup_iterations': Number of warmup iterations (default: 10)

        Returns:
            AgentResult containing benchmark results from all backends
        """
        try:
            model = input_data.get("model")
            input_shape = input_data.get("input_shape")
            requested_backends = input_data.get("backends")
            iterations = input_data.get("iterations", 100)
            warmup_iterations = input_data.get("warmup_iterations", 10)

            if model is None:
                return AgentResult(success=False, data={}, error="No model provided")

            if input_shape is None:
                return AgentResult(success=False, data={}, error="No input_shape provided")

            # Determine which backends to use
            if requested_backends is None:
                backends_to_use = list(self.backends.keys())
            else:
                backends_to_use = [b for b in requested_backends if b in self.backends]

            if not backends_to_use:
                return AgentResult(
                    success=False, data={}, error="No available backends to run benchmarks"
                )

            # Run benchmarks on each backend
            results = {}
            for backend_name in backends_to_use:
                backend = self.backends[backend_name]
                print(f"\n  Running on {backend_name}...")

                benchmark_result = backend.execute_benchmark(
                    model=model,
                    input_shape=input_shape,
                    iterations=iterations,
                    warmup_iterations=warmup_iterations,
                )

                results[backend_name] = benchmark_result.model_dump()

            # Create summary
            summary = self._create_summary(results)

            return AgentResult(
                success=True,
                data={"benchmarks": results, "summary": summary},
                metadata={
                    "agent": self.name,
                    "backends_used": backends_to_use,
                    "iterations": iterations,
                },
            )

        except Exception as e:
            return AgentResult(
                success=False, data={}, error=f"Benchmark execution failed: {str(e)}"
            )

    def _create_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of benchmark results.

        Args:
            results: Benchmark results from all backends

        Returns:
            Summary dictionary
        """
        if not results:
            return {}

        summary = {
            "total_backends": len(results),
            "fastest_backend": None,
            "fastest_latency_ms": float("inf"),
        }

        # Find fastest backend
        for backend_name, result in results.items():
            mean_latency = result.get("mean_latency_ms", float("inf"))
            if mean_latency < summary["fastest_latency_ms"]:
                summary["fastest_latency_ms"] = mean_latency
                summary["fastest_backend"] = backend_name

        return summary

    def list_backends(self) -> List[str]:
        """List all registered backends.

        Returns:
            List of backend names
        """
        return list(self.backends.keys())

    def get_backend_capabilities(self, backend_name: str) -> Dict[str, Any] | None:
        """Get capabilities of a specific backend.

        Args:
            backend_name: Name of the backend

        Returns:
            Capability dictionary or None if backend not found
        """
        backend = self.backends.get(backend_name)
        if backend:
            return backend.get_capabilities()
        return None
