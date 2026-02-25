"""Architecture runner for executing complete pipelines.

Loads architectures from embodied-schemas and executes them with timing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import time
import statistics

import numpy as np

from embodied_schemas import SoftwareArchitecture, Registry

from ..operators import create_operator, Operator


@dataclass
class OperatorTiming:
    """Timing results for a single operator."""

    operator_id: str
    instance_id: str
    execution_target: str
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.latencies_ms) if len(self.latencies_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        return sorted_lat[int(len(sorted_lat) * 0.95)]


@dataclass
class PipelineTiming:
    """Timing results for a complete pipeline run."""

    total_latencies_ms: list[float] = field(default_factory=list)
    operator_timings: dict[str, OperatorTiming] = field(default_factory=dict)

    @property
    def mean_total_ms(self) -> float:
        return statistics.mean(self.total_latencies_ms) if self.total_latencies_ms else 0.0

    @property
    def std_total_ms(self) -> float:
        return (
            statistics.stdev(self.total_latencies_ms) if len(self.total_latencies_ms) > 1 else 0.0
        )

    @property
    def throughput_fps(self) -> float:
        return 1000.0 / self.mean_total_ms if self.mean_total_ms > 0 else 0.0


@dataclass
class ArchitectureBenchmarkResult:
    """Complete benchmark result for an architecture."""

    architecture_id: str
    hardware_id: str
    variant_id: str | None

    # Timing
    timing: PipelineTiming

    # Power (optional)
    mean_power_w: float | None = None
    peak_power_w: float | None = None

    # Memory
    peak_memory_mb: float | None = None

    # Configuration
    iterations: int = 100
    warmup_iterations: int = 10

    # Metadata
    timestamp: str | None = None
    hardware_fingerprint: str | None = None
    software_fingerprint: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "architecture_id": self.architecture_id,
            "hardware_id": self.hardware_id,
            "variant_id": self.variant_id,
            "timing": {
                "mean_total_ms": self.timing.mean_total_ms,
                "std_total_ms": self.timing.std_total_ms,
                "throughput_fps": self.timing.throughput_fps,
                "operators": {
                    op_id: {
                        "operator_id": t.operator_id,
                        "execution_target": t.execution_target,
                        "mean_ms": t.mean_ms,
                        "std_ms": t.std_ms,
                        "min_ms": t.min_ms,
                        "max_ms": t.max_ms,
                        "p95_ms": t.p95_ms,
                    }
                    for op_id, t in self.timing.operator_timings.items()
                },
            },
            "power": (
                {
                    "mean_w": self.mean_power_w,
                    "peak_w": self.peak_power_w,
                }
                if self.mean_power_w
                else None
            ),
            "memory": (
                {
                    "peak_mb": self.peak_memory_mb,
                }
                if self.peak_memory_mb
                else None
            ),
            "config": {
                "iterations": self.iterations,
                "warmup_iterations": self.warmup_iterations,
            },
            "metadata": {
                "timestamp": self.timestamp,
                "hardware_fingerprint": self.hardware_fingerprint,
                "software_fingerprint": self.software_fingerprint,
            },
        }


class ArchitectureRunner:
    """Execute complete architecture pipelines with timing.

    Loads architecture definitions from embodied-schemas,
    instantiates operators, and runs the dataflow graph.
    """

    def __init__(self, architecture: SoftwareArchitecture, hardware_id: str | None = None):
        """Initialize runner with architecture.

        Args:
            architecture: SoftwareArchitecture from embodied-schemas
            hardware_id: Optional hardware ID for operator configuration
        """
        self.architecture = architecture
        self.hardware_id = hardware_id
        self.operators: dict[str, Operator] = {}
        self.dataflow: dict[str, list[tuple[str, str, str]]] = (
            {}
        )  # source -> [(target, src_port, tgt_port)]
        self._is_loaded = False

    @classmethod
    def from_registry(
        cls,
        architecture_id: str,
        variant_id: str | None = None,
        hardware_id: str | None = None,
        registry: Registry | None = None,
    ) -> "ArchitectureRunner":
        """Create runner from registry lookup.

        Args:
            architecture_id: Architecture ID to load
            variant_id: Optional variant to apply
            hardware_id: Optional hardware ID for configuration
            registry: Optional registry (loads default if None)

        Returns:
            Configured ArchitectureRunner
        """
        registry = registry or Registry.load()
        arch = registry.architectures.get(architecture_id)

        if not arch:
            raise ValueError(f"Architecture not found: {architecture_id}")

        # Apply variant if specified
        if variant_id:
            arch = cls._apply_variant(arch, variant_id)

        return cls(arch, hardware_id)

    @staticmethod
    def _apply_variant(
        arch: SoftwareArchitecture,
        variant_id: str,
    ) -> SoftwareArchitecture:
        """Apply variant overrides to architecture."""
        variant = None
        for v in arch.variants:
            if v.id == variant_id:
                variant = v
                break

        if not variant:
            raise ValueError(f"Variant not found: {variant_id}")

        # Create modified copy
        arch_dict = arch.model_dump()

        # Apply operator overrides
        for op in arch_dict["operators"]:
            if op["id"] in variant.operator_overrides:
                op["operator_id"] = variant.operator_overrides[op["id"]]

            if op["id"] in variant.config_overrides:
                op["config"].update(variant.config_overrides[op["id"]])

        return SoftwareArchitecture.model_validate(arch_dict)

    def load_operators(self) -> None:
        """Instantiate all operators in the architecture."""
        print(f"[ArchitectureRunner] Loading {len(self.architecture.operators)} operators...")

        for op_inst in self.architecture.operators:
            try:
                operator = create_operator(
                    op_inst.operator_id,
                    config=op_inst.config,
                    execution_target=op_inst.execution_target or "cpu",
                )
                self.operators[op_inst.id] = operator
                print(
                    f"  Loaded: {op_inst.id} ({op_inst.operator_id}) -> {op_inst.execution_target or 'cpu'}"
                )
            except Exception as e:
                print(f"  FAILED: {op_inst.id} ({op_inst.operator_id}): {e}")
                raise

        # Build dataflow graph
        self.dataflow = {}
        for edge in self.architecture.dataflow:
            if edge.source_op not in self.dataflow:
                self.dataflow[edge.source_op] = []
            self.dataflow[edge.source_op].append(
                (edge.target_op, edge.source_port, edge.target_port)
            )

        self._is_loaded = True
        print(f"[ArchitectureRunner] Ready with {len(self.operators)} operators")

    def run_pipeline(
        self,
        inputs: dict[str, Any],
        collect_timing: bool = True,
    ) -> tuple[dict[str, Any], dict[str, float] | None]:
        """Run full pipeline once.

        Args:
            inputs: Initial inputs keyed by operator instance ID
            collect_timing: Whether to collect per-operator timing

        Returns:
            (outputs dict, timings dict or None)
        """
        if not self._is_loaded:
            raise RuntimeError("Operators not loaded. Call load_operators() first.")

        # Data store for operator outputs
        data: dict[str, dict[str, Any]] = {}
        timings: dict[str, float] = {} if collect_timing else None

        # Initialize with inputs
        for op_id, op_inputs in inputs.items():
            data[op_id] = op_inputs

        # Execute operators in topological order
        # Simple approach: process operators in definition order
        # (assumes architecture defines operators in valid execution order)
        for op_inst in self.architecture.operators:
            op_id = op_inst.id
            operator = self.operators[op_id]

            # Gather inputs from upstream operators
            op_inputs = data.get(op_id, {})

            # Add outputs from upstream operators via dataflow edges
            for edge in self.architecture.dataflow:
                if edge.target_op == op_id:
                    src_data = data.get(edge.source_op, {})
                    if edge.source_port in src_data:
                        op_inputs[edge.target_port] = src_data[edge.source_port]

            # Execute operator
            if collect_timing:
                start = time.perf_counter_ns()

            try:
                outputs = operator.process(op_inputs)
            except Exception as e:
                print(f"Error in {op_id}: {e}")
                raise

            if collect_timing:
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                timings[op_id] = elapsed_ms

            # Store outputs
            data[op_id] = outputs

        return data, timings

    def benchmark(
        self,
        sample_inputs: dict[str, Any],
        iterations: int = 100,
        warmup: int = 10,
        power_monitor: Any = None,
    ) -> ArchitectureBenchmarkResult:
        """Run benchmark and return comprehensive results.

        Args:
            sample_inputs: Representative inputs for benchmarking
            iterations: Number of timed iterations
            warmup: Number of warmup iterations
            power_monitor: Optional PowerMonitor instance

        Returns:
            ArchitectureBenchmarkResult with timing and power data
        """
        if not self._is_loaded:
            self.load_operators()

        print(f"\n[Benchmark] Running {warmup} warmup + {iterations} iterations...")

        # Initialize timing storage
        timing = PipelineTiming()
        for op_inst in self.architecture.operators:
            timing.operator_timings[op_inst.id] = OperatorTiming(
                operator_id=op_inst.operator_id,
                instance_id=op_inst.id,
                execution_target=op_inst.execution_target or "cpu",
            )

        # Warmup
        print(f"[Benchmark] Warmup ({warmup} iterations)...")
        for _ in range(warmup):
            self.run_pipeline(sample_inputs, collect_timing=False)

        # Start power monitoring if available
        if power_monitor:
            power_monitor.start()

        # Timed iterations
        print(f"[Benchmark] Running ({iterations} iterations)...")
        for i in range(iterations):
            start = time.perf_counter_ns()
            _, op_timings = self.run_pipeline(sample_inputs, collect_timing=True)
            total_ms = (time.perf_counter_ns() - start) / 1e6

            timing.total_latencies_ms.append(total_ms)

            for op_id, lat in op_timings.items():
                timing.operator_timings[op_id].latencies_ms.append(lat)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{iterations} ({total_ms:.1f}ms)")

        # Stop power monitoring
        power_data = None
        if power_monitor:
            power_data = power_monitor.stop()

        # Get fingerprints if available
        hw_fingerprint = None
        sw_fingerprint = None
        try:
            from ...operators.hardware_detect import detect_hardware

            hw_info = detect_hardware()
            hw_fingerprint = hw_info.get("hardware_fingerprint")
            sw_fingerprint = hw_info.get("software_fingerprint")
        except Exception:
            pass

        # Build result
        result = ArchitectureBenchmarkResult(
            architecture_id=self.architecture.id,
            hardware_id=self.hardware_id or "unknown",
            variant_id=None,
            timing=timing,
            mean_power_w=power_data.mean_watts if power_data else None,
            peak_power_w=power_data.peak_watts if power_data else None,
            iterations=iterations,
            warmup_iterations=warmup,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            hardware_fingerprint=hw_fingerprint,
            software_fingerprint=sw_fingerprint,
        )

        return result

    def print_summary(self, result: ArchitectureBenchmarkResult) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print(f"BENCHMARK RESULTS: {result.architecture_id}")
        print("=" * 60)
        print(f"Hardware: {result.hardware_id}")
        if result.hardware_fingerprint:
            print(f"HW Fingerprint: {result.hardware_fingerprint}")
        print(f"Iterations: {result.iterations}")
        print()

        print("Pipeline Timing:")
        print(f"  Total: {result.timing.mean_total_ms:.2f} ± {result.timing.std_total_ms:.2f} ms")
        print(f"  Throughput: {result.timing.throughput_fps:.1f} fps")
        print()

        print("Per-Operator Timing:")
        print(f"  {'Operator':<20} {'Target':<8} {'Mean':<10} {'Std':<10} {'P95':<10}")
        print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

        for op_id, t in result.timing.operator_timings.items():
            print(
                f"  {op_id:<20} {t.execution_target:<8} {t.mean_ms:>8.2f}ms {t.std_ms:>8.2f}ms {t.p95_ms:>8.2f}ms"
            )

        if result.mean_power_w:
            print()
            print("Power:")
            print(f"  Mean: {result.mean_power_w:.1f} W")
            print(f"  Peak: {result.peak_power_w:.1f} W")

        # Check against architecture requirements
        print()
        print("Requirement Check:")
        if self.architecture.end_to_end_latency_ms:
            meets = result.timing.mean_total_ms <= self.architecture.end_to_end_latency_ms
            status = "✓" if meets else "✗"
            print(
                f"  {status} Latency: {result.timing.mean_total_ms:.1f}ms (target: {self.architecture.end_to_end_latency_ms}ms)"
            )

        if self.architecture.min_throughput_fps:
            meets = result.timing.throughput_fps >= self.architecture.min_throughput_fps
            status = "✓" if meets else "✗"
            print(
                f"  {status} Throughput: {result.timing.throughput_fps:.1f}fps (target: {self.architecture.min_throughput_fps}fps)"
            )

        if self.architecture.power_budget_w and result.mean_power_w:
            meets = result.mean_power_w <= self.architecture.power_budget_w
            status = "✓" if meets else "✗"
            print(
                f"  {status} Power: {result.mean_power_w:.1f}W (budget: {self.architecture.power_budget_w}W)"
            )

        print("=" * 60)

    def teardown(self) -> None:
        """Release all operator resources."""
        for op in self.operators.values():
            op.teardown()
        self.operators = {}
        self._is_loaded = False
