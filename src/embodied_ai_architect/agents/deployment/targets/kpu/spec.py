"""Stillwater KPU Simulator Specification.

This module defines the contracts and data models that the KPU simulator
must implement for integration with the deployment system.

The specification is organized into:
1. Configuration models - Hardware configuration and constraints
2. Program representation - Compiled model format
3. Compiler interface - PyTorch/torch.compile to KPU program compilation
4. Runtime interface - Program execution
5. Simulator interface - Cycle-accurate or functional simulation
6. Performance models - Latency, power, memory estimation

IMPLEMENTATION REQUIREMENTS:
---------------------------
To integrate a KPU simulator, implement these interfaces:

1. KPUCompilerInterface - Compile PyTorch models to KPUProgram via torch.compile
2. KPURuntimeInterface - Execute KPUProgram and return results
3. KPUSimulatorInterface - (Optional) Cycle-accurate simulation

The compiler integrates with PyTorch 2.0+ torch.compile:
- Register a custom backend for KPU code generation
- Lower FX graphs to KPU operations
- Handle quantization and tiling

The simulator can operate in multiple modes:
- FUNCTIONAL: Fast execution, approximate performance estimates
- CYCLE_ACCURATE: Detailed timing, cache effects, memory stalls
- POWER_TRACE: Per-operation power consumption tracking
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

# =============================================================================
# Configuration Models
# =============================================================================


class KPUPrecision(str, Enum):
    """Supported numeric precisions on KPU."""

    INT8 = "int8"  # 8-bit integer with scale/zero-point
    INT4 = "int4"  # 4-bit integer (weights only typically)
    FP16 = "fp16"  # IEEE 754 half precision
    BF16 = "bf16"  # Brain floating point
    POSIT8 = "posit8"  # 8-bit posit (es=0 or es=1)
    POSIT16 = "posit16"  # 16-bit posit


class MemoryType(str, Enum):
    """Memory hierarchy levels."""

    REGISTER = "register"  # Register file
    SRAM_L1 = "sram_l1"  # L1 scratchpad/cache
    SRAM_L2 = "sram_l2"  # L2 shared memory
    DRAM = "dram"  # External DRAM
    FLASH = "flash"  # Non-volatile storage


class SimulationMode(str, Enum):
    """Simulator fidelity levels."""

    FUNCTIONAL = "functional"  # Fast, approximate timing
    CYCLE_ACCURATE = "cycle_accurate"  # Detailed timing model
    POWER_TRACE = "power_trace"  # Power consumption tracking


@dataclass
class MemoryConfig:
    """Memory subsystem configuration."""

    sram_l1_bytes: int = 256 * 1024  # 256 KB L1
    sram_l2_bytes: int = 2 * 1024 * 1024  # 2 MB L2
    dram_bytes: int = 4 * 1024 * 1024 * 1024  # 4 GB DRAM
    dram_bandwidth_gbps: float = 25.6  # GB/s
    sram_bandwidth_gbps: float = 256.0  # GB/s (on-chip)

    # Cache configuration (if applicable)
    cache_line_bytes: int = 64
    l1_associativity: int = 4
    l2_associativity: int = 8


@dataclass
class ComputeConfig:
    """Compute unit configuration."""

    # Systolic array dimensions (if applicable)
    systolic_rows: int = 16
    systolic_cols: int = 16

    # Vector unit width
    vector_lanes: int = 16

    # Clock frequency
    clock_mhz: float = 500.0

    # Peak throughput per precision
    tops_int8: float = 4.0  # INT8 TOPS
    tops_fp16: float = 2.0  # FP16 TFLOPS
    tops_posit8: float = 3.0  # Posit8 TOPS (estimated)

    # Power characteristics
    tdp_watts: float = 5.0
    idle_watts: float = 0.5


@dataclass
class KPUConfig:
    """Complete KPU hardware configuration.

    This configuration should match the target hardware or simulator settings.
    The deployment system uses this to:
    1. Validate model fits in memory
    2. Estimate performance/power
    3. Guide tiling decisions
    """

    name: str = "swkpu-v1"
    version: str = "1.0.0"

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)

    # Supported precisions
    supported_precisions: list[KPUPrecision] = field(
        default_factory=lambda: [KPUPrecision.INT8, KPUPrecision.FP16]
    )

    # Maximum tensor dimensions
    max_tensor_dims: int = 6
    max_batch_size: int = 16

    # Operator support (operators that have native acceleration)
    native_ops: list[str] = field(
        default_factory=lambda: [
            # Compute ops
            "Conv",
            "MatMul",
            "Gemm",
            "Relu",
            "Add",
            "Mul",
            "Sub",
            "Div",
            "MaxPool",
            "AveragePool",
            "GlobalAveragePool",
            "BatchNormalization",
            "Softmax",
            "Sigmoid",
            "Tanh",
            "Concat",
            # Shape/data movement ops
            "Reshape",
            "Transpose",
            "Flatten",
            "Squeeze",
            "Unsqueeze",
            "Gather",
            "Slice",
            "Split",
            # Utility ops (always supported)
            "Constant",
            "Identity",
            "Shape",
            "Cast",
            "Pad",
            "Clip",
            "ReduceMean",
            "ReduceSum",
        ]
    )

    def validate(self) -> list[str]:
        """Validate configuration consistency."""
        errors = []
        if self.memory.sram_l1_bytes <= 0:
            errors.append("L1 SRAM must be positive")
        if self.compute.clock_mhz <= 0:
            errors.append("Clock frequency must be positive")
        if not self.supported_precisions:
            errors.append("At least one precision must be supported")
        return errors


# =============================================================================
# Program Representation
# =============================================================================


@dataclass
class MemoryLayout:
    """Tensor memory layout specification."""

    # Memory location
    memory_type: MemoryType = MemoryType.DRAM

    # Address/offset (for simulator)
    base_address: int = 0

    # Layout format: "NCHW", "NHWC", "tiled", etc.
    format: str = "NCHW"

    # For tiled layouts
    tile_shape: tuple[int, ...] | None = None

    # Alignment requirements
    alignment_bytes: int = 64

    # Whether tensor is contiguous in memory
    contiguous: bool = True


@dataclass
class TileConfig:
    """Configuration for tiled execution.

    Large tensors must be tiled to fit in on-chip SRAM.
    This config specifies how a tensor operation is tiled.
    """

    # Tile dimensions for each tensor axis
    # e.g., for Conv: {"N": 1, "C": 32, "H": 16, "W": 16}
    tile_sizes: dict[str, int] = field(default_factory=dict)

    # Execution order: which axis to iterate first
    # e.g., ["N", "H", "W", "C"] for NHWC-major tiling
    loop_order: list[str] = field(default_factory=list)

    # Double buffering for latency hiding
    double_buffer: bool = False

    # Prefetch distance (tiles ahead to prefetch)
    prefetch_distance: int = 1


@dataclass
class KPUTensor:
    """Tensor representation in KPU program.

    Tensors are the data flowing through the computation graph.
    Each tensor has a unique ID, shape, dtype, and memory layout.
    """

    id: str  # Unique identifier
    name: str  # Human-readable name
    shape: tuple[int, ...]
    dtype: KPUPrecision
    layout: MemoryLayout = field(default_factory=MemoryLayout)

    # Quantization parameters (for INT8/INT4)
    scale: float | None = None
    zero_point: int | None = None

    # Whether this is a model weight (constant)
    is_weight: bool = False

    # Whether this is an input/output of the model
    is_input: bool = False
    is_output: bool = False

    @property
    def size_bytes(self) -> int:
        """Calculate tensor size in bytes."""
        elements = int(np.prod(self.shape))
        bytes_per_element = {
            KPUPrecision.INT8: 1,
            KPUPrecision.INT4: 0.5,
            KPUPrecision.FP16: 2,
            KPUPrecision.BF16: 2,
            KPUPrecision.POSIT8: 1,
            KPUPrecision.POSIT16: 2,
        }
        return int(elements * bytes_per_element.get(self.dtype, 4))


@dataclass
class KPUOp:
    """Single operation in KPU program.

    Operations are the computation nodes in the graph.
    Each op has inputs, outputs, and operation-specific attributes.
    """

    id: str  # Unique identifier
    op_type: str  # ONNX op type: "Conv", "MatMul", etc.

    # Input/output tensor IDs
    input_ids: list[str] = field(default_factory=list)
    output_ids: list[str] = field(default_factory=list)

    # Operation attributes (from ONNX)
    attributes: dict[str, Any] = field(default_factory=dict)

    # Tiling configuration for this op
    tiling: TileConfig | None = None

    # Scheduling metadata
    schedule_order: int = 0  # Execution order
    fused_with: list[str] = field(default_factory=list)  # Fused op IDs

    # Performance estimates (filled by compiler)
    estimated_cycles: int = 0
    estimated_energy_pj: float = 0.0  # picojoules


@dataclass
class KPUProgram:
    """Compiled program for KPU execution.

    This is the output of the KPU compiler and input to the runtime.
    It contains everything needed to execute the model:
    - Computation graph (ops)
    - Tensor metadata
    - Memory allocation plan
    - Scheduling information
    """

    # Program metadata
    name: str
    version: str = "1.0"

    # Source model info
    source_model: str = ""  # Original ONNX path
    source_hash: str = ""  # Hash for cache invalidation

    # Target configuration
    target_config: KPUConfig = field(default_factory=KPUConfig)
    precision: KPUPrecision = KPUPrecision.INT8

    # Computation graph
    ops: list[KPUOp] = field(default_factory=list)
    tensors: dict[str, KPUTensor] = field(default_factory=dict)

    # Input/output tensor IDs
    input_ids: list[str] = field(default_factory=list)
    output_ids: list[str] = field(default_factory=list)

    # Memory allocation summary
    peak_sram_bytes: int = 0
    peak_dram_bytes: int = 0
    weight_bytes: int = 0
    activation_bytes: int = 0

    # Performance estimates
    estimated_total_cycles: int = 0
    estimated_total_energy_uj: float = 0.0  # microjoules

    # Serialized weights (if included)
    weights_blob: bytes | None = None

    def get_input_tensors(self) -> list[KPUTensor]:
        """Get input tensor specifications."""
        return [self.tensors[tid] for tid in self.input_ids]

    def get_output_tensors(self) -> list[KPUTensor]:
        """Get output tensor specifications."""
        return [self.tensors[tid] for tid in self.output_ids]

    def get_ops_in_order(self) -> list[KPUOp]:
        """Get operations in scheduled execution order."""
        return sorted(self.ops, key=lambda op: op.schedule_order)

    def validate(self) -> list[str]:
        """Validate program consistency."""
        errors = []

        # Check all tensor references exist
        for op in self.ops:
            for tid in op.input_ids + op.output_ids:
                if tid not in self.tensors:
                    errors.append(f"Op {op.id} references unknown tensor {tid}")

        # Check inputs/outputs defined
        if not self.input_ids:
            errors.append("No input tensors defined")
        if not self.output_ids:
            errors.append("No output tensors defined")

        return errors


# =============================================================================
# Execution Results
# =============================================================================


@dataclass
class KPUPerformanceMetrics:
    """Detailed performance metrics from execution."""

    # Timing
    total_cycles: int = 0
    total_time_us: float = 0.0  # microseconds

    # Breakdown by category
    compute_cycles: int = 0
    memory_stall_cycles: int = 0
    other_cycles: int = 0

    # Memory statistics
    dram_reads_bytes: int = 0
    dram_writes_bytes: int = 0
    sram_reads_bytes: int = 0
    sram_writes_bytes: int = 0

    # Cache statistics (if applicable)
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0

    # Power/energy
    average_power_watts: float = 0.0
    total_energy_uj: float = 0.0  # microjoules
    energy_per_inference_mj: float = 0.0  # millijoules

    # Utilization
    compute_utilization: float = 0.0  # 0-1
    memory_bandwidth_utilization: float = 0.0  # 0-1

    # Per-op breakdown (optional)
    op_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class KPUExecutionResult:
    """Result of executing a KPUProgram."""

    # Success status
    success: bool
    error_message: str | None = None

    # Output tensors as numpy arrays
    outputs: dict[str, np.ndarray] = field(default_factory=dict)

    # Performance metrics
    metrics: KPUPerformanceMetrics = field(default_factory=KPUPerformanceMetrics)

    # Execution mode used
    mode: SimulationMode = SimulationMode.FUNCTIONAL

    # Debug information
    trace: list[str] = field(default_factory=list)


# =============================================================================
# Compiler Interface
# =============================================================================


class KPUCompilerInterface(ABC):
    """Interface for PyTorch to KPU program compilation.

    IMPLEMENTATION REQUIREMENTS:
    ----------------------------
    The production compiler integrates with torch.compile:
    1. Register as a torch.compile backend
    2. Receive FX graph from torch.compile
    3. Lower FX operations to KPU operations
    4. Apply graph optimizations (fusion, constant folding)
    5. Determine tiling strategy for each operation
    6. Allocate memory for tensors
    7. Schedule operations for execution
    8. Quantize weights if target precision requires it

    The interface accepts model paths for flexibility:
    - .pt files: Load and trace via torch.compile
    - .onnx files: Legacy support / stub mode

    The compiler should be configurable via KPUConfig and should
    provide meaningful error messages for unsupported operators
    or configurations.
    """

    @abstractmethod
    def compile(
        self,
        model_path: Path,
        config: KPUConfig,
        precision: KPUPrecision,
        calibration_data: Iterator[np.ndarray] | None = None,
    ) -> KPUProgram:
        """Compile model to KPU program.

        Args:
            model_path: Path to model file (.pt for PyTorch, .onnx for legacy)
            config: Target KPU configuration
            precision: Target precision (may trigger quantization)
            calibration_data: Iterator yielding calibration inputs (for INT8)

        Returns:
            Compiled KPUProgram ready for execution

        Raises:
            CompilationError: If compilation fails
            UnsupportedOpError: If model contains unsupported operators
        """
        pass

    @abstractmethod
    def get_supported_ops(self) -> list[str]:
        """Return list of operators supported by this compiler."""
        pass

    @abstractmethod
    def estimate_memory(
        self,
        model_path: Path,
        config: KPUConfig,
        precision: KPUPrecision,
    ) -> dict[str, int]:
        """Estimate memory requirements without full compilation.

        Returns:
            Dictionary with keys: "weights", "activations", "peak_sram", "peak_dram"
        """
        pass

    @abstractmethod
    def validate_model(
        self,
        model_path: Path,
        config: KPUConfig,
    ) -> list[str]:
        """Validate model compatibility without compilation.

        Returns:
            List of issues (empty if model is compatible)
        """
        pass


# =============================================================================
# Runtime Interface
# =============================================================================


class KPURuntimeInterface(ABC):
    """Interface for KPU program execution.

    IMPLEMENTATION REQUIREMENTS:
    ----------------------------
    1. Load compiled KPUProgram
    2. Allocate memory for inputs/outputs
    3. Execute program with provided inputs
    4. Return outputs and performance metrics

    The runtime can be backed by:
    - Software simulator (functional or cycle-accurate)
    - Hardware device driver
    - Remote execution service
    """

    @abstractmethod
    def load_program(self, program: KPUProgram) -> None:
        """Load a compiled program for execution.

        This may allocate memory, load weights, etc.

        Args:
            program: Compiled KPU program

        Raises:
            RuntimeError: If program cannot be loaded
        """
        pass

    @abstractmethod
    def execute(
        self,
        inputs: dict[str, np.ndarray],
        mode: SimulationMode = SimulationMode.FUNCTIONAL,
    ) -> KPUExecutionResult:
        """Execute the loaded program with given inputs.

        Args:
            inputs: Dictionary mapping input tensor names to numpy arrays
            mode: Simulation fidelity mode

        Returns:
            KPUExecutionResult with outputs and metrics
        """
        pass

    @abstractmethod
    def get_input_specs(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Get input tensor specifications.

        Returns:
            List of (name, shape, dtype) tuples
        """
        pass

    @abstractmethod
    def get_output_specs(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Get output tensor specifications.

        Returns:
            List of (name, shape, dtype) tuples
        """
        pass

    @abstractmethod
    def unload_program(self) -> None:
        """Unload the current program and free resources."""
        pass


# =============================================================================
# Simulator Interface (Optional, for detailed simulation)
# =============================================================================


class KPUSimulatorInterface(ABC):
    """Interface for cycle-accurate KPU simulation.

    IMPLEMENTATION REQUIREMENTS:
    ----------------------------
    This is an optional interface for detailed simulation that provides:
    1. Cycle-accurate timing model
    2. Memory hierarchy simulation
    3. Power consumption tracking
    4. Execution tracing

    If not implemented, the runtime can use approximate models.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset simulator state."""
        pass

    @abstractmethod
    def step(self, cycles: int = 1) -> None:
        """Advance simulation by given cycles."""
        pass

    @abstractmethod
    def run_until_complete(self) -> int:
        """Run until program completes. Returns total cycles."""
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get current simulator state for debugging."""
        pass

    @abstractmethod
    def get_power_trace(self) -> list[tuple[int, float]]:
        """Get power consumption trace.

        Returns:
            List of (cycle, power_watts) tuples
        """
        pass

    @abstractmethod
    def get_memory_trace(self) -> list[tuple[int, str, int, int]]:
        """Get memory access trace.

        Returns:
            List of (cycle, "read"|"write", address, size) tuples
        """
        pass


# =============================================================================
# Calibration Data Provider
# =============================================================================


class CalibrationDataProvider(ABC):
    """Interface for providing calibration data for quantization.

    IMPLEMENTATION REQUIREMENTS:
    ----------------------------
    Provide representative input samples for INT8 quantization.
    The compiler uses these to determine optimal scale/zero-point values.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over calibration samples."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return number of calibration samples."""
        pass

    @abstractmethod
    def get_input_shape(self) -> tuple[int, ...]:
        """Return expected input shape."""
        pass


# =============================================================================
# Error Types
# =============================================================================


class KPUError(Exception):
    """Base exception for KPU errors."""

    pass


class CompilationError(KPUError):
    """Error during model compilation."""

    pass


class UnsupportedOpError(CompilationError):
    """Model contains unsupported operators."""

    def __init__(self, ops: list[str]):
        self.ops = ops
        super().__init__(f"Unsupported operators: {', '.join(ops)}")


class MemoryError(KPUError):
    """Model exceeds memory constraints."""

    def __init__(self, required: int, available: int, memory_type: str):
        self.required = required
        self.available = available
        self.memory_type = memory_type
        super().__init__(
            f"{memory_type} memory exceeded: {required} bytes required, "
            f"{available} bytes available"
        )


class RuntimeError(KPUError):
    """Error during program execution."""

    pass
