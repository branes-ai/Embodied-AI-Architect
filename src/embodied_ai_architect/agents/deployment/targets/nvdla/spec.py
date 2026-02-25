"""NVDLA Virtual Platform Specification.

This module defines the contracts and data models for integrating with
the NVIDIA Deep Learning Accelerator Virtual Platform.

NVDLA Architecture Overview:
---------------------------
NVDLA is a modular, scalable, and open deep learning accelerator with:
- Convolution Core: Main compute engine for convolutions
- SDP (Single Data Processor): Element-wise operations, batch norm, ReLU
- PDP (Planar Data Processor): Pooling operations
- CDP (Channel Data Processor): LRN, softmax
- RUBIK: Data reshape engine
- BDMA: Bridge DMA for data movement

Variants:
- nv_small: Minimal configuration for embedded
- nv_large: Full configuration for server/automotive
- nv_full: Maximum configuration

The software stack consists of:
1. Compiler: Caffe model → Loadable (offline, on host)
2. Runtime: Loadable → Inference (on target with NVDLA)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# =============================================================================
# Configuration Models
# =============================================================================


class NVDLAPrecision(str, Enum):
    """Supported precisions on NVDLA."""

    FP16 = "fp16"  # 16-bit floating point
    INT8 = "int8"  # 8-bit integer with calibration


class NVDLAVariant(str, Enum):
    """NVDLA hardware variants."""

    NV_SMALL = "nv_small"  # Minimal: 64 MACs, 512KB SRAM
    NV_LARGE = "nv_large"  # Large: 2048 MACs, 2MB SRAM
    NV_FULL = "nv_full"  # Full: 4096 MACs, 4MB SRAM


class SimulationMode(str, Enum):
    """Virtual Platform simulation modes."""

    SYSTEMC = "systemc"  # Pure SystemC simulation (default)
    COSIM = "cosim"  # RTL co-simulation (--cosim flag)
    FPGA = "fpga"  # AWS FPGA execution (--fpga flag)


@dataclass
class NVDLAHardwareConfig:
    """NVDLA hardware configuration matching hw/spec files."""

    # Convolution core
    mac_atomic_c: int = 64  # Atomic channel size
    mac_atomic_k: int = 32  # Atomic kernel size
    memory_atomic_size: int = 32  # Memory transaction size

    # Memory
    cbuf_bank_width: int = 64  # Convolution buffer bank width
    cbuf_bank_depth: int = 512  # Convolution buffer depth
    cbuf_bank_num: int = 16  # Number of banks

    # Data types
    weight_compression: bool = True
    winograd: bool = True
    batch_size_max: int = 32

    # Derived: total CBUF size
    @property
    def cbuf_size_bytes(self) -> int:
        return self.cbuf_bank_width * self.cbuf_bank_depth * self.cbuf_bank_num


@dataclass
class NVDLAConfig:
    """Complete NVDLA configuration for deployment.

    This configuration specifies the target NVDLA variant and
    Virtual Platform settings.
    """

    # Hardware variant
    variant: NVDLAVariant = NVDLAVariant.NV_FULL
    hardware: NVDLAHardwareConfig = field(default_factory=NVDLAHardwareConfig)

    # Supported precisions
    supported_precisions: list[NVDLAPrecision] = field(
        default_factory=lambda: [NVDLAPrecision.FP16, NVDLAPrecision.INT8]
    )

    # Virtual Platform paths
    vp_install_path: Path | None = None  # Path to VP installation
    compiler_path: Path | None = None  # Path to nvdla_compiler
    runtime_path: Path | None = None  # Path to nvdla_runtime

    # Simulation settings
    simulation_mode: SimulationMode = SimulationMode.SYSTEMC
    platform_config: Path | None = None  # Lua config file

    # Performance characteristics (from nv_full @ ~500MHz)
    clock_mhz: float = 500.0
    peak_tops_int8: float = 4.0  # 4096 MACs @ 500MHz
    peak_tflops_fp16: float = 2.0
    tdp_watts: float = 5.0  # Estimated for embedded implementation

    # Supported operators (NVDLA native ops)
    native_ops: list[str] = field(
        default_factory=lambda: [
            # Convolution core
            "Conv",
            "ConvTranspose",
            "Gemm",  # Fully connected via convolution
            "MatMul",
            # SDP (Single Data Processor)
            "Relu",
            "PRelu",
            "Sigmoid",
            "Tanh",
            "BatchNormalization",
            "Add",
            "Mul",
            "Sub",
            "Div",
            "Eltwise",
            "Clip",
            # PDP (Planar Data Processor)
            "MaxPool",
            "AveragePool",
            "GlobalAveragePool",
            "GlobalMaxPool",
            # CDP (Channel Data Processor)
            "LRN",
            "Softmax",
            # RUBIK (reshape)
            "Reshape",
            "Flatten",
            "Split",
            "Concat",
            "Transpose",
            "Squeeze",
            "Unsqueeze",
            # Memory/data utilities
            "Constant",
            "Identity",
            "Shape",
            "Cast",
            "Pad",
            "ReduceMean",
        ]
    )

    def validate(self) -> list[str]:
        """Validate configuration."""
        errors = []
        if self.clock_mhz <= 0:
            errors.append("Clock frequency must be positive")
        return errors


# =============================================================================
# Loadable Representation
# =============================================================================


@dataclass
class NVDLALoadable:
    """NVDLA compiled loadable representation.

    A loadable is the output of the NVDLA compiler - a binary blob
    containing network structure, weights, and execution schedule.
    """

    path: Path  # Path to .nvdla loadable file
    precision: NVDLAPrecision

    # Source model information
    source_model: str = ""
    source_hash: str = ""

    # Network information
    input_names: list[str] = field(default_factory=list)
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)

    # Size and memory requirements
    file_size_bytes: int = 0
    weight_bytes: int = 0
    activation_bytes: int = 0

    # Compilation metadata
    target_variant: NVDLAVariant = NVDLAVariant.NV_FULL
    compiler_version: str = ""
    compilation_flags: dict[str, Any] = field(default_factory=dict)

    # Performance estimates from compiler
    estimated_cycles: int = 0
    layer_count: int = 0


# =============================================================================
# Execution Results
# =============================================================================


@dataclass
class NVDLAPerformanceMetrics:
    """Performance metrics from NVDLA execution."""

    # Timing
    total_cycles: int = 0
    total_time_us: float = 0.0

    # Per-engine breakdown
    conv_cycles: int = 0
    sdp_cycles: int = 0
    pdp_cycles: int = 0
    cdp_cycles: int = 0
    rubik_cycles: int = 0
    bdma_cycles: int = 0

    # Memory statistics
    dram_read_bytes: int = 0
    dram_write_bytes: int = 0
    sram_read_bytes: int = 0
    sram_write_bytes: int = 0

    # Utilization
    conv_utilization: float = 0.0
    memory_bandwidth_utilization: float = 0.0

    # Power (estimated or measured)
    average_power_watts: float = 0.0
    energy_per_inference_mj: float = 0.0


@dataclass
class NVDLAExecutionResult:
    """Result from NVDLA runtime execution."""

    success: bool
    error_message: str | None = None

    # Output tensors
    outputs: dict[str, np.ndarray] = field(default_factory=dict)

    # Performance metrics
    metrics: NVDLAPerformanceMetrics = field(default_factory=NVDLAPerformanceMetrics)

    # Simulation mode used
    mode: SimulationMode = SimulationMode.SYSTEMC

    # Debug/trace information
    trace_log: str = ""


# =============================================================================
# Compiler Interface
# =============================================================================


class NVDLACompilerInterface(ABC):
    """Interface for NVDLA model compilation.

    IMPLEMENTATION REQUIREMENTS:
    ----------------------------
    The NVDLA compiler requires Caffe format input:
    1. .prototxt - Network architecture
    2. .caffemodel - Trained weights

    For ONNX models, you must first convert to Caffe format using
    tools like onnx2caffe or similar converters.

    Compilation steps:
    1. Convert ONNX → Caffe (if needed)
    2. Run nvdla_compiler to generate loadable
    3. Parse loadable metadata

    INT8 calibration:
    - NVDLA compiler supports INT8 with calibration tables
    - Calibration determines per-layer scaling factors
    """

    @abstractmethod
    def compile(
        self,
        model_path: Path,
        config: NVDLAConfig,
        precision: NVDLAPrecision,
        output_path: Path,
        calibration_data: Path | None = None,
    ) -> NVDLALoadable:
        """Compile model to NVDLA loadable.

        Args:
            model_path: Path to ONNX or Caffe model
            config: Target NVDLA configuration
            precision: Target precision (FP16 or INT8)
            output_path: Path for output loadable
            calibration_data: Path to calibration data for INT8

        Returns:
            NVDLALoadable with compiled model information

        Raises:
            CompilationError: If compilation fails
        """
        pass

    @abstractmethod
    def convert_onnx_to_caffe(
        self,
        onnx_path: Path,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Convert ONNX model to Caffe format.

        Args:
            onnx_path: Path to ONNX model
            output_dir: Directory for Caffe outputs

        Returns:
            Tuple of (prototxt_path, caffemodel_path)
        """
        pass

    @abstractmethod
    def validate_model(
        self,
        model_path: Path,
        config: NVDLAConfig,
    ) -> list[str]:
        """Validate model compatibility with NVDLA.

        Returns:
            List of issues (empty if compatible)
        """
        pass

    @abstractmethod
    def get_supported_ops(self) -> list[str]:
        """Get list of operators supported by NVDLA."""
        pass


# =============================================================================
# Runtime Interface
# =============================================================================


class NVDLARuntimeInterface(ABC):
    """Interface for NVDLA runtime execution.

    IMPLEMENTATION REQUIREMENTS:
    ----------------------------
    The runtime executes loadables on the Virtual Platform or hardware.

    For Virtual Platform:
    1. Start VP simulation (aarch64_toplevel)
    2. Wait for Linux boot
    3. Transfer loadable and input data
    4. Run nvdla_runtime
    5. Collect outputs and metrics

    The VP runs a full Linux system, so execution involves:
    - QEMU-like system simulation
    - File transfer via virtio or network
    - Command execution via SSH or console
    """

    @abstractmethod
    def load(self, loadable: NVDLALoadable) -> None:
        """Load a compiled loadable for execution.

        Args:
            loadable: Compiled NVDLA loadable
        """
        pass

    @abstractmethod
    def execute(
        self,
        inputs: dict[str, np.ndarray],
        mode: SimulationMode = SimulationMode.SYSTEMC,
    ) -> NVDLAExecutionResult:
        """Execute loaded model with given inputs.

        Args:
            inputs: Dictionary mapping input names to numpy arrays
            mode: Simulation mode to use

        Returns:
            NVDLAExecutionResult with outputs and metrics
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
    def unload(self) -> None:
        """Unload current loadable and free resources."""
        pass


# =============================================================================
# Virtual Platform Interface
# =============================================================================


class NVDLAVirtualPlatformInterface(ABC):
    """Interface for NVDLA Virtual Platform control.

    IMPLEMENTATION REQUIREMENTS:
    ----------------------------
    The Virtual Platform is a SystemC simulation that requires:
    1. VP binary (aarch64_toplevel)
    2. Platform config (Lua file)
    3. Linux kernel image
    4. Root filesystem

    The interface provides methods to:
    - Start/stop the VP simulation
    - Transfer files to/from the simulated system
    - Execute commands in the simulated Linux
    - Collect performance traces
    """

    @abstractmethod
    def start(self, config: NVDLAConfig) -> None:
        """Start the Virtual Platform simulation.

        This boots the simulated Linux system and waits for readiness.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the Virtual Platform simulation."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if VP is currently running."""
        pass

    @abstractmethod
    def transfer_file_to_vp(self, local_path: Path, remote_path: str) -> None:
        """Transfer a file to the simulated system."""
        pass

    @abstractmethod
    def transfer_file_from_vp(self, remote_path: str, local_path: Path) -> None:
        """Transfer a file from the simulated system."""
        pass

    @abstractmethod
    def execute_command(self, command: str, timeout: float = 60.0) -> tuple[int, str, str]:
        """Execute a command in the simulated Linux.

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        pass

    @abstractmethod
    def get_performance_trace(self) -> dict[str, Any]:
        """Get performance trace from the last execution."""
        pass


# =============================================================================
# Error Types
# =============================================================================


class NVDLAError(Exception):
    """Base exception for NVDLA errors."""

    pass


class CompilationError(NVDLAError):
    """Error during model compilation."""

    pass


class ConversionError(NVDLAError):
    """Error during ONNX to Caffe conversion."""

    pass


class RuntimeError(NVDLAError):
    """Error during runtime execution."""

    pass


class VirtualPlatformError(NVDLAError):
    """Error with Virtual Platform."""

    pass
