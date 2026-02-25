"""Data models for hardware profiles and capabilities."""

from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class HardwareType(str, Enum):
    """Types of hardware accelerators."""

    CPU = "cpu"
    DSP = "dsp"
    GPU = "gpu"
    TPU = "tpu"
    DPU = "dpu"  # Dataflow Processing Unit
    KPU = "kpu"  # Knowledge Processing Unit / AI accelerator
    NPU = "npu"  # Neural Processing Unit
    FPGA = "fpga"


class ComputeParadigm(str, Enum):
    """Computation paradigm of the hardware."""

    VON_NEUMANN = "von_neumann"  # Traditional CPU architecture
    SIMD = "simd"  # Single Instruction Multiple Data
    DATAFLOW = "dataflow"  # Dataflow architectures
    SYSTOLIC_ARRAY = "systolic_array"  # TPU-style
    RECONFIGURABLE = "reconfigurable"  # FPGA


class OperationType(str, Enum):
    """Types of operations hardware is optimized for."""

    GENERAL_PURPOSE = "general_purpose"
    MATRIX_MULTIPLY = "matrix_multiply"
    CONVOLUTION = "convolution"
    ATTENTION = "attention"
    SPARSE_OPS = "sparse_ops"
    INT_QUANTIZED = "int_quantized"
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"


class HardwareCapability(BaseModel):
    """Detailed capabilities of a hardware platform."""

    # Compute specifications
    peak_tflops_fp32: float | None = None
    peak_tflops_fp16: float | None = None
    peak_tflops_int8: float | None = None

    # Memory specifications
    memory_gb: float
    memory_bandwidth_gbps: float
    cache_hierarchy: Dict[str, float] = Field(default_factory=dict)  # L1, L2, L3 sizes

    # Parallelism
    compute_units: int  # Cores, SMs, etc.
    simd_width: int | None = None

    # Power and thermal
    tdp_watts: float
    typical_power_watts: float | None = None

    # Software support
    frameworks: List[str] = Field(default_factory=list)  # PyTorch, TensorFlow, ONNX, etc.
    quantization_support: List[str] = Field(default_factory=list)  # int8, fp16, etc.

    # Special features
    tensor_cores: bool = False
    sparse_acceleration: bool = False
    custom_ops: List[str] = Field(default_factory=list)


class HardwareProfile(BaseModel):
    """Complete profile of a hardware platform."""

    # Identity
    name: str
    vendor: str
    hardware_type: HardwareType
    model: str

    # Architecture
    compute_paradigm: ComputeParadigm
    optimized_for: List[OperationType] = Field(default_factory=list)

    # Capabilities
    capabilities: HardwareCapability

    # Use case fit
    suitable_for: List[str] = Field(default_factory=list)  # Edge, cloud, datacenter, embedded
    target_applications: List[str] = Field(default_factory=list)  # Vision, NLP, robotics, etc.

    # Cost and availability
    approximate_cost_usd: float | None = None
    availability: str = "widely_available"  # widely_available, limited, proprietary

    # Metadata
    notes: str = ""

    def get_score_for_model(
        self,
        model_params: int,
        model_memory_mb: float,
        operation_types: List[str],
        target_latency_ms: float | None = None,
        max_power_watts: float | None = None,
    ) -> float:
        """Calculate a fitness score for this hardware given model requirements.

        Args:
            model_params: Number of model parameters
            model_memory_mb: Estimated model memory footprint
            operation_types: Types of operations in the model
            target_latency_ms: Target latency constraint
            max_power_watts: Maximum power budget

        Returns:
            Fitness score between 0-100 (higher is better)
        """
        score = 0.0
        max_score = 0.0

        # Memory fit (20 points)
        max_score += 20
        memory_gb_needed = model_memory_mb / 1024
        if self.capabilities.memory_gb >= memory_gb_needed:
            memory_ratio = memory_gb_needed / self.capabilities.memory_gb
            score += 20 * (1.0 - memory_ratio * 0.5)  # Better score for having headroom
        else:
            score += 0  # Doesn't fit

        # Power constraint (20 points)
        max_score += 20
        if max_power_watts is None:
            score += 15  # Neutral if no constraint
        elif self.capabilities.tdp_watts <= max_power_watts:
            power_ratio = self.capabilities.tdp_watts / max_power_watts
            score += 20 * (1.0 - power_ratio * 0.3)  # Reward lower power
        else:
            score += 5  # Penalty for exceeding power budget

        # Compute capability (30 points) - based on FLOPs
        max_score += 30
        if self.capabilities.peak_tflops_fp32:
            # Rough estimate: 2 FLOPs per parameter per inference
            estimated_gflops_needed = (model_params * 2) / 1e9
            if target_latency_ms:
                # Need to process in target_latency_ms
                gflops_per_sec_needed = estimated_gflops_needed / (target_latency_ms / 1000)
                tflops_needed = gflops_per_sec_needed / 1000
                if self.capabilities.peak_tflops_fp32 >= tflops_needed:
                    score += 30
                else:
                    score += 30 * (self.capabilities.peak_tflops_fp32 / tflops_needed)
            else:
                # No latency constraint, give full points if reasonable
                score += 25
        else:
            score += 15  # Unknown compute, neutral score

        # Operation type match (20 points)
        max_score += 20
        matching_ops = sum(
            1 for op in operation_types if op in [o.value for o in self.optimized_for]
        )
        if operation_types:
            score += 20 * (matching_ops / len(operation_types))
        else:
            score += 10

        # Special features bonus (10 points)
        max_score += 10
        if "matrix_multiply" in operation_types or "convolution" in operation_types:
            if self.capabilities.tensor_cores:
                score += 5
        if "sparse" in str(operation_types).lower():
            if self.capabilities.sparse_acceleration:
                score += 5

        # Normalize to 0-100
        return (score / max_score) * 100 if max_score > 0 else 0


class HardwareRecommendation(BaseModel):
    """A hardware recommendation with reasoning."""

    hardware: HardwareProfile
    score: float
    reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    estimated_performance: Dict[str, Any] = Field(default_factory=dict)
