"""Hardware knowledge base with profiles of various accelerators."""

from typing import List
from .models import (
    HardwareProfile,
    HardwareCapability,
    HardwareType,
    ComputeParadigm,
    OperationType,
)


def get_default_hardware_profiles() -> List[HardwareProfile]:
    """Return a list of common hardware profiles.

    This knowledge base can be extended or replaced with custom profiles.
    """
    profiles = []

    # Intel CPU (x86)
    profiles.append(
        HardwareProfile(
            name="Intel Xeon Silver 4314",
            vendor="Intel",
            hardware_type=HardwareType.CPU,
            model="Xeon Silver 4314",
            compute_paradigm=ComputeParadigm.VON_NEUMANN,
            optimized_for=[OperationType.GENERAL_PURPOSE, OperationType.FP32],
            capabilities=HardwareCapability(
                peak_tflops_fp32=1.08,
                peak_tflops_fp16=None,
                peak_tflops_int8=None,
                memory_gb=128,
                memory_bandwidth_gbps=204.8,
                cache_hierarchy={"L1": 0.001, "L2": 0.025, "L3": 24.0},
                compute_units=16,
                simd_width=512,  # AVX-512
                tdp_watts=135,
                typical_power_watts=100,
                frameworks=["PyTorch", "TensorFlow", "ONNX", "OpenVINO"],
                quantization_support=["int8", "fp32"],
                tensor_cores=False,
                sparse_acceleration=False,
            ),
            suitable_for=["cloud", "datacenter", "edge_server"],
            target_applications=["general_purpose", "inference", "training"],
            approximate_cost_usd=500,
            availability="widely_available",
            notes="General-purpose server CPU with AVX-512 support",
        )
    )

    # NVIDIA GPU (A100)
    profiles.append(
        HardwareProfile(
            name="NVIDIA A100",
            vendor="NVIDIA",
            hardware_type=HardwareType.GPU,
            model="A100-40GB",
            compute_paradigm=ComputeParadigm.SIMD,
            optimized_for=[
                OperationType.MATRIX_MULTIPLY,
                OperationType.CONVOLUTION,
                OperationType.FP16,
                OperationType.FP32,
                OperationType.INT_QUANTIZED,
            ],
            capabilities=HardwareCapability(
                peak_tflops_fp32=19.5,
                peak_tflops_fp16=312,  # With tensor cores
                peak_tflops_int8=624,
                memory_gb=40,
                memory_bandwidth_gbps=1555,
                cache_hierarchy={"L1": 0.192, "L2": 40.0},
                compute_units=108,  # SMs
                simd_width=32,
                tdp_watts=400,
                typical_power_watts=300,
                frameworks=["PyTorch", "TensorFlow", "ONNX", "TensorRT", "JAX"],
                quantization_support=["int8", "fp16", "fp32", "bf16"],
                tensor_cores=True,
                sparse_acceleration=True,
            ),
            suitable_for=["cloud", "datacenter"],
            target_applications=["training", "inference", "large_models"],
            approximate_cost_usd=10000,
            availability="widely_available",
            notes="High-end datacenter GPU with 3rd gen tensor cores",
        )
    )

    # NVIDIA Edge GPU (Jetson AGX Orin)
    profiles.append(
        HardwareProfile(
            name="NVIDIA Jetson AGX Orin",
            vendor="NVIDIA",
            hardware_type=HardwareType.GPU,
            model="AGX Orin 64GB",
            compute_paradigm=ComputeParadigm.SIMD,
            optimized_for=[
                OperationType.MATRIX_MULTIPLY,
                OperationType.CONVOLUTION,
                OperationType.FP16,
                OperationType.INT_QUANTIZED,
            ],
            capabilities=HardwareCapability(
                peak_tflops_fp32=5.3,
                peak_tflops_fp16=170,  # With tensor cores
                peak_tflops_int8=340,
                memory_gb=64,
                memory_bandwidth_gbps=204.8,
                cache_hierarchy={"L1": 0.064, "L2": 4.0},
                compute_units=16,  # SMs
                simd_width=32,
                tdp_watts=60,
                typical_power_watts=30,
                frameworks=["PyTorch", "TensorFlow", "ONNX", "TensorRT"],
                quantization_support=["int8", "fp16", "fp32"],
                tensor_cores=True,
                sparse_acceleration=True,
            ),
            suitable_for=["edge", "robotics", "autonomous_vehicles"],
            target_applications=["vision", "robotics", "real_time_inference"],
            approximate_cost_usd=2000,
            availability="widely_available",
            notes="High-performance edge AI platform for robotics",
        )
    )

    # Google TPU v4
    profiles.append(
        HardwareProfile(
            name="Google TPU v4",
            vendor="Google",
            hardware_type=HardwareType.TPU,
            model="TPU v4",
            compute_paradigm=ComputeParadigm.SYSTOLIC_ARRAY,
            optimized_for=[
                OperationType.MATRIX_MULTIPLY,
                OperationType.CONVOLUTION,
                OperationType.ATTENTION,
                OperationType.BF16,
            ],
            capabilities=HardwareCapability(
                peak_tflops_fp32=None,
                peak_tflops_fp16=275,  # BF16
                peak_tflops_int8=550,
                memory_gb=32,
                memory_bandwidth_gbps=1200,
                cache_hierarchy={},  # Systolic array architecture
                compute_units=1,  # Single systolic array
                simd_width=None,
                tdp_watts=175,
                typical_power_watts=150,
                frameworks=["TensorFlow", "JAX", "PyTorch/XLA"],
                quantization_support=["int8", "bf16"],
                tensor_cores=False,  # Uses systolic array instead
                sparse_acceleration=False,
            ),
            suitable_for=["cloud", "datacenter"],
            target_applications=["training", "inference", "transformers"],
            approximate_cost_usd=None,  # Cloud only
            availability="limited",
            notes="Optimized for large-scale ML workloads, especially transformers",
        )
    )

    # Edge TPU (Coral)
    profiles.append(
        HardwareProfile(
            name="Google Coral Edge TPU",
            vendor="Google",
            hardware_type=HardwareType.TPU,
            model="Edge TPU",
            compute_paradigm=ComputeParadigm.SYSTOLIC_ARRAY,
            optimized_for=[
                OperationType.MATRIX_MULTIPLY,
                OperationType.CONVOLUTION,
                OperationType.INT_QUANTIZED,
            ],
            capabilities=HardwareCapability(
                peak_tflops_fp32=None,
                peak_tflops_fp16=None,
                peak_tflops_int8=4.0,
                memory_gb=0.256,  # 256 MB on-chip
                memory_bandwidth_gbps=34,
                cache_hierarchy={},
                compute_units=1,
                simd_width=None,
                tdp_watts=2,
                typical_power_watts=1.5,
                frameworks=["TensorFlow Lite", "ONNX"],
                quantization_support=["int8"],
                tensor_cores=False,
                sparse_acceleration=False,
            ),
            suitable_for=["edge", "embedded", "iot"],
            target_applications=["vision", "object_detection", "embedded_inference"],
            approximate_cost_usd=60,
            availability="widely_available",
            notes="Ultra-low power edge inference accelerator",
        )
    )

    # Intel Neural Compute Stick (Myriad X)
    profiles.append(
        HardwareProfile(
            name="Intel Neural Compute Stick 2",
            vendor="Intel",
            hardware_type=HardwareType.KPU,
            model="Myriad X",
            compute_paradigm=ComputeParadigm.DATAFLOW,
            optimized_for=[OperationType.CONVOLUTION, OperationType.FP16],
            capabilities=HardwareCapability(
                peak_tflops_fp32=None,
                peak_tflops_fp16=1.0,
                peak_tflops_int8=None,
                memory_gb=0.512,
                memory_bandwidth_gbps=34,
                cache_hierarchy={},
                compute_units=16,  # Imaging cores
                simd_width=None,
                tdp_watts=1,
                typical_power_watts=0.5,
                frameworks=["OpenVINO", "ONNX"],
                quantization_support=["fp16"],
                tensor_cores=False,
                sparse_acceleration=False,
            ),
            suitable_for=["edge", "embedded", "iot"],
            target_applications=["vision", "object_detection"],
            approximate_cost_usd=70,
            availability="widely_available",
            notes="USB stick form factor for edge inference",
        )
    )

    # Raspberry Pi 4 (ARM CPU)
    profiles.append(
        HardwareProfile(
            name="Raspberry Pi 4",
            vendor="Raspberry Pi Foundation",
            hardware_type=HardwareType.CPU,
            model="BCM2711",
            compute_paradigm=ComputeParadigm.VON_NEUMANN,
            optimized_for=[OperationType.GENERAL_PURPOSE],
            capabilities=HardwareCapability(
                peak_tflops_fp32=0.024,  # Very rough estimate
                peak_tflops_fp16=None,
                peak_tflops_int8=None,
                memory_gb=8,
                memory_bandwidth_gbps=12.8,
                cache_hierarchy={"L1": 0.000048, "L2": 0.001},
                compute_units=4,
                simd_width=128,  # NEON
                tdp_watts=15,
                typical_power_watts=7,
                frameworks=["PyTorch", "TensorFlow Lite", "ONNX"],
                quantization_support=["int8", "fp32"],
                tensor_cores=False,
                sparse_acceleration=False,
            ),
            suitable_for=["edge", "embedded", "iot", "hobbyist"],
            target_applications=["small_models", "prototyping", "education"],
            approximate_cost_usd=75,
            availability="widely_available",
            notes="Popular low-cost ARM-based SBC",
        )
    )

    # AMD GPU (MI250X)
    profiles.append(
        HardwareProfile(
            name="AMD Instinct MI250X",
            vendor="AMD",
            hardware_type=HardwareType.GPU,
            model="MI250X",
            compute_paradigm=ComputeParadigm.SIMD,
            optimized_for=[
                OperationType.MATRIX_MULTIPLY,
                OperationType.CONVOLUTION,
                OperationType.FP16,
                OperationType.FP32,
            ],
            capabilities=HardwareCapability(
                peak_tflops_fp32=47.9,
                peak_tflops_fp16=383,  # With matrix cores
                peak_tflops_int8=None,
                memory_gb=128,  # HBM2e
                memory_bandwidth_gbps=3277,
                cache_hierarchy={"L1": 0.128, "L2": 16.0},
                compute_units=220,  # CUs
                simd_width=64,
                tdp_watts=560,
                typical_power_watts=500,
                frameworks=["PyTorch", "TensorFlow", "ONNX", "ROCm"],
                quantization_support=["fp16", "fp32", "bf16"],
                tensor_cores=False,  # Uses matrix cores
                sparse_acceleration=False,
            ),
            suitable_for=["cloud", "datacenter", "hpc"],
            target_applications=["training", "inference", "large_models"],
            approximate_cost_usd=15000,
            availability="widely_available",
            notes="High-end datacenter GPU competitor to NVIDIA",
        )
    )

    return profiles


def get_hardware_by_type(hardware_type: HardwareType) -> List[HardwareProfile]:
    """Get all hardware profiles of a specific type.

    Args:
        hardware_type: Type of hardware to filter

    Returns:
        List of hardware profiles matching the type
    """
    all_profiles = get_default_hardware_profiles()
    return [p for p in all_profiles if p.hardware_type == hardware_type]


def get_hardware_by_name(name: str) -> HardwareProfile | None:
    """Get a specific hardware profile by name.

    Args:
        name: Name of the hardware

    Returns:
        Hardware profile or None if not found
    """
    all_profiles = get_default_hardware_profiles()
    for profile in all_profiles:
        if profile.name.lower() == name.lower():
            return profile
    return None
