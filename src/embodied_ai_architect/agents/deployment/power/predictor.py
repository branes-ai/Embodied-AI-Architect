"""Power prediction model for deployment validation.

Predicts power consumption based on model characteristics and target hardware.
Uses a combination of roofline model parameters and empirical calibration data.

Power prediction is essential for embodied AI planning - we need to know if a
model will fit within the power budget before deployment.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PowerProfile(str, Enum):
    """Hardware power profiles for prediction."""

    # Ultra-low power edge devices
    CORAL_TPU = "coral_tpu"  # ~2W TDP
    SWKPU = "swkpu"  # ~5W TDP
    NVDLA_SMALL = "nvdla_small"  # ~1W (nv_small config)
    NVDLA_LARGE = "nvdla_large"  # ~3W (nv_large config)
    NVDLA_FULL = "nvdla_full"  # ~5W (nv_full config)
    JETSON_NANO = "jetson_nano"  # 5-10W modes
    JETSON_ORIN_NANO = "jetson_orin_nano"  # 7-15W modes

    # Mid-range edge devices
    JETSON_XAVIER_NX = "jetson_xavier_nx"  # 10-20W modes
    JETSON_ORIN_NX = "jetson_orin_nx"  # 10-25W modes
    INTEL_NCS2 = "intel_ncs2"  # ~1.5W TDP

    # High-power edge devices
    JETSON_AGX_XAVIER = "jetson_agx_xavier"  # 15-30W modes
    JETSON_AGX_ORIN = "jetson_agx_orin"  # 15-60W modes

    # Desktop/Server GPUs (for reference)
    NVIDIA_RTX_3080 = "nvidia_rtx_3080"  # 320W TDP
    NVIDIA_RTX_4090 = "nvidia_rtx_4090"  # 450W TDP
    NVIDIA_A100 = "nvidia_a100"  # 300-400W TDP

    # CPU-only
    INTEL_CORE_I7 = "intel_core_i7"  # ~65W TDP
    AMD_RYZEN_7 = "amd_ryzen_7"  # ~65W TDP


@dataclass
class HardwarePowerSpec:
    """Power specifications for a hardware target."""

    name: str
    tdp_watts: float  # Thermal Design Power
    idle_watts: float  # Idle power consumption
    peak_watts: float  # Maximum power draw

    # Efficiency parameters (ops/watt at different utilization levels)
    tops_per_watt_peak: float  # At full utilization
    tops_per_watt_typical: float  # At typical workload

    # Power scaling parameters
    power_exponent: float = 1.5  # Power scales with utilization^exponent


# Empirical power specifications for common hardware
HARDWARE_POWER_SPECS: dict[PowerProfile, HardwarePowerSpec] = {
    PowerProfile.CORAL_TPU: HardwarePowerSpec(
        name="Google Coral Edge TPU",
        tdp_watts=2.0,
        idle_watts=0.5,
        peak_watts=2.5,
        tops_per_watt_peak=2.0,  # 4 TOPS @ 2W
        tops_per_watt_typical=1.5,
    ),
    PowerProfile.SWKPU: HardwarePowerSpec(
        name="Stillwater KPU",
        tdp_watts=5.0,
        idle_watts=0.5,
        peak_watts=5.0,
        tops_per_watt_peak=0.8,  # 4 TOPS INT8 @ 5W
        tops_per_watt_typical=0.6,
        power_exponent=1.3,  # More linear scaling
    ),
    PowerProfile.NVDLA_SMALL: HardwarePowerSpec(
        name="NVDLA nv_small",
        tdp_watts=1.0,
        idle_watts=0.1,
        peak_watts=1.5,
        tops_per_watt_peak=0.5,  # 0.5 TOPS @ 1W (64 MACs @ 500MHz)
        tops_per_watt_typical=0.3,
    ),
    PowerProfile.NVDLA_LARGE: HardwarePowerSpec(
        name="NVDLA nv_large",
        tdp_watts=3.0,
        idle_watts=0.3,
        peak_watts=4.0,
        tops_per_watt_peak=0.67,  # 2 TOPS @ 3W (2048 MACs @ 500MHz)
        tops_per_watt_typical=0.5,
    ),
    PowerProfile.NVDLA_FULL: HardwarePowerSpec(
        name="NVDLA nv_full",
        tdp_watts=5.0,
        idle_watts=0.5,
        peak_watts=6.0,
        tops_per_watt_peak=0.8,  # 4 TOPS @ 5W (4096 MACs @ 500MHz)
        tops_per_watt_typical=0.6,
    ),
    PowerProfile.JETSON_NANO: HardwarePowerSpec(
        name="NVIDIA Jetson Nano",
        tdp_watts=10.0,
        idle_watts=2.0,
        peak_watts=10.0,
        tops_per_watt_peak=0.047,  # 0.47 TFLOPS @ 10W
        tops_per_watt_typical=0.035,
    ),
    PowerProfile.JETSON_ORIN_NANO: HardwarePowerSpec(
        name="NVIDIA Jetson Orin Nano",
        tdp_watts=15.0,
        idle_watts=3.0,
        peak_watts=15.0,
        tops_per_watt_peak=2.67,  # 40 TOPS @ 15W
        tops_per_watt_typical=2.0,
    ),
    PowerProfile.JETSON_XAVIER_NX: HardwarePowerSpec(
        name="NVIDIA Jetson Xavier NX",
        tdp_watts=15.0,
        idle_watts=3.0,
        peak_watts=20.0,
        tops_per_watt_peak=1.4,  # 21 TOPS @ 15W
        tops_per_watt_typical=1.0,
    ),
    PowerProfile.JETSON_ORIN_NX: HardwarePowerSpec(
        name="NVIDIA Jetson Orin NX",
        tdp_watts=25.0,
        idle_watts=5.0,
        peak_watts=25.0,
        tops_per_watt_peak=4.0,  # 100 TOPS @ 25W
        tops_per_watt_typical=3.0,
    ),
    PowerProfile.JETSON_AGX_XAVIER: HardwarePowerSpec(
        name="NVIDIA Jetson AGX Xavier",
        tdp_watts=30.0,
        idle_watts=5.0,
        peak_watts=30.0,
        tops_per_watt_peak=1.07,  # 32 TOPS @ 30W
        tops_per_watt_typical=0.8,
    ),
    PowerProfile.JETSON_AGX_ORIN: HardwarePowerSpec(
        name="NVIDIA Jetson AGX Orin",
        tdp_watts=60.0,
        idle_watts=10.0,
        peak_watts=60.0,
        tops_per_watt_peak=4.17,  # 275 TOPS @ 60W (INT8)
        tops_per_watt_typical=3.0,
    ),
    PowerProfile.INTEL_NCS2: HardwarePowerSpec(
        name="Intel Neural Compute Stick 2",
        tdp_watts=1.5,
        idle_watts=0.3,
        peak_watts=2.0,
        tops_per_watt_peak=0.67,  # 1 TOPS @ 1.5W
        tops_per_watt_typical=0.5,
    ),
    PowerProfile.NVIDIA_RTX_3080: HardwarePowerSpec(
        name="NVIDIA RTX 3080",
        tdp_watts=320.0,
        idle_watts=20.0,
        peak_watts=370.0,
        tops_per_watt_peak=0.89,  # 285 TFLOPS FP16 @ 320W
        tops_per_watt_typical=0.6,
    ),
    PowerProfile.NVIDIA_RTX_4090: HardwarePowerSpec(
        name="NVIDIA RTX 4090",
        tdp_watts=450.0,
        idle_watts=25.0,
        peak_watts=500.0,
        tops_per_watt_peak=1.46,  # 660 TFLOPS FP16 @ 450W
        tops_per_watt_typical=1.0,
    ),
    PowerProfile.NVIDIA_A100: HardwarePowerSpec(
        name="NVIDIA A100",
        tdp_watts=400.0,
        idle_watts=40.0,
        peak_watts=400.0,
        tops_per_watt_peak=0.78,  # 312 TFLOPS FP16 @ 400W
        tops_per_watt_typical=0.6,
    ),
    PowerProfile.INTEL_CORE_I7: HardwarePowerSpec(
        name="Intel Core i7",
        tdp_watts=65.0,
        idle_watts=10.0,
        peak_watts=125.0,  # Boost
        tops_per_watt_peak=0.01,  # ~0.65 TFLOPS @ 65W
        tops_per_watt_typical=0.008,
    ),
    PowerProfile.AMD_RYZEN_7: HardwarePowerSpec(
        name="AMD Ryzen 7",
        tdp_watts=65.0,
        idle_watts=10.0,
        peak_watts=142.0,  # Boost
        tops_per_watt_peak=0.012,
        tops_per_watt_typical=0.009,
    ),
}


@dataclass
class ModelPowerCharacteristics:
    """Power-relevant characteristics extracted from a model."""

    total_macs: int  # Multiply-Accumulate operations
    total_params: int  # Number of parameters
    memory_bandwidth_bytes: int  # Estimated memory access
    batch_size: int = 1
    precision_bits: int = 32  # fp32=32, fp16=16, int8=8


@dataclass
class PowerPrediction:
    """Predicted power consumption."""

    mean_watts: float
    min_watts: float
    max_watts: float
    confidence: float  # 0-1, how confident in prediction
    breakdown: dict[str, float]  # Component breakdown if available
    notes: list[str]  # Additional notes/warnings


class PowerPredictor:
    """Predicts power consumption for model-hardware combinations.

    Uses a combination of:
    1. Hardware TDP and power curves
    2. Model computational intensity (MACs, memory bandwidth)
    3. Precision (INT8 vs FP16 vs FP32)
    4. Empirical calibration data when available
    """

    def __init__(self):
        self._calibration_data: dict[str, dict[str, float]] = {}

    def add_calibration(
        self,
        model_name: str,
        hardware: PowerProfile,
        measured_watts: float,
        precision: str = "fp32",
    ) -> None:
        """Add calibration data from actual measurements.

        Calibration data improves prediction accuracy for similar models.
        """
        key = f"{model_name}:{hardware.value}:{precision}"
        self._calibration_data[key] = {
            "measured_watts": measured_watts,
            "hardware": hardware.value,
            "precision": precision,
        }

    def predict(
        self,
        model_characteristics: ModelPowerCharacteristics,
        hardware: PowerProfile,
        precision: str = "fp32",
        target_latency_ms: float | None = None,
    ) -> PowerPrediction:
        """Predict power consumption for a model on target hardware.

        Args:
            model_characteristics: Model computational characteristics
            hardware: Target hardware profile
            precision: Precision mode (fp32, fp16, int8)
            target_latency_ms: Target latency constraint (affects utilization)

        Returns:
            PowerPrediction with estimated power consumption
        """
        spec = HARDWARE_POWER_SPECS.get(hardware)
        if spec is None:
            return PowerPrediction(
                mean_watts=0.0,
                min_watts=0.0,
                max_watts=0.0,
                confidence=0.0,
                breakdown={},
                notes=[f"Unknown hardware profile: {hardware}"],
            )

        # Compute operations based on precision
        precision_multiplier = self._get_precision_multiplier(precision)
        effective_ops = model_characteristics.total_macs * 2  # MAC = 2 ops
        effective_ops / 1e12 * precision_multiplier

        # Estimate utilization based on compute intensity
        compute_intensity = self._compute_intensity(model_characteristics)
        estimated_utilization = min(1.0, compute_intensity * 0.8)

        # Power estimation using power law model
        # P = P_idle + (P_peak - P_idle) * utilization^exponent
        dynamic_power = (spec.peak_watts - spec.idle_watts) * (
            estimated_utilization**spec.power_exponent
        )
        mean_power = spec.idle_watts + dynamic_power

        # Adjust for precision efficiency
        if precision == "int8":
            mean_power *= 0.6  # INT8 typically 40% more efficient
        elif precision == "fp16":
            mean_power *= 0.75  # FP16 typically 25% more efficient

        # Bound predictions by hardware limits
        min_power = spec.idle_watts
        max_power = spec.peak_watts

        # Calculate confidence based on available data
        confidence = self._calculate_confidence(hardware, precision)

        notes = []
        if compute_intensity < 0.1:
            notes.append("Memory-bound workload - power may be lower than predicted")
        if compute_intensity > 10:
            notes.append("Compute-bound workload - likely at peak power")
        if precision == "int8" and hardware == PowerProfile.CORAL_TPU:
            confidence = min(confidence + 0.2, 1.0)  # Coral is INT8-native
            notes.append("Coral TPU is optimized for INT8 - high confidence")

        breakdown = {
            "compute_watts": dynamic_power * 0.7,
            "memory_watts": dynamic_power * 0.2,
            "other_watts": dynamic_power * 0.1 + spec.idle_watts,
        }

        return PowerPrediction(
            mean_watts=round(mean_power, 2),
            min_watts=round(min_power, 2),
            max_watts=round(max_power, 2),
            confidence=round(confidence, 2),
            breakdown=breakdown,
            notes=notes,
        )

    def predict_from_model_info(
        self,
        total_params: int,
        total_macs: int,
        hardware: PowerProfile | str,
        precision: str = "fp32",
        batch_size: int = 1,
    ) -> PowerPrediction:
        """Convenience method to predict from basic model info.

        Args:
            total_params: Number of model parameters
            total_macs: Number of MAC operations per inference
            hardware: Target hardware profile or string name
            precision: Precision mode
            batch_size: Batch size for inference

        Returns:
            PowerPrediction
        """
        if isinstance(hardware, str):
            try:
                hardware = PowerProfile(hardware)
            except ValueError:
                return PowerPrediction(
                    mean_watts=0.0,
                    min_watts=0.0,
                    max_watts=0.0,
                    confidence=0.0,
                    breakdown={},
                    notes=[f"Unknown hardware: {hardware}"],
                )

        # Estimate memory bandwidth (rough heuristic)
        bytes_per_param = 4 if precision == "fp32" else (2 if precision == "fp16" else 1)
        memory_bandwidth = total_params * bytes_per_param * batch_size * 2  # Read + write

        characteristics = ModelPowerCharacteristics(
            total_macs=total_macs * batch_size,
            total_params=total_params,
            memory_bandwidth_bytes=memory_bandwidth,
            batch_size=batch_size,
            precision_bits=32 if precision == "fp32" else (16 if precision == "fp16" else 8),
        )

        return self.predict(characteristics, hardware, precision)

    def _get_precision_multiplier(self, precision: str) -> float:
        """Get throughput multiplier for precision mode."""
        multipliers = {
            "fp32": 1.0,
            "fp16": 2.0,  # 2x ops/cycle for FP16
            "int8": 4.0,  # 4x ops/cycle for INT8
        }
        return multipliers.get(precision, 1.0)

    def _compute_intensity(self, characteristics: ModelPowerCharacteristics) -> float:
        """Calculate compute intensity (ops/byte)."""
        if characteristics.memory_bandwidth_bytes == 0:
            return 1.0
        return (characteristics.total_macs * 2) / characteristics.memory_bandwidth_bytes

    def _calculate_confidence(self, hardware: PowerProfile, precision: str) -> float:
        """Calculate prediction confidence based on calibration data."""
        base_confidence = 0.5  # Default confidence

        # Check for calibration data
        calibration_count = sum(
            1 for k, v in self._calibration_data.items() if v["hardware"] == hardware.value
        )
        if calibration_count > 0:
            base_confidence += min(0.3, calibration_count * 0.1)

        # Well-characterized hardware gets higher confidence
        well_characterized = {
            PowerProfile.JETSON_AGX_ORIN,
            PowerProfile.JETSON_ORIN_NX,
            PowerProfile.CORAL_TPU,
            PowerProfile.NVIDIA_A100,
        }
        if hardware in well_characterized:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def get_hardware_specs(self, hardware: PowerProfile) -> HardwarePowerSpec | None:
        """Get power specifications for a hardware target."""
        return HARDWARE_POWER_SPECS.get(hardware)

    def list_hardware_profiles(self) -> list[dict[str, Any]]:
        """List all available hardware power profiles."""
        return [
            {
                "profile": profile.value,
                "name": spec.name,
                "tdp_watts": spec.tdp_watts,
                "efficiency_tops_per_watt": spec.tops_per_watt_typical,
            }
            for profile, spec in HARDWARE_POWER_SPECS.items()
        ]


def estimate_model_power(
    total_params: int,
    total_macs: int,
    hardware: str,
    precision: str = "fp32",
) -> PowerPrediction:
    """Convenience function to estimate model power.

    Args:
        total_params: Number of model parameters
        total_macs: Number of MAC operations
        hardware: Hardware profile name (e.g., "jetson_orin_nano")
        precision: Precision mode ("fp32", "fp16", "int8")

    Returns:
        PowerPrediction with estimated power consumption
    """
    predictor = PowerPredictor()
    return predictor.predict_from_model_info(
        total_params=total_params,
        total_macs=total_macs,
        hardware=hardware,
        precision=precision,
    )
