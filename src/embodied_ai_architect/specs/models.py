"""Pydantic model hierarchy for system specifications.

Defines structured, hierarchical specs for embodied AI systems with
8 subsystem categories. All fields are Optional to support incremental
definition — users build up specs over time.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# Import SuccessCriterion from embodied-schemas if available
try:
    from embodied_schemas import ConstraintCriticality, SuccessCriterion

    HAS_SCHEMAS = True
except ImportError:
    HAS_SCHEMAS = False
    ConstraintCriticality = None
    SuccessCriterion = None


# ── Enums ──────────────────────────────────────────────────────────────────


class PlatformType(str, Enum):
    """Platform form factors."""

    DRONE = "drone"
    QUADRUPED = "quadruped"
    BIPED = "biped"
    AMR = "amr"
    FIXED_CAMERA = "fixed_camera"
    HANDHELD = "handheld"
    VEHICLE = "vehicle"
    INDUSTRIAL_ARM = "industrial_arm"
    CUSTOM = "custom"


class AutonomyLevel(str, Enum):
    """Levels of autonomy (SAE-inspired)."""

    TELEOPERATED = "teleoperated"
    ASSISTED = "assisted"
    PARTIAL = "partial"
    CONDITIONAL = "conditional"
    HIGH = "high"
    FULL = "full"


class SafetyLevel(str, Enum):
    """Safety integrity levels."""

    NONE = "none"
    BASIC = "basic"
    SIL_1 = "sil_1"
    SIL_2 = "sil_2"
    SIL_3 = "sil_3"
    SIL_4 = "sil_4"


class CoolingType(str, Enum):
    """Thermal cooling methods."""

    PASSIVE = "passive"
    FAN = "fan"
    LIQUID = "liquid"
    HEATPIPE = "heatpipe"


class EnvironmentalRating(str, Enum):
    """Environmental protection ratings."""

    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    IP54 = "ip54"
    IP67 = "ip67"
    IP68 = "ip68"
    MIL_STD = "mil_std"


# ── Subsystem Specs ────────────────────────────────────────────────────────


class PerceptionSpec(BaseModel):
    """Perception subsystem requirements."""

    cameras: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of camera inputs",
    )
    camera_types: list[str] = Field(
        default_factory=list,
        description="Camera types (e.g., monocular, stereo, depth, thermal)",
    )
    detection_classes: list[str] = Field(
        default_factory=list,
        description="Object classes to detect",
    )
    tracking: Optional[bool] = Field(
        default=None,
        description="Whether multi-object tracking is required",
    )
    min_accuracy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum detection accuracy (mAP) as decimal",
    )
    max_latency_ms: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum perception latency in milliseconds",
    )
    min_fps: Optional[float] = Field(
        default=None,
        gt=0,
        description="Minimum frames per second",
    )
    resolution: Optional[str] = Field(
        default=None,
        description="Input resolution (e.g., 1920x1080, 640x480)",
    )
    model_family: Optional[str] = Field(
        default=None,
        description="Preferred model family (e.g., YOLOv8, EfficientDet)",
    )


class ComputeSpec(BaseModel):
    """Compute subsystem requirements."""

    soc: Optional[str] = Field(
        default=None,
        description="Target SoC (e.g., Jetson Orin NX, RK3588)",
    )
    cpu_cores: Optional[int] = Field(
        default=None,
        ge=1,
        description="Minimum CPU cores",
    )
    gpu: Optional[str] = Field(
        default=None,
        description="GPU type or requirement",
    )
    accelerator: Optional[str] = Field(
        default=None,
        description="AI accelerator (e.g., DLA, NPU, TPU)",
    )
    memory_gb: Optional[float] = Field(
        default=None,
        gt=0,
        description="Minimum system memory in GB",
    )
    storage_gb: Optional[float] = Field(
        default=None,
        gt=0,
        description="Minimum storage in GB",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Model quantization (e.g., FP16, INT8, INT4)",
    )
    max_tdp_watts: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum thermal design power in watts",
    )


class PowerSpec(BaseModel):
    """Power subsystem requirements."""

    battery_wh: Optional[float] = Field(
        default=None,
        gt=0,
        description="Battery capacity in watt-hours",
    )
    power_budget_watts: Optional[float] = Field(
        default=None,
        gt=0,
        description="Total system power budget in watts",
    )
    compute_power_watts: Optional[float] = Field(
        default=None,
        gt=0,
        description="Power allocated to compute in watts",
    )
    thermal_limit_c: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum operating temperature in Celsius",
    )
    cooling: Optional[CoolingType] = Field(
        default=None,
        description="Cooling method",
    )
    mission_duration_min: Optional[float] = Field(
        default=None,
        gt=0,
        description="Required mission duration in minutes",
    )


class SensorSpec(BaseModel):
    """Sensor subsystem requirements (beyond perception cameras)."""

    modalities: list[str] = Field(
        default_factory=list,
        description="Sensor modalities (e.g., lidar, imu, gps, barometer, ultrasonic)",
    )
    lidar_points_per_sec: Optional[int] = Field(
        default=None,
        gt=0,
        description="LiDAR point cloud rate",
    )
    imu_rate_hz: Optional[float] = Field(
        default=None,
        gt=0,
        description="IMU sampling rate in Hz",
    )
    data_rate_mbps: Optional[float] = Field(
        default=None,
        gt=0,
        description="Total sensor data rate in Mbps",
    )
    environmental_rating: Optional[EnvironmentalRating] = Field(
        default=None,
        description="Environmental protection requirement",
    )


class ActuatorSpec(BaseModel):
    """Actuator subsystem requirements."""

    dof: Optional[int] = Field(
        default=None,
        ge=0,
        description="Degrees of freedom",
    )
    control_rate_hz: Optional[float] = Field(
        default=None,
        gt=0,
        description="Control loop rate in Hz",
    )
    max_speed_mps: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum speed in meters per second",
    )
    payload_kg: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum payload in kilograms",
    )
    actuator_types: list[str] = Field(
        default_factory=list,
        description="Actuator types (e.g., brushless_motor, servo, linear)",
    )


class CommsSpec(BaseModel):
    """Communications subsystem requirements."""

    protocols: list[str] = Field(
        default_factory=list,
        description="Communication protocols (e.g., wifi, 5g, lora, ethernet, can)",
    )
    bandwidth_mbps: Optional[float] = Field(
        default=None,
        gt=0,
        description="Required bandwidth in Mbps",
    )
    max_latency_ms: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum communication latency in ms",
    )
    range_m: Optional[float] = Field(
        default=None,
        gt=0,
        description="Communication range in meters",
    )
    mesh_networking: Optional[bool] = Field(
        default=None,
        description="Whether mesh networking is required",
    )


class AutonomySpec(BaseModel):
    """Autonomy subsystem requirements."""

    level: Optional[AutonomyLevel] = Field(
        default=None,
        description="Level of autonomy",
    )
    planning: Optional[str] = Field(
        default=None,
        description="Planning approach (e.g., waypoint, path_planning, behavior_tree)",
    )
    navigation: Optional[str] = Field(
        default=None,
        description="Navigation method (e.g., slam, vio, gps_waypoint)",
    )
    decision_rate_hz: Optional[float] = Field(
        default=None,
        gt=0,
        description="Decision-making loop rate in Hz",
    )
    obstacle_avoidance: Optional[bool] = Field(
        default=None,
        description="Whether obstacle avoidance is required",
    )
    multi_agent: Optional[bool] = Field(
        default=None,
        description="Whether multi-agent coordination is needed",
    )


class SafetySpec(BaseModel):
    """Safety subsystem requirements."""

    level: Optional[SafetyLevel] = Field(
        default=None,
        description="Safety integrity level",
    )
    redundancy: Optional[str] = Field(
        default=None,
        description="Redundancy approach (e.g., none, dual, triple)",
    )
    failsafe_action: Optional[str] = Field(
        default=None,
        description="Failsafe behavior (e.g., land, stop, return_home)",
    )
    geofencing: Optional[bool] = Field(
        default=None,
        description="Whether geofencing is required",
    )
    certifications: list[str] = Field(
        default_factory=list,
        description="Required certifications (e.g., DO-178C, IEC-61508)",
    )
    watchdog_timeout_ms: Optional[float] = Field(
        default=None,
        gt=0,
        description="Watchdog timer timeout in milliseconds",
    )


# ── Root Spec ──────────────────────────────────────────────────────────────


class SystemSpec(BaseModel):
    """Root specification for an embodied AI system.

    All subsystem fields are Optional — users build up specs incrementally.
    """

    name: str = Field(
        description="Spec name (unique identifier)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the system",
    )
    platform_type: Optional[PlatformType] = Field(
        default=None,
        description="Platform form factor",
    )
    perception: Optional[PerceptionSpec] = Field(
        default=None,
        description="Perception subsystem requirements",
    )
    compute: Optional[ComputeSpec] = Field(
        default=None,
        description="Compute subsystem requirements",
    )
    power: Optional[PowerSpec] = Field(
        default=None,
        description="Power subsystem requirements",
    )
    sensors: Optional[SensorSpec] = Field(
        default=None,
        description="Sensor subsystem requirements",
    )
    actuators: Optional[ActuatorSpec] = Field(
        default=None,
        description="Actuator subsystem requirements",
    )
    comms: Optional[CommsSpec] = Field(
        default=None,
        description="Communications subsystem requirements",
    )
    autonomy: Optional[AutonomySpec] = Field(
        default=None,
        description="Autonomy subsystem requirements",
    )
    safety: Optional[SafetySpec] = Field(
        default=None,
        description="Safety subsystem requirements",
    )
    constraints: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Cross-cutting success criteria (SuccessCriterion dicts)",
    )
    custom: dict[str, Any] = Field(
        default_factory=dict,
        description="User-defined extensibility fields",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Freeform tags for organization",
    )

    def summary(self) -> str:
        """One-line summary of the spec."""
        parts = [self.name]
        if self.platform_type:
            parts.append(f"[{self.platform_type.value}]")
        subsystems = []
        for name in [
            "perception",
            "compute",
            "power",
            "sensors",
            "actuators",
            "comms",
            "autonomy",
            "safety",
        ]:
            if getattr(self, name) is not None:
                subsystems.append(name)
        if subsystems:
            parts.append(f"({', '.join(subsystems)})")
        if self.tags:
            parts.append(f"tags={self.tags}")
        return " ".join(parts)

    def subsystem_names(self) -> list[str]:
        """Return names of defined (non-None) subsystems."""
        return [
            name
            for name in [
                "perception",
                "compute",
                "power",
                "sensors",
                "actuators",
                "comms",
                "autonomy",
                "safety",
            ]
            if getattr(self, name) is not None
        ]


# ── Path Utilities ─────────────────────────────────────────────────────────


# Map of subsystem name → Pydantic model class
SUBSYSTEM_MODELS: dict[str, type[BaseModel]] = {
    "perception": PerceptionSpec,
    "compute": ComputeSpec,
    "power": PowerSpec,
    "sensors": SensorSpec,
    "actuators": ActuatorSpec,
    "comms": CommsSpec,
    "autonomy": AutonomySpec,
    "safety": SafetySpec,
}


def parse_path(path: str) -> list[str]:
    """Parse a JSON pointer-style path into segments.

    Examples:
        "/perception/min_fps" → ["perception", "min_fps"]
        "/tags" → ["tags"]
        "/custom/my_field" → ["custom", "my_field"]
    """
    if not path.startswith("/"):
        from .exceptions import InvalidPathError

        raise InvalidPathError(path, "must start with /")
    parts = path.strip("/").split("/")
    if not parts or parts == [""]:
        from .exceptions import InvalidPathError

        raise InvalidPathError(path, "empty path")
    return parts


def get_at_path(spec: SystemSpec, path: str) -> Any:
    """Get a value from a spec at the given path."""
    parts = parse_path(path)
    obj: Any = spec
    for part in parts:
        if isinstance(obj, BaseModel):
            if not hasattr(obj, part):
                from .exceptions import InvalidPathError

                raise InvalidPathError(path, f"field '{part}' not found")
            obj = getattr(obj, part)
        elif isinstance(obj, dict):
            if part not in obj:
                from .exceptions import InvalidPathError

                raise InvalidPathError(path, f"key '{part}' not found")
            obj = obj[part]
        elif isinstance(obj, list):
            try:
                idx = int(part)
                obj = obj[idx]
            except (ValueError, IndexError):
                from .exceptions import InvalidPathError

                raise InvalidPathError(path, f"invalid list index '{part}'")
        else:
            from .exceptions import InvalidPathError

            raise InvalidPathError(path, f"cannot traverse into {type(obj).__name__}")
    return obj


def set_at_path(spec: SystemSpec, path: str, value: Any) -> SystemSpec:
    """Set a value on a spec at the given path, returning a new SystemSpec.

    Auto-creates intermediate subsystem models if they don't exist.
    Handles type coercion for enums and numeric fields.
    """
    parts = parse_path(path)
    data = spec.model_dump()
    _set_nested(data, parts, value, spec)
    return SystemSpec.model_validate(data)


def _set_nested(data: dict, parts: list[str], value: Any, spec: SystemSpec) -> None:
    """Recursively set a value in a nested dict, auto-creating subsystems."""
    key = parts[0]
    if len(parts) == 1:
        data[key] = _coerce_value(key, value, data)
    else:
        if key not in data or data[key] is None:
            data[key] = {}
        if not isinstance(data[key], dict):
            from .exceptions import InvalidPathError

            raise InvalidPathError("/" + "/".join(parts), f"'{key}' is not a traversable object")
        _set_nested(data[key], parts[1:], value, spec)


def _coerce_value(key: str, value: Any, context: dict) -> Any:
    """Coerce string values to appropriate types."""
    if isinstance(value, str):
        low = value.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low == "null" or low == "none":
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def delete_at_path(spec: SystemSpec, path: str) -> SystemSpec:
    """Delete a value from a spec at the given path, returning a new SystemSpec."""
    parts = parse_path(path)
    data = spec.model_dump()
    _delete_nested(data, parts)
    return SystemSpec.model_validate(data)


def _delete_nested(data: dict, parts: list[str]) -> None:
    """Recursively delete a value in a nested dict."""
    key = parts[0]
    if len(parts) == 1:
        if key in data:
            data[key] = None
        else:
            from .exceptions import InvalidPathError

            raise InvalidPathError("/" + key, "field not found")
    else:
        if key not in data or data[key] is None:
            from .exceptions import InvalidPathError

            raise InvalidPathError("/" + "/".join(parts), f"'{key}' does not exist")
        if not isinstance(data[key], dict):
            from .exceptions import InvalidPathError

            raise InvalidPathError("/" + "/".join(parts), f"'{key}' is not traversable")
        _delete_nested(data[key], parts[1:])
