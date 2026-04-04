"""Universal question bank for registry-driven qualification.

These questions cover all 5 tangibility dimensions and apply to ANY embodied
AI platform. When the platform registry matches a platform that has no
hand-crafted DomainTemplate, these questions are used instead.

Each question's implications map to the same spec fields that the tangibility
assessment checks, ensuring that answering all questions produces a tangible goal.
"""

from __future__ import annotations

from typing import Any

from embodied_ai_architect.qualification.models import Question, QuestionType


def build_universal_questions(
    platform_data: dict[str, Any] | None = None,
) -> list[Question]:
    """Build universal questions, optionally customized with platform defaults.

    Args:
        platform_data: A platform definition dict from the registry. Used to
            pre-fill defaults and customize option lists.

    Returns:
        List of Question objects covering all 5 tangibility dimensions.
    """
    attrs = (platform_data or {}).get("attributes", {})
    classification = (platform_data or {}).get("classification", {})
    implications_data = (platform_data or {}).get("implications", {})

    # Extract typical values for defaults
    power_typical = _get_typical(attrs, "power_watts")
    latency_typical = _get_typical(attrs, "latency_ms")

    # Build environment options from platform classification
    env_list = classification.get("environment", [])
    if isinstance(env_list, str):
        env_list = [env_list]

    env_options = _ENVIRONMENT_OPTIONS.copy()
    # If platform specifies environments, put them first
    if env_list:
        # Normalize platform environments to match our option keys
        for env in env_list:
            env_key = env.lower().replace(" ", "_").replace("(", "").replace(")", "")
            if env_key not in env_options:
                env_options[env_key] = env

    # Build perception task options from platform implications
    detection_classes = (implications_data.get("perception", {}) or {}).get("detection_classes", [])

    questions = [
        # --- Dimension: perception ---
        Question(
            id="perception_tasks",
            dimension="perception",
            text="What perception tasks does this system need to perform?",
            question_type=QuestionType.MULTI_CHOICE,
            options=[
                "object_detection",
                "classification",
                "tracking",
                "segmentation",
                "pose_estimation",
                "depth_estimation",
                "anomaly_detection",
                "ocr_text_reading",
                "face_recognition",
                "slam",
                "visual_odometry",
            ],
            option_descriptions={
                "object_detection": "Detect and localize objects (bounding boxes)",
                "classification": "Classify images or regions into categories",
                "tracking": "Track objects across video frames",
                "segmentation": "Pixel-level object boundaries (instance/semantic)",
                "pose_estimation": "Estimate object or human pose (keypoints)",
                "depth_estimation": "Estimate depth from mono/stereo cameras",
                "anomaly_detection": "Detect deviations from normal patterns",
                "ocr_text_reading": "Read text, barcodes, labels",
                "face_recognition": "Identify or verify faces",
                "slam": "Simultaneous localization and mapping",
                "visual_odometry": "Estimate motion from camera frames",
            },
            min_selections=1,
            allow_custom=True,
            custom_prompt=(
                "Describe the perception task: what sensor input, what output, "
                "and what latency/accuracy is needed."
            ),
            explanation=(
                "Each perception task requires specific model architectures, "
                "compute budgets, and sensor configurations. This determines "
                "the inference pipeline design."
            ),
            implications={
                "object_detection": {
                    "perception.detection_classes": detection_classes or ["object"],
                    "perception.max_latency_ms": latency_typical or 30,
                    "perception.min_fps": 15,
                },
                "classification": {
                    "perception.max_latency_ms": latency_typical or 50,
                    "perception.min_fps": 10,
                },
                "tracking": {
                    "perception.tracking": True,
                    "perception.min_fps": 20,
                },
                "segmentation": {
                    "perception.max_latency_ms": latency_typical or 50,
                },
                "slam": {
                    "perception.camera_types": ["stereo"],
                    "perception.max_latency_ms": 50,
                    "sensors.modalities": ["imu"],
                    "autonomy.navigation": "slam",
                },
                "visual_odometry": {
                    "perception.camera_types": ["stereo"],
                    "sensors.modalities": ["imu"],
                    "perception.max_latency_ms": 30,
                },
                "anomaly_detection": {
                    "perception.max_latency_ms": latency_typical or 100,
                },
                "ocr_text_reading": {
                    "perception.max_latency_ms": 200,
                },
                "face_recognition": {
                    "perception.detection_classes": ["face"],
                    "perception.max_latency_ms": 100,
                },
                "pose_estimation": {
                    "perception.max_latency_ms": 50,
                    "perception.min_fps": 15,
                },
                "depth_estimation": {
                    "perception.camera_types": ["stereo"],
                    "perception.max_latency_ms": 50,
                },
            },
        ),
        # --- Dimension: control ---
        Question(
            id="control_output",
            dimension="control",
            text="What does the perception output drive?",
            question_type=QuestionType.SINGLE_CHOICE,
            options=[
                "real_time_controller",
                "path_planner",
                "alerting_system",
                "data_logging",
                "human_display",
                "actuator_command",
            ],
            option_descriptions={
                "real_time_controller": (
                    "Direct input to a real-time controller (flight, motor, servo) "
                    "— latency-critical, 50-1000Hz"
                ),
                "path_planner": "Feed local/global path planning — moderate latency, 5-20Hz",
                "alerting_system": "Trigger alerts, alarms, or notifications — tolerant latency",
                "data_logging": "Record data for offline analysis — no real-time requirement",
                "human_display": "Show results to a human operator — moderate latency",
                "actuator_command": (
                    "Directly command an actuator (gripper, valve, nozzle) — "
                    "latency depends on process"
                ),
            },
            explanation=(
                "The downstream consumer determines the latency budget. A flight "
                "controller at 50Hz needs perception within 20ms. An alerting "
                "system can tolerate seconds."
            ),
            implications={
                "real_time_controller": {
                    "actuators.control_rate_hz": 50,
                    "autonomy.decision_rate_hz": 50,
                },
                "path_planner": {
                    "autonomy.planning": "path_planning",
                    "autonomy.decision_rate_hz": 10,
                },
                "alerting_system": {
                    "autonomy.decision_rate_hz": 1,
                },
                "data_logging": {
                    "autonomy.decision_rate_hz": 1,
                },
                "human_display": {
                    "autonomy.decision_rate_hz": 5,
                },
                "actuator_command": {
                    "actuators.control_rate_hz": 20,
                    "autonomy.decision_rate_hz": 20,
                },
            },
        ),
        # --- Dimension: power ---
        Question(
            id="power_budget",
            dimension="power",
            text="What is the compute power budget (watts)?",
            question_type=QuestionType.NUMERIC,
            numeric_min=0.1,
            numeric_max=50000,
            numeric_unit="watts",
            default=str(power_typical) if power_typical else None,
            explanation=(
                "The power available for the compute subsystem. This constrains "
                "which SoCs and accelerators are feasible. Battery-powered systems "
                "are typically 3-15W; mains-powered can be 50-300W."
            ),
            implications={},  # Handled specially — numeric value → spec field
        ),
        # --- Dimension: environment ---
        Question(
            id="deployment_environment",
            dimension="environment",
            text="What is the primary operating environment?",
            question_type=QuestionType.SINGLE_CHOICE,
            options=list(env_options.keys()),
            option_descriptions=env_options,
            explanation=(
                "Environment determines sensor modalities, environmental protection "
                "(IP rating), thermal constraints, and regulatory requirements."
            ),
            implications={
                "indoor_structured": {
                    "sensors.environmental_rating": "indoor",
                },
                "indoor_unstructured": {
                    "sensors.environmental_rating": "indoor",
                },
                "indoor_cleanroom": {
                    "sensors.environmental_rating": "indoor",
                },
                "indoor_sterile": {
                    "sensors.environmental_rating": "indoor",
                },
                "indoor_lab": {
                    "sensors.environmental_rating": "indoor",
                },
                "outdoor_urban": {
                    "sensors.environmental_rating": "outdoor",
                },
                "outdoor_rural": {
                    "sensors.environmental_rating": "outdoor",
                },
                "outdoor_extreme": {
                    "sensors.environmental_rating": "ip67",
                },
                "hazardous": {
                    "sensors.environmental_rating": "ip67",
                },
            },
        ),
        # --- Dimension: perception (latency refinement) ---
        Question(
            id="latency_requirement",
            dimension="perception",
            text="What is the maximum acceptable perception latency (ms)?",
            question_type=QuestionType.NUMERIC,
            numeric_min=1,
            numeric_max=10000,
            numeric_unit="ms",
            default=str(int(latency_typical)) if latency_typical else None,
            required=False,
            explanation=(
                "End-to-end latency from sensor input to perception output. "
                "Real-time control needs <20ms; analytics can tolerate 100-500ms."
            ),
            implications={},  # Handled specially — numeric value → spec field
        ),
    ]

    return questions


def apply_numeric_answer(spec_fields: dict, question_id: str, value: float) -> None:
    """Apply a numeric answer to the spec fields.

    Numeric questions don't use the implications dict — the raw value
    goes directly into the appropriate spec field.
    """
    _NUMERIC_FIELD_MAP = {
        "power_budget": "power.compute_power_watts",
        "latency_requirement": "perception.max_latency_ms",
    }
    field_path = _NUMERIC_FIELD_MAP.get(question_id)
    if not field_path:
        return

    parts = field_path.split(".")
    d = spec_fields
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

_ENVIRONMENT_OPTIONS: dict[str, str] = {
    "indoor_structured": "Warehouse, factory, hospital — known map, controlled lighting",
    "indoor_unstructured": "Home, office — unknown layout, dynamic occupants",
    "indoor_cleanroom": "ISO cleanroom — semiconductor, pharma",
    "indoor_sterile": "Operating room, BSL lab — aseptic, biocontainment",
    "indoor_lab": "Wet lab, dry lab — benchtop, chemical fume hoods",
    "outdoor_urban": "Streets, sidewalks, buildings",
    "outdoor_rural": "Fields, forests, open terrain",
    "outdoor_extreme": "Desert, arctic, high altitude, maritime",
    "hazardous": "Chemical plant, radiation zone, fire, explosive atmosphere",
}


def _get_typical(attrs: dict, key: str) -> float | None:
    """Extract the 'typical' value from an attribute range dict."""
    val = attrs.get(key)
    if isinstance(val, dict):
        return val.get("typical")
    return None
