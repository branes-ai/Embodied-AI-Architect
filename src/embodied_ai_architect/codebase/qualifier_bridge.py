"""Codebase scan → goal qualification bridge.

Maps a `ScanResult` (and optionally a `CodebaseAnalysisResult`) into pre-filled
answers for the `GoalQualifier`. This lets the user run a single command:

    branes codebase qualify /path/to/drone_app

and have the platform, perception tasks, control output, and frameworks
auto-detected from the codebase. Only the genuine gaps need interactive Q&A.

Detection is **scan-only** (no LLM call needed) — uses dependency lists, file
patterns, and ML model presence. Fast, deterministic, no API key required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from embodied_ai_architect.codebase.models import CodebaseAnalysisResult, ScanResult

# ---------------------------------------------------------------------------
# Detection rules — keyed by domain
# ---------------------------------------------------------------------------

# Dependency name → domain hint. First match wins.
_DOMAIN_HINTS: dict[str, str] = {
    # Drone indicators
    "mavros": "drone",
    "mavlink": "drone",
    "pymavlink": "drone",
    "px4": "drone",
    "ardupilot": "drone",
    "dronekit": "drone",
    "olympe": "drone",
    "djitellopy": "drone",
    "betaflight": "drone",
    # Robot arm indicators
    "moveit": "robot_arm",
    "moveit_msgs": "robot_arm",
    "franka": "robot_arm",
    "franka_ros": "robot_arm",
    "panda_py": "robot_arm",
    "ur_rtde": "robot_arm",
    "rtde_control": "robot_arm",
    "kuka": "robot_arm",
    "abb_libegm": "robot_arm",
    "robotiq": "robot_arm",
    "pybullet": "robot_arm",  # often used for arm sim
    # UGV / mobile robot indicators
    "nav2": "ugv",
    "nav2_msgs": "ugv",
    "turtlebot3": "ugv",
    "turtlebot4": "ugv",
    "amcl": "ugv",
    "slam_toolbox": "ugv",
    "carla": "ugv",
    "lgsvl": "ugv",
    "autoware": "ugv",
}

# Dependency name → list of perception_tasks for the matching domain.
# Tasks are domain-specific question option values.
_PERCEPTION_HINTS: dict[str, list[str]] = {
    # Object detection
    "ultralytics": ["object_detection"],
    "yolov5": ["object_detection"],
    "yolov8": ["object_detection"],
    "yolov7": ["object_detection"],
    "detectron2": ["object_detection"],
    "mmdetection": ["object_detection"],
    "torchvision": ["object_detection"],
    # SLAM / VIO
    "orb_slam3": ["slam"],
    "orb_slam2": ["slam"],
    "rtabmap": ["slam"],
    "rtabmap_ros": ["slam"],
    "open3d": ["slam"],
    "openvslam": ["slam"],
    "vins": ["visual_odometry"],
    "vins_fusion": ["visual_odometry"],
    "vins_mono": ["visual_odometry"],
    "okvis": ["visual_odometry"],
    "kimera": ["visual_odometry", "slam"],
    # Person / face
    "mediapipe": ["person_detection"],
    "face_recognition": ["person_detection"],
    "deepface": ["person_detection"],
    "openpose": ["person_detection"],
    # Tracking
    "deep_sort": ["tracking"],
    "deep_sort_realtime": ["tracking"],
    "bytetrack": ["tracking"],
    "norfair": ["tracking"],
    "supervision": ["tracking"],
    # Mapping / terrain
    "octomap": ["terrain_mapping"],
    "elevation_mapping": ["terrain_mapping"],
}

# Dependency name → control output hint.
_CONTROL_HINTS: dict[str, str] = {
    "simple_pid": "flight_controller",
    "control": "flight_controller",
    "px4_msgs": "flight_controller",
    "mavros_msgs": "flight_controller",
    "nav2_msgs": "path_planner",
    "moveit_msgs": "trajectory_to_joints",
    "geometry_msgs": "path_planner",
    "trajectory_msgs": "trajectory_to_joints",
}

# Dependency name → ML framework family (used to flag inference-heavy projects).
_ML_FRAMEWORKS: set[str] = {
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "tflite_runtime",
    "tflite",
    "onnxruntime",
    "onnxruntime-gpu",
    "tensorrt",
    "openvino",
    "coremltools",
    "jax",
    "flax",
    "mxnet",
}


@dataclass
class DetectionReport:
    """Human-readable report of what the bridge auto-detected from the codebase."""

    domain: Optional[str] = None
    domain_evidence: list[str] = field(default_factory=list)
    perception_tasks: list[str] = field(default_factory=list)
    perception_evidence: list[str] = field(default_factory=list)
    control_output: list[str] = field(default_factory=list)
    control_evidence: list[str] = field(default_factory=list)
    ml_frameworks: list[str] = field(default_factory=list)
    has_ml_models: bool = False
    confidence: str = "low"  # low | medium | high

    def has_any_signal(self) -> bool:
        return bool(
            self.domain
            or self.perception_tasks
            or self.control_output
            or self.ml_frameworks
            or self.has_ml_models
        )


@dataclass
class BridgeResult:
    """Output of the bridge: detected domain + answers to pre-fill + report."""

    domain: Optional[str]
    prefilled_answers: dict[str, list[str] | str]
    report: DetectionReport


def codebase_to_qualification(
    scan: ScanResult,
    analysis: Optional[CodebaseAnalysisResult] = None,
) -> BridgeResult:
    """Map a codebase scan into pre-filled qualifier answers.

    Args:
        scan: Required ScanResult from CodebaseScanner.scan()
        analysis: Optional richer analysis from CodeAnalyzer.analyze()
            (provides kernel-level signals like control_loop kernels)

    Returns:
        BridgeResult with detected domain, answers dict for prefill_answers(),
        and a DetectionReport explaining what was found.
    """
    deps = _normalize_deps(scan.dependencies)
    report = DetectionReport()

    # 1. Domain detection
    domain = _detect_domain(deps, report)

    # 2. Perception tasks
    perception_tasks = _detect_perception_tasks(deps, scan, analysis, report)

    # 3. Control output
    control_outputs = _detect_control_output(deps, analysis, report)

    # 4. ML frameworks
    report.ml_frameworks = sorted([d for d in deps if d in _ML_FRAMEWORKS])
    report.has_ml_models = bool(scan.ml_models)

    # 5. Compute confidence
    report.confidence = _compute_confidence(report)

    # 6. Build prefilled answers dict (domain-specific question_ids)
    prefilled: dict[str, list[str] | str] = {}
    if perception_tasks:
        prefilled["perception_tasks"] = perception_tasks
    if control_outputs:
        prefilled["control_output"] = control_outputs

    return BridgeResult(domain=domain, prefilled_answers=prefilled, report=report)


# ---------------------------------------------------------------------------
# Internal detection helpers
# ---------------------------------------------------------------------------


def _normalize_deps(deps: list[str]) -> set[str]:
    """Lowercase and strip version specifiers from dependency names."""
    normalized: set[str] = set()
    for dep in deps:
        # Strip version specifiers like "torch>=2.0" or "torch==1.13.0"
        name = dep.lower().strip()
        for sep in [">=", "<=", "==", ">", "<", "~=", "!=", "[", " "]:
            if sep in name:
                name = name.split(sep)[0]
        name = name.strip()
        if name:
            normalized.add(name)
    return normalized


def _detect_domain(deps: set[str], report: DetectionReport) -> Optional[str]:
    """Find the first matching domain hint and record evidence."""
    # Count matches per domain to handle ROS-based projects that match multiple
    domain_scores: dict[str, list[str]] = {}
    for dep in deps:
        if dep in _DOMAIN_HINTS:
            d = _DOMAIN_HINTS[dep]
            domain_scores.setdefault(d, []).append(dep)

    if not domain_scores:
        return None

    # Pick the domain with the most evidence
    best_domain = max(domain_scores.keys(), key=lambda d: len(domain_scores[d]))
    report.domain = best_domain
    report.domain_evidence = sorted(domain_scores[best_domain])
    return best_domain


def _detect_perception_tasks(
    deps: set[str],
    scan: ScanResult,
    analysis: Optional[CodebaseAnalysisResult],
    report: DetectionReport,
) -> list[str]:
    """Detect perception tasks from dependencies and ML model presence."""
    tasks: set[str] = set()
    evidence: list[str] = []

    for dep in deps:
        if dep in _PERCEPTION_HINTS:
            for task in _PERCEPTION_HINTS[dep]:
                tasks.add(task)
            evidence.append(dep)

    # If we have ML models but no specific perception hint, default to object_detection
    if not tasks and scan.ml_models:
        tasks.add("object_detection")
        evidence.append(f"{len(scan.ml_models)} ML model file(s)")

    # Use analysis kernels if available for richer signal
    if analysis and analysis.kernels:
        for kernel in analysis.kernels:
            if kernel.kernel_type == "ml_inference" and not tasks:
                tasks.add("object_detection")
                evidence.append(f"ml_inference kernel: {kernel.name}")
            elif kernel.kernel_type == "image_processing" and "object_detection" not in tasks:
                tasks.add("object_detection")
                evidence.append(f"image_processing kernel: {kernel.name}")

    report.perception_tasks = sorted(tasks)
    report.perception_evidence = sorted(set(evidence))
    return sorted(tasks)


def _detect_control_output(
    deps: set[str],
    analysis: Optional[CodebaseAnalysisResult],
    report: DetectionReport,
) -> list[str]:
    """Detect control outputs from dependencies and control_loop kernels."""
    outputs: set[str] = set()
    evidence: list[str] = []

    for dep in deps:
        if dep in _CONTROL_HINTS:
            outputs.add(_CONTROL_HINTS[dep])
            evidence.append(dep)

    # Control loop kernels signal motor/controller output
    if analysis and analysis.kernels:
        has_control_loop = any(k.kernel_type == "control_loop" for k in analysis.kernels)
        if has_control_loop and not outputs:
            outputs.add("flight_controller")
            evidence.append("control_loop kernel detected")

    report.control_output = sorted(outputs)
    report.control_evidence = sorted(set(evidence))
    return sorted(outputs)


def _compute_confidence(report: DetectionReport) -> str:
    """Compute overall confidence based on signal strength."""
    score = 0
    if report.domain and len(report.domain_evidence) >= 2:
        score += 2
    elif report.domain:
        score += 1
    if len(report.perception_tasks) >= 1 and report.perception_evidence:
        score += 1
    if len(report.control_output) >= 1 and report.control_evidence:
        score += 1
    if report.has_ml_models:
        score += 1

    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"
