"""LangGraph nodes for perception operators."""

from __future__ import annotations

from typing import Any, Callable, Optional

from embodied_ai_architect.graphs.state import EmbodiedPipelineState, PipelineStage
from embodied_ai_architect.graphs.utils import NodeTimer, is_stage_enabled
from embodied_ai_architect.operators import create_operator, Operator


def create_preprocess_node(
    execution_target: str = "cpu",
    config: Optional[dict[str, Any]] = None,
) -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create image preprocessing node.

    Wraps ImagePreprocessor operator for LangGraph execution.

    Args:
        execution_target: Hardware target (cpu, gpu)
        config: Operator configuration overrides

    Returns:
        Node function for LangGraph
    """
    operator: Optional[Operator] = None

    def preprocess_node(state: EmbodiedPipelineState) -> dict:
        nonlocal operator

        # Skip if not enabled
        if not is_stage_enabled(state, PipelineStage.PREPROCESS):
            return {"next_stage": PipelineStage.DETECT.value}

        # Get input image
        frame = state.get("frame", {})
        image = frame.get("image")

        if image is None:
            return {
                "next_stage": PipelineStage.ERROR.value,
                "errors": state.get("errors", []) + ["preprocess: No input image"],
            }

        # Lazy initialization
        if operator is None:
            merged_config = {**(config or {})}
            # Allow per-run config overrides
            merged_config.update(state.get("operator_configs", {}).get("preprocess", {}))
            target = state.get("execution_target", execution_target)
            operator = create_operator("image_preprocessor", merged_config, target)

        # Execute with timing
        with NodeTimer(state, "preprocess") as timer:
            try:
                result = operator.process({"image": image})
            except Exception as e:
                return {
                    "next_stage": PipelineStage.ERROR.value,
                    "errors": state.get("errors", []) + [f"preprocess: {e}"],
                    "timing": timer.timing,
                }

        return {
            "preprocessed": result,
            "next_stage": PipelineStage.DETECT.value,
            "timing": timer.timing,
        }

    return preprocess_node


def create_detect_node(
    variant: str = "s",
    execution_target: str = "cpu",
    config: Optional[dict[str, Any]] = None,
) -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create YOLO detection node.

    Wraps YOLOv8ONNX operator for LangGraph execution.

    Args:
        variant: YOLO model variant (n, s, m, l, x)
        execution_target: Hardware target (cpu, gpu, npu)
        config: Operator configuration overrides

    Returns:
        Node function for LangGraph
    """
    operator: Optional[Operator] = None

    def detect_node(state: EmbodiedPipelineState) -> dict:
        nonlocal operator

        # Skip if not enabled
        if not is_stage_enabled(state, PipelineStage.DETECT):
            return {"next_stage": PipelineStage.TRACK.value}

        # Get preprocessed image
        preprocessed = state.get("preprocessed", {})
        image = preprocessed.get("processed")

        # Fallback to raw frame if no preprocessing
        if image is None:
            frame = state.get("frame", {})
            image = frame.get("image")

        if image is None:
            return {
                "next_stage": PipelineStage.ERROR.value,
                "errors": state.get("errors", []) + ["detect: No input image"],
            }

        # Lazy initialization
        if operator is None:
            merged_config = {**(config or {})}
            merged_config.update(state.get("operator_configs", {}).get("detect", {}))
            target = state.get("execution_target", execution_target)
            operator = create_operator(f"yolo_detector_{variant}", merged_config, target)

        # Execute with timing
        with NodeTimer(state, "detect") as timer:
            try:
                result = operator.process({"image": image})
            except Exception as e:
                return {
                    "next_stage": PipelineStage.ERROR.value,
                    "errors": state.get("errors", []) + [f"detect: {e}"],
                    "timing": timer.timing,
                }

        return {
            "detections": result,
            "next_stage": PipelineStage.TRACK.value,
            "timing": timer.timing,
        }

    return detect_node


def create_track_node(
    config: Optional[dict[str, Any]] = None,
) -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create ByteTrack tracking node.

    Wraps ByteTrack operator for LangGraph execution.

    Args:
        config: Operator configuration overrides

    Returns:
        Node function for LangGraph
    """
    operator: Optional[Operator] = None

    def track_node(state: EmbodiedPipelineState) -> dict:
        nonlocal operator

        # Skip if not enabled
        if not is_stage_enabled(state, PipelineStage.TRACK):
            return {"next_stage": PipelineStage.SCENE_GRAPH.value}

        # Get detections
        detections = state.get("detections", {})
        detection_list = detections.get("detections", [])

        # Skip tracking if no detections (not an error)
        if not detection_list:
            return {
                "tracks": {"tracks": []},
                "next_stage": PipelineStage.SCENE_GRAPH.value,
                "timing": {**state.get("timing", {}), "track": 0.0},
            }

        # Lazy initialization
        if operator is None:
            merged_config = {**(config or {})}
            merged_config.update(state.get("operator_configs", {}).get("track", {}))
            operator = create_operator("bytetrack", merged_config, "cpu")

        # Execute with timing
        with NodeTimer(state, "track") as timer:
            try:
                result = operator.process({"detections": detection_list})
            except Exception as e:
                return {
                    "next_stage": PipelineStage.ERROR.value,
                    "errors": state.get("errors", []) + [f"track: {e}"],
                    "timing": timer.timing,
                }

        return {
            "tracks": result,
            "next_stage": PipelineStage.SCENE_GRAPH.value,
            "timing": timer.timing,
        }

    return track_node


def create_scene_graph_node(
    config: Optional[dict[str, Any]] = None,
) -> Callable[[EmbodiedPipelineState], dict]:
    """
    Create scene graph management node.

    Wraps SceneGraphManager operator for LangGraph execution.
    Converts 2D tracks to 3D scene representation.

    Args:
        config: Operator configuration overrides

    Returns:
        Node function for LangGraph
    """
    operator: Optional[Operator] = None

    def scene_graph_node(state: EmbodiedPipelineState) -> dict:
        nonlocal operator

        # Skip if not enabled
        if not is_stage_enabled(state, PipelineStage.SCENE_GRAPH):
            return {"next_stage": PipelineStage.COMPLETE.value}

        # Get tracks
        tracks = state.get("tracks", {})
        track_list = tracks.get("tracks", [])

        # Get frame dimensions for depth estimation
        frame = state.get("frame", {})
        image = frame.get("image")
        if image is not None:
            image_size = (image.shape[0], image.shape[1])  # (h, w)
        else:
            image_size = (480, 640)  # Default

        # Skip if no tracks (not an error)
        if not track_list:
            return {
                "scene_objects": {"objects": [], "obstacles": []},
                "next_stage": PipelineStage.COMPLETE.value,
                "timing": {**state.get("timing", {}), "scene_graph": 0.0},
            }

        # Lazy initialization
        if operator is None:
            merged_config = {**(config or {})}
            merged_config.update(state.get("operator_configs", {}).get("scene_graph", {}))
            operator = create_operator("scene_graph_manager", merged_config, "cpu")

        # Execute with timing
        with NodeTimer(state, "scene_graph") as timer:
            try:
                result = operator.process(
                    {
                        "tracks": track_list,
                        "depth_map": None,  # Optional depth map
                        "image_size": image_size,
                    }
                )
            except Exception as e:
                return {
                    "next_stage": PipelineStage.ERROR.value,
                    "errors": state.get("errors", []) + [f"scene_graph: {e}"],
                    "timing": timer.timing,
                }

        return {
            "scene_objects": result,
            "next_stage": PipelineStage.COMPLETE.value,
            "timing": timer.timing,
        }

    return scene_graph_node
