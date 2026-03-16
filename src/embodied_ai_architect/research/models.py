"""Data models for mission decomposition.

Defines the structured output of the MissionDecomposer: a MissionPlan that
contains sub-capabilities, an operator graph, and a design space definition.
These models bridge the gap between a natural-language mission description
and a formal optimization problem.

Usage:
    from embodied_ai_architect.research.models import MissionPlan, SubCapability

    plan = MissionPlan(
        mission="Drone perception for GPS-denied navigation at 5 m/s",
        platform="drone",
        sub_capabilities=[...],
        operator_graph=OperatorGraph(...),
        design_space=DesignSpaceDefinition(...),
    )
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class OperatorType(str, Enum):
    """Types of operators in an embodied AI pipeline."""

    PREPROCESS = "preprocess"
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    TRACKING = "tracking"
    DEPTH_ESTIMATION = "depth_estimation"
    POSE_ESTIMATION = "pose_estimation"
    SCENE_GRAPH = "scene_graph"
    SENSOR_FUSION = "sensor_fusion"
    STATE_ESTIMATION = "state_estimation"
    PATH_PLANNING = "path_planning"
    CONTROL = "control"
    REASONING = "reasoning"
    COMMUNICATION = "communication"
    CUSTOM = "custom"


class SubCapability(BaseModel):
    """A sub-capability required to achieve the mission."""

    name: str = Field(..., description="Capability name (e.g. 'obstacle_detection')")
    description: str = Field(default="", description="What this capability does")
    required: bool = Field(default=True, description="Whether this is mandatory")
    latency_budget_ms: Optional[float] = Field(
        default=None,
        description="Maximum allowable latency for this capability",
    )
    accuracy_requirement: Optional[str] = Field(
        default=None,
        description="Accuracy requirement description (e.g. '>90% mAP@50')",
    )
    rate_hz: Optional[float] = Field(
        default=None,
        description="Required update rate in Hz",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Names of capabilities this depends on",
    )


class OperatorSpec(BaseModel):
    """Specification for a pipeline operator."""

    name: str = Field(..., description="Operator name (e.g. 'yolo_detector')")
    operator_type: OperatorType = Field(..., description="Type classification")
    model_family: Optional[str] = Field(
        default=None,
        description="Suggested model family (e.g. 'yolov8')",
    )
    model_variant: Optional[str] = Field(
        default=None,
        description="Suggested model variant (e.g. 's')",
    )
    estimated_gflops: Optional[float] = Field(
        default=None,
        description="Estimated compute cost in GFLOPS",
    )
    latency_budget_ms: Optional[float] = Field(
        default=None,
        description="Allocated latency budget in ms",
    )
    rate_hz: Optional[float] = Field(
        default=None,
        description="Target execution rate in Hz",
    )
    input_resolution: Optional[str] = Field(
        default=None,
        description="Input dimensions (e.g. '640x640', '224x224')",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Suggested quantization (e.g. 'int8', 'fp16')",
    )
    capability: Optional[str] = Field(
        default=None,
        description="Which sub-capability this operator serves",
    )
    notes: str = Field(default="", description="Additional notes or rationale")


class OperatorEdge(BaseModel):
    """A directed edge in the operator graph."""

    source: str = Field(..., description="Source operator name")
    target: str = Field(..., description="Target operator name")
    data_type: str = Field(
        default="tensor",
        description="Type of data flowing (tensor, detections, tracks, etc.)",
    )
    conditional: bool = Field(
        default=False,
        description="Whether this edge is conditional (e.g. collision → evasion)",
    )
    condition: Optional[str] = Field(
        default=None,
        description="Condition for traversal (if conditional)",
    )


class OperatorGraph(BaseModel):
    """Directed graph of pipeline operators."""

    operators: list[OperatorSpec] = Field(default_factory=list)
    edges: list[OperatorEdge] = Field(default_factory=list)
    total_estimated_gflops: Optional[float] = Field(
        default=None,
        description="Sum of per-operator GFLOPS estimates",
    )
    end_to_end_latency_budget_ms: Optional[float] = Field(
        default=None,
        description="Total pipeline latency budget",
    )

    def get_operator(self, name: str) -> Optional[OperatorSpec]:
        """Look up an operator by name."""
        for op in self.operators:
            if op.name == name:
                return op
        return None

    def get_sources(self, operator_name: str) -> list[str]:
        """Get the names of operators that feed into the given operator."""
        return [e.source for e in self.edges if e.target == operator_name]

    def get_sinks(self, operator_name: str) -> list[str]:
        """Get the names of operators this operator feeds into."""
        return [e.target for e in self.edges if e.source == operator_name]

    def topological_order(self) -> list[str]:
        """Return operators in topological order (for sequential execution)."""
        in_degree: dict[str, int] = {op.name: 0 for op in self.operators}
        for edge in self.edges:
            if edge.target in in_degree:
                in_degree[edge.target] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for edge in self.edges:
                if edge.source == node and edge.target in in_degree:
                    in_degree[edge.target] -= 1
                    if in_degree[edge.target] == 0:
                        queue.append(edge.target)
        return order


class DesignObjective(BaseModel):
    """An objective for the optimization problem."""

    name: str = Field(..., description="Objective name (e.g. 'capability_per_watt')")
    direction: str = Field(
        default="minimize",
        description="Optimization direction: 'minimize' or 'maximize'",
    )
    weight: float = Field(
        default=1.0,
        description="Relative importance weight (for weighted scalarization)",
    )


class DesignConstraint(BaseModel):
    """A constraint for the optimization problem."""

    name: str = Field(..., description="Constraint name (e.g. 'max_power_watts')")
    value: float = Field(..., description="Constraint bound value")
    constraint_type: str = Field(
        default="upper",
        description="Bound type: 'upper' (max), 'lower' (min)",
    )
    unit: str = Field(default="", description="Unit (e.g. 'watts', 'ms', '%')")


class DesignSpaceDefinition(BaseModel):
    """Definition of the optimization problem derived from the mission.

    This is the bridge between mission decomposition and the MOO engine.
    It specifies what objectives to optimize, what constraints to satisfy,
    and what workload profile to use for evaluation.
    """

    objectives: list[DesignObjective] = Field(default_factory=list)
    constraints: list[DesignConstraint] = Field(default_factory=list)
    include_accuracy: bool = Field(
        default=True,
        description="Include accuracy/capability_per_watt in objectives",
    )
    workload_type: str = Field(
        default="detection",
        description="Primary workload type for accuracy estimation",
    )
    quantization_dtype: str = Field(
        default="fp32",
        description="Default quantization data type",
    )
    total_estimated_gflops: Optional[float] = Field(
        default=None,
        description="Total workload compute budget",
    )
    platform: str = Field(
        default="",
        description="Target platform for SWaP-C profiles (drone, vehicle, etc.)",
    )

    def to_constraints_dict(self) -> dict[str, Any]:
        """Convert to the dict format expected by create_soc_design_space()."""
        result: dict[str, Any] = {}
        for c in self.constraints:
            if c.constraint_type == "upper":
                result[f"max_{c.name}"] = c.value
            else:
                result[f"min_{c.name}"] = c.value
        result["include_accuracy"] = self.include_accuracy
        result["workload_type"] = self.workload_type
        result["quantization_dtype"] = self.quantization_dtype
        return result


class MissionPlan(BaseModel):
    """Complete mission decomposition output.

    Bridges natural-language mission description to a formal optimization
    problem. Produced by the MissionDecomposer, consumed by the optimization
    loop.
    """

    mission: str = Field(..., description="Original mission description")
    platform: str = Field(
        default="",
        description="Platform type (drone, ugv, vehicle, wearable, edge)",
    )
    mission_type: str = Field(
        default="",
        description="Mission classification (perception, navigation, inspection, etc.)",
    )
    environment: str = Field(
        default="",
        description="Operating environment (indoor, outdoor, gps-denied, etc.)",
    )
    sub_capabilities: list[SubCapability] = Field(
        default_factory=list,
        description="Required sub-capabilities",
    )
    operator_graph: OperatorGraph = Field(
        default_factory=OperatorGraph,
        description="Pipeline operator graph",
    )
    design_space: DesignSpaceDefinition = Field(
        default_factory=DesignSpaceDefinition,
        description="Optimization problem definition",
    )
    research_context_used: list[str] = Field(
        default_factory=list,
        description="Paths of research documents used during decomposition",
    )
    rationale: str = Field(
        default="",
        description="LLM's reasoning about the decomposition",
    )

    def summary(self) -> str:
        """Generate a human-readable summary of the plan."""
        lines = [
            f"Mission: {self.mission}",
            f"Platform: {self.platform or 'unspecified'}",
            f"Type: {self.mission_type or 'unspecified'}",
            f"Environment: {self.environment or 'unspecified'}",
            f"Sub-capabilities: {len(self.sub_capabilities)}",
        ]
        for cap in self.sub_capabilities:
            req = "required" if cap.required else "optional"
            rate = f" @ {cap.rate_hz}Hz" if cap.rate_hz else ""
            latency = f" <{cap.latency_budget_ms}ms" if cap.latency_budget_ms else ""
            lines.append(f"  - {cap.name} ({req}){rate}{latency}")

        lines.append(f"Operators: {len(self.operator_graph.operators)}")
        for op in self.operator_graph.operators:
            model = f" [{op.model_family}/{op.model_variant}]" if op.model_family else ""
            gflops = f" {op.estimated_gflops} GFLOPS" if op.estimated_gflops else ""
            lines.append(f"  - {op.name} ({op.operator_type.value}){model}{gflops}")

        lines.append(f"Constraints: {len(self.design_space.constraints)}")
        for c in self.design_space.constraints:
            lines.append(f"  - {c.name} {'<' if c.constraint_type == 'upper' else '>'} {c.value}")

        return "\n".join(lines)
