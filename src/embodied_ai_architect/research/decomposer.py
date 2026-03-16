"""Mission decomposer: NL mission → structured MissionPlan.

Uses the LLM with research library context to decompose a natural-language
mission description into a structured MissionPlan with sub-capabilities,
an operator graph, and a design space definition.

Works without an API key by falling back to a rule-based heuristic decomposer
for common mission types.

Usage:
    from embodied_ai_architect.research.decomposer import MissionDecomposer

    decomposer = MissionDecomposer()
    plan = decomposer.decompose("Drone perception for GPS-denied navigation at 5 m/s")
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from embodied_ai_architect.research.library import ResearchLibrary
from embodied_ai_architect.research.models import (
    DesignConstraint,
    DesignObjective,
    DesignSpaceDefinition,
    MissionPlan,
    OperatorEdge,
    OperatorGraph,
    OperatorSpec,
    OperatorType,
    SubCapability,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decomposition prompt
# ---------------------------------------------------------------------------

DECOMPOSITION_SYSTEM_PROMPT = """\
You are an embodied AI systems architect. Given a mission description, decompose it into:

1. **Platform**: drone, ugv, vehicle, wearable, edge, or other
2. **Mission type**: perception, navigation, inspection, surveillance, delivery, etc.
3. **Environment**: indoor, outdoor, gps-denied, urban, rural, etc.
4. **Sub-capabilities**: Required capabilities (e.g., obstacle_detection, tracking, SLAM)
5. **Operator graph**: Pipeline of compute operators with models, rates, and latency budgets
6. **Design constraints**: Power, latency, cost, weight, accuracy requirements

Use the research context provided to set realistic latency budgets, power constraints,
and model recommendations. Cite specific numbers from the research when available.

Respond with a JSON object matching this schema:
{
  "platform": "string",
  "mission_type": "string",
  "environment": "string",
  "sub_capabilities": [
    {
      "name": "string",
      "description": "string",
      "required": true,
      "latency_budget_ms": number or null,
      "accuracy_requirement": "string or null",
      "rate_hz": number or null,
      "depends_on": ["string"]
    }
  ],
  "operators": [
    {
      "name": "string",
      "operator_type": "detection|classification|tracking|...",
      "model_family": "string or null",
      "model_variant": "string or null",
      "estimated_gflops": number or null,
      "latency_budget_ms": number or null,
      "rate_hz": number or null,
      "input_resolution": "string or null",
      "quantization": "string or null",
      "capability": "string (which sub-capability this serves)",
      "notes": "string"
    }
  ],
  "edges": [
    {"source": "string", "target": "string", "data_type": "string"}
  ],
  "constraints": [
    {"name": "string", "value": number, "constraint_type": "upper|lower", "unit": "string"}
  ],
  "total_latency_budget_ms": number,
  "total_estimated_gflops": number,
  "workload_type": "detection|classification|segmentation",
  "quantization_dtype": "fp32|fp16|int8|int4",
  "rationale": "string explaining your reasoning"
}
"""


class MissionDecomposer:
    """Decomposes natural-language mission descriptions into MissionPlans.

    Uses the LLM (when available) enriched with research library context.
    Falls back to rule-based heuristics for common mission types when no
    LLM client is provided.
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        research_library: ResearchLibrary | None = None,
    ):
        self.llm_client = llm_client
        self.library = research_library or ResearchLibrary()

    def decompose(
        self,
        mission_description: str,
        max_context_tokens: int = 8000,
    ) -> MissionPlan:
        """Decompose a mission description into a structured plan.

        If an LLM client is available, uses Claude with research context.
        Otherwise falls back to rule-based heuristics.

        Args:
            mission_description: Natural-language mission description.
            max_context_tokens: Token budget for research context.

        Returns:
            Structured MissionPlan.
        """
        # Classify mission type for research retrieval
        mission_hints = self._extract_mission_hints(mission_description)

        # Retrieve relevant research
        docs = self.library.retrieve(
            tags=mission_hints["tags"],
            relevance="mission_decomposition",
            mission_type=mission_hints["platform"],
            max_results=8,
        )
        research_context = self.library.build_context_block(
            docs,
            max_tokens=max_context_tokens,
            include_frontmatter=True,
        )
        doc_paths = [d.path for d in docs]

        # Use LLM if available, otherwise fall back to heuristics
        if self.llm_client is not None:
            plan = self._decompose_with_llm(mission_description, research_context, doc_paths)
        else:
            plan = self._decompose_heuristic(mission_description, mission_hints)
            plan.research_context_used = doc_paths

        return plan

    def _extract_mission_hints(self, description: str) -> dict[str, Any]:
        """Extract hints from the mission description for retrieval."""
        desc_lower = description.lower()

        # Platform detection
        platform = ""
        platform_keywords = {
            "drone": ["drone", "uav", "quadcopter", "multirotor", "aerial"],
            "ugv": ["ugv", "ground vehicle", "rover", "robot", "wheeled"],
            "vehicle": ["car", "vehicle", "automotive", "adas", "driving"],
            "wearable": ["wearable", "glasses", "watch", "headset"],
            "edge": ["edge", "camera", "iot", "embedded"],
        }
        for plat, keywords in platform_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                platform = plat
                break

        # Mission type detection
        mission_type = ""
        type_keywords = {
            "perception": ["perception", "detect", "see", "vision", "recognize"],
            "navigation": ["navigate", "navigation", "gps-denied", "slam", "path"],
            "inspection": ["inspect", "inspection", "survey", "monitor"],
            "surveillance": ["surveillance", "security", "watch", "patrol"],
            "delivery": ["delivery", "transport", "carry", "pick"],
        }
        for mtype, keywords in type_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                mission_type = mtype
                break

        # Speed extraction
        speed_match = re.search(r"(\d+(?:\.\d+)?)\s*m/s", desc_lower)
        speed_ms = float(speed_match.group(1)) if speed_match else None

        # Power budget extraction
        power_match = re.search(r"(\d+(?:\.\d+)?)\s*[wW](?:att)?", desc_lower)
        power_w = float(power_match.group(1)) if power_match else None

        # Build tags for retrieval
        tags = []
        if platform:
            tags.append(platform)
        if mission_type:
            tags.append(mission_type)
        # Add common related tags
        if "perception" in desc_lower or "detect" in desc_lower:
            tags.extend(["detection", "inference", "edge"])
        if "track" in desc_lower:
            tags.append("tracking")
        if "slam" in desc_lower or "gps-denied" in desc_lower:
            tags.append("slam")
        if "real-time" in desc_lower or speed_ms:
            tags.append("real-time")

        return {
            "platform": platform,
            "mission_type": mission_type,
            "speed_ms": speed_ms,
            "power_w": power_w,
            "tags": tags,
        }

    def _decompose_with_llm(
        self,
        description: str,
        research_context: str,
        doc_paths: list[str],
    ) -> MissionPlan:
        """Decompose using LLM with research context."""
        prompt = f"""Decompose this mission into a structured plan:

{description}

{research_context}

Respond with JSON only, no markdown fences."""

        try:
            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                system=DECOMPOSITION_SYSTEM_PROMPT,
            )
            # Parse the JSON response
            text = response.text if hasattr(response, "text") else str(response)
            # Strip markdown code fences if present
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*$", "", text)
            data = json.loads(text)
            return self._parse_llm_response(data, description, doc_paths)
        except Exception as e:
            logger.warning("LLM decomposition failed, falling back to heuristic: %s", e)
            hints = self._extract_mission_hints(description)
            plan = self._decompose_heuristic(description, hints)
            plan.research_context_used = doc_paths
            return plan

    def _parse_llm_response(
        self,
        data: dict[str, Any],
        description: str,
        doc_paths: list[str],
    ) -> MissionPlan:
        """Parse LLM JSON response into a MissionPlan."""
        # Parse sub-capabilities
        sub_caps = []
        for cap_data in data.get("sub_capabilities", []):
            sub_caps.append(SubCapability(**cap_data))

        # Parse operators
        operators = []
        for op_data in data.get("operators", []):
            op_type = op_data.get("operator_type", "custom")
            try:
                op_type_enum = OperatorType(op_type)
            except ValueError:
                op_type_enum = OperatorType.CUSTOM
            op_data["operator_type"] = op_type_enum
            operators.append(OperatorSpec(**op_data))

        # Parse edges
        edges = []
        for edge_data in data.get("edges", []):
            edges.append(OperatorEdge(**edge_data))

        # Parse constraints
        constraints = []
        for c_data in data.get("constraints", []):
            constraints.append(DesignConstraint(**c_data))

        # Compute total GFLOPS
        total_gflops = data.get("total_estimated_gflops")
        if total_gflops is None:
            total_gflops = sum(op.estimated_gflops for op in operators if op.estimated_gflops)

        operator_graph = OperatorGraph(
            operators=operators,
            edges=edges,
            total_estimated_gflops=total_gflops,
            end_to_end_latency_budget_ms=data.get("total_latency_budget_ms"),
        )

        design_space = DesignSpaceDefinition(
            objectives=[
                DesignObjective(name="power_watts", direction="minimize"),
                DesignObjective(name="latency_ms", direction="minimize"),
                DesignObjective(name="cost_usd", direction="minimize"),
                DesignObjective(name="accuracy_percent", direction="maximize"),
                DesignObjective(name="capability_per_watt", direction="maximize", weight=2.0),
            ],
            constraints=constraints,
            include_accuracy=True,
            workload_type=data.get("workload_type", "detection"),
            quantization_dtype=data.get("quantization_dtype", "fp32"),
            total_estimated_gflops=total_gflops,
            platform=data.get("platform", ""),
        )

        return MissionPlan(
            mission=description,
            platform=data.get("platform", ""),
            mission_type=data.get("mission_type", ""),
            environment=data.get("environment", ""),
            sub_capabilities=sub_caps,
            operator_graph=operator_graph,
            design_space=design_space,
            research_context_used=doc_paths,
            rationale=data.get("rationale", ""),
        )

    def _decompose_heuristic(
        self,
        description: str,
        hints: dict[str, Any],
    ) -> MissionPlan:
        """Rule-based decomposition for common mission types.

        Used when no LLM is available. Produces reasonable defaults based on
        platform type and mission keywords.
        """
        platform = hints.get("platform", "edge")
        mission_type = hints.get("mission_type", "perception")
        speed_ms = hints.get("speed_ms")
        power_w = hints.get("power_w")

        # Determine latency budget from speed
        if speed_ms and speed_ms >= 5:
            e2e_latency_ms = 33.3  # 30 FPS
            frame_rate = 30.0
        elif speed_ms and speed_ms >= 2:
            e2e_latency_ms = 66.6  # 15 FPS
            frame_rate = 15.0
        else:
            e2e_latency_ms = 100.0  # 10 FPS
            frame_rate = 10.0

        # Determine power budget from platform
        if power_w is None:
            power_budgets = {
                "drone": 10.0,
                "ugv": 30.0,
                "vehicle": 75.0,
                "wearable": 1.0,
                "edge": 15.0,
            }
            power_w = power_budgets.get(platform, 15.0)

        # Build sub-capabilities
        sub_caps = self._build_standard_capabilities(mission_type, frame_rate, e2e_latency_ms)

        # Build operator graph
        operator_graph = self._build_standard_operator_graph(
            mission_type, frame_rate, e2e_latency_ms
        )

        # Build constraints
        constraints = [
            DesignConstraint(
                name="power_watts", value=power_w, constraint_type="upper", unit="watts"
            ),
            DesignConstraint(
                name="latency_ms",
                value=e2e_latency_ms,
                constraint_type="upper",
                unit="ms",
            ),
            DesignConstraint(name="cost_usd", value=50.0, constraint_type="upper", unit="usd"),
        ]

        design_space = DesignSpaceDefinition(
            objectives=[
                DesignObjective(name="power_watts", direction="minimize"),
                DesignObjective(name="latency_ms", direction="minimize"),
                DesignObjective(name="cost_usd", direction="minimize"),
                DesignObjective(name="accuracy_percent", direction="maximize"),
                DesignObjective(name="capability_per_watt", direction="maximize", weight=2.0),
            ],
            constraints=constraints,
            include_accuracy=True,
            workload_type="detection",
            quantization_dtype="int8" if power_w <= 5.0 else "fp16",
            total_estimated_gflops=operator_graph.total_estimated_gflops,
            platform=platform,
        )

        environment = ""
        desc_lower = description.lower()
        if "indoor" in desc_lower:
            environment = "indoor"
        elif "outdoor" in desc_lower:
            environment = "outdoor"
        if "gps-denied" in desc_lower or "gps denied" in desc_lower:
            environment = f"gps-denied {environment}".strip()

        return MissionPlan(
            mission=description,
            platform=platform,
            mission_type=mission_type,
            environment=environment,
            sub_capabilities=sub_caps,
            operator_graph=operator_graph,
            design_space=design_space,
            rationale=(
                f"Heuristic decomposition for {platform} {mission_type} mission. "
                f"Latency budget: {e2e_latency_ms}ms, power budget: {power_w}W."
            ),
        )

    def _build_standard_capabilities(
        self, mission_type: str, frame_rate: float, e2e_latency_ms: float
    ) -> list[SubCapability]:
        """Build standard sub-capabilities for common mission types."""
        caps = [
            SubCapability(
                name="object_detection",
                description="Detect objects in camera frames",
                required=True,
                latency_budget_ms=e2e_latency_ms * 0.5,
                accuracy_requirement=">35% mAP@50",
                rate_hz=frame_rate,
            ),
            SubCapability(
                name="object_tracking",
                description="Track detected objects across frames",
                required=True,
                latency_budget_ms=e2e_latency_ms * 0.15,
                rate_hz=frame_rate,
                depends_on=["object_detection"],
            ),
        ]

        if mission_type in ("navigation", "perception"):
            caps.append(
                SubCapability(
                    name="scene_understanding",
                    description="Build semantic scene graph for reasoning",
                    required=True,
                    latency_budget_ms=e2e_latency_ms * 0.1,
                    rate_hz=frame_rate,
                    depends_on=["object_tracking"],
                )
            )
            caps.append(
                SubCapability(
                    name="state_estimation",
                    description="Estimate platform state (pose, velocity)",
                    required=True,
                    latency_budget_ms=1.0,
                    rate_hz=100.0,
                )
            )

        return caps

    def _build_standard_operator_graph(
        self, mission_type: str, frame_rate: float, e2e_latency_ms: float
    ) -> OperatorGraph:
        """Build a standard operator graph for common mission types."""
        operators = [
            OperatorSpec(
                name="preprocess",
                operator_type=OperatorType.PREPROCESS,
                estimated_gflops=0.1,
                latency_budget_ms=1.0,
                rate_hz=frame_rate,
                input_resolution="640x640",
                capability="object_detection",
            ),
            OperatorSpec(
                name="detector",
                operator_type=OperatorType.DETECTION,
                model_family="yolov8",
                model_variant="s",
                estimated_gflops=28.6,
                latency_budget_ms=e2e_latency_ms * 0.5,
                rate_hz=frame_rate,
                input_resolution="640x640",
                quantization="int8",
                capability="object_detection",
                notes="YOLOv8s balances accuracy (44.9 mAP@50) and compute",
            ),
            OperatorSpec(
                name="tracker",
                operator_type=OperatorType.TRACKING,
                model_family="bytetrack",
                estimated_gflops=0.01,
                latency_budget_ms=e2e_latency_ms * 0.1,
                rate_hz=frame_rate,
                capability="object_tracking",
            ),
        ]

        edges = [
            OperatorEdge(source="preprocess", target="detector", data_type="tensor"),
            OperatorEdge(source="detector", target="tracker", data_type="detections"),
        ]

        if mission_type in ("navigation", "perception"):
            operators.append(
                OperatorSpec(
                    name="scene_graph",
                    operator_type=OperatorType.SCENE_GRAPH,
                    estimated_gflops=0.001,
                    latency_budget_ms=e2e_latency_ms * 0.1,
                    rate_hz=frame_rate,
                    capability="scene_understanding",
                )
            )
            operators.append(
                OperatorSpec(
                    name="state_estimator",
                    operator_type=OperatorType.STATE_ESTIMATION,
                    model_family="ekf",
                    estimated_gflops=0.001,
                    latency_budget_ms=0.5,
                    rate_hz=100.0,
                    capability="state_estimation",
                    notes="EKF with 12-state (pos, vel, orientation, bias)",
                )
            )
            edges.append(OperatorEdge(source="tracker", target="scene_graph", data_type="tracks"))

        total_gflops = sum(op.estimated_gflops or 0 for op in operators)

        return OperatorGraph(
            operators=operators,
            edges=edges,
            total_estimated_gflops=round(total_gflops, 2),
            end_to_end_latency_budget_ms=e2e_latency_ms,
        )
