"""Tests for the research library and mission decomposition (Phase 2).

Tests cover:
- ResearchLibrary: document loading, tag retrieval, context block building
- MissionPlan models: serialization, summary, constraints conversion
- MissionDecomposer: heuristic decomposition, hint extraction
- Decomposition tools: tool definitions, executor wiring
"""

from __future__ import annotations

import json

from embodied_ai_architect.research.library import ResearchLibrary, Document
from embodied_ai_architect.research.models import (
    DesignConstraint,
    DesignSpaceDefinition,
    MissionPlan,
    OperatorEdge,
    OperatorGraph,
    OperatorSpec,
    OperatorType,
    SubCapability,
)
from embodied_ai_architect.research.decomposer import MissionDecomposer

# ---------------------------------------------------------------------------
# ResearchLibrary tests
# ---------------------------------------------------------------------------


class TestResearchLibrary:
    """Tests for ResearchLibrary loading and retrieval."""

    def test_library_loads_from_default_path(self):
        lib = ResearchLibrary()
        assert lib.count > 0, "Library should load at least one document"

    def test_library_lists_tags(self):
        lib = ResearchLibrary()
        tags = lib.list_tags()
        assert len(tags) > 0
        assert isinstance(tags[0], str)

    def test_library_lists_domains(self):
        lib = ResearchLibrary()
        domains = lib.list_domains()
        assert len(domains) > 0

    def test_retrieve_by_tags(self):
        lib = ResearchLibrary()
        docs = lib.retrieve(tags=["drone"])
        assert len(docs) > 0
        # At least one returned doc should have 'drone' in its tags
        found = any("drone" in (t.lower() for t in d.tags) for d in docs)
        assert found, "Should find at least one doc tagged 'drone'"

    def test_retrieve_by_relevance(self):
        lib = ResearchLibrary()
        docs = lib.retrieve(relevance="mission_decomposition")
        assert len(docs) > 0

    def test_retrieve_by_domain(self):
        lib = ResearchLibrary()
        docs = lib.retrieve(domain="accelerator", tags=["edge"])
        assert len(docs) > 0

    def test_retrieve_max_results(self):
        lib = ResearchLibrary()
        docs = lib.retrieve(tags=["edge"], max_results=2)
        assert len(docs) <= 2

    def test_retrieve_no_match_returns_empty(self):
        lib = ResearchLibrary()
        docs = lib.retrieve(tags=["nonexistent_tag_xyz"])
        assert docs == []

    def test_build_context_block(self):
        lib = ResearchLibrary()
        docs = lib.retrieve(tags=["drone"], max_results=2)
        block = lib.build_context_block(docs, max_tokens=4000)
        assert "<research_context>" in block
        assert "</research_context>" in block

    def test_build_context_block_empty(self):
        lib = ResearchLibrary()
        block = lib.build_context_block([], max_tokens=4000)
        assert block == ""

    def test_build_context_block_respects_token_budget(self):
        lib = ResearchLibrary()
        docs = lib.retrieve(tags=["edge"], max_results=10)
        block = lib.build_context_block(docs, max_tokens=500)
        # Rough token estimate: 4 chars per token
        assert len(block) // 4 < 600, "Context block should stay near token budget"

    def test_get_by_path(self):
        lib = ResearchLibrary()
        # Get a known document
        docs = lib.documents
        if docs:
            first = docs[0]
            found = lib.get_by_path(first.path)
            assert found is not None
            assert found.path == first.path

    def test_document_estimated_tokens(self):
        doc = Document(
            path="test.md",
            title="Test",
            content="a" * 400,
        )
        assert doc.estimated_tokens == 100  # 400 chars / 4


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestModels:
    """Tests for MissionPlan and related Pydantic models."""

    def test_sub_capability_defaults(self):
        cap = SubCapability(name="detection")
        assert cap.required is True
        assert cap.latency_budget_ms is None
        assert cap.depends_on == []

    def test_operator_spec_serialization(self):
        op = OperatorSpec(
            name="detector",
            operator_type=OperatorType.DETECTION,
            model_family="yolov8",
            model_variant="s",
            estimated_gflops=28.6,
        )
        data = op.model_dump()
        assert data["name"] == "detector"
        assert data["operator_type"] == "detection"
        assert data["estimated_gflops"] == 28.6

    def test_operator_graph_topological_order(self):
        graph = OperatorGraph(
            operators=[
                OperatorSpec(name="a", operator_type=OperatorType.PREPROCESS),
                OperatorSpec(name="b", operator_type=OperatorType.DETECTION),
                OperatorSpec(name="c", operator_type=OperatorType.TRACKING),
            ],
            edges=[
                OperatorEdge(source="a", target="b"),
                OperatorEdge(source="b", target="c"),
            ],
        )
        order = graph.topological_order()
        assert order == ["a", "b", "c"]

    def test_operator_graph_get_sources_sinks(self):
        graph = OperatorGraph(
            operators=[
                OperatorSpec(name="a", operator_type=OperatorType.PREPROCESS),
                OperatorSpec(name="b", operator_type=OperatorType.DETECTION),
            ],
            edges=[OperatorEdge(source="a", target="b")],
        )
        assert graph.get_sources("b") == ["a"]
        assert graph.get_sinks("a") == ["b"]
        assert graph.get_sources("a") == []

    def test_design_space_to_constraints_dict(self):
        ds = DesignSpaceDefinition(
            constraints=[
                DesignConstraint(name="power_watts", value=10.0, constraint_type="upper"),
                DesignConstraint(name="accuracy_percent", value=30.0, constraint_type="lower"),
            ],
            include_accuracy=True,
            workload_type="detection",
            quantization_dtype="int8",
        )
        d = ds.to_constraints_dict()
        assert d["max_power_watts"] == 10.0
        assert d["min_accuracy_percent"] == 30.0
        assert d["include_accuracy"] is True
        assert d["workload_type"] == "detection"
        assert d["quantization_dtype"] == "int8"

    def test_mission_plan_summary(self):
        plan = MissionPlan(
            mission="Test mission",
            platform="drone",
            mission_type="perception",
            sub_capabilities=[
                SubCapability(name="detection", latency_budget_ms=20.0, rate_hz=30.0),
            ],
            operator_graph=OperatorGraph(
                operators=[
                    OperatorSpec(
                        name="det",
                        operator_type=OperatorType.DETECTION,
                        model_family="yolov8",
                    ),
                ],
            ),
            design_space=DesignSpaceDefinition(
                constraints=[
                    DesignConstraint(name="power", value=10.0, constraint_type="upper"),
                ],
            ),
        )
        summary = plan.summary()
        assert "Test mission" in summary
        assert "drone" in summary
        assert "detection" in summary

    def test_operator_type_values(self):
        assert OperatorType.DETECTION.value == "detection"
        assert OperatorType.TRACKING.value == "tracking"
        assert OperatorType.SENSOR_FUSION.value == "sensor_fusion"


# ---------------------------------------------------------------------------
# MissionDecomposer tests
# ---------------------------------------------------------------------------


class TestMissionDecomposer:
    """Tests for heuristic mission decomposition."""

    def test_drone_perception_decomposition(self):
        decomposer = MissionDecomposer()
        plan = decomposer.decompose("Drone perception for GPS-denied navigation at 5 m/s")
        assert plan.platform == "drone"
        assert plan.mission_type in ("perception", "navigation")
        assert len(plan.sub_capabilities) > 0
        assert len(plan.operator_graph.operators) > 0
        assert plan.design_space.include_accuracy is True

    def test_automotive_decomposition(self):
        decomposer = MissionDecomposer()
        plan = decomposer.decompose("Automotive ADAS perception for highway driving")
        assert plan.platform == "vehicle"
        assert len(plan.sub_capabilities) > 0

    def test_edge_detection_decomposition(self):
        decomposer = MissionDecomposer()
        plan = decomposer.decompose("Edge camera object detection for security monitoring")
        assert plan.platform == "edge"

    def test_extract_speed(self):
        decomposer = MissionDecomposer()
        hints = decomposer._extract_mission_hints("Drone navigation at 5 m/s")
        assert hints["speed_ms"] == 5.0

    def test_extract_power(self):
        decomposer = MissionDecomposer()
        hints = decomposer._extract_mission_hints("Low-power 3W wearable vision")
        assert hints["power_w"] == 3.0

    def test_high_speed_latency_budget(self):
        decomposer = MissionDecomposer()
        plan = decomposer.decompose("Drone perception at 10 m/s")
        # High speed should give 30 FPS → ~33ms budget
        assert plan.operator_graph.end_to_end_latency_budget_ms < 40.0

    def test_research_context_populated(self):
        decomposer = MissionDecomposer()
        plan = decomposer.decompose("Drone perception for obstacle avoidance")
        assert len(plan.research_context_used) > 0

    def test_operator_graph_has_edges(self):
        decomposer = MissionDecomposer()
        plan = decomposer.decompose("Drone perception at 5 m/s")
        assert len(plan.operator_graph.edges) > 0

    def test_design_space_has_constraints(self):
        decomposer = MissionDecomposer()
        plan = decomposer.decompose("Drone perception at 5 m/s within 10W")
        constraints = plan.design_space.constraints
        assert len(constraints) > 0
        # Should have a power constraint
        names = [c.name for c in constraints]
        assert "power_watts" in names

    def test_design_space_to_soc_constraints(self):
        """Verify the design space bridges to create_soc_design_space()."""
        decomposer = MissionDecomposer()
        plan = decomposer.decompose("Drone perception at 5 m/s")
        d = plan.design_space.to_constraints_dict()
        assert "include_accuracy" in d
        assert "workload_type" in d
        assert d["include_accuracy"] is True


# ---------------------------------------------------------------------------
# Decomposition tools tests
# ---------------------------------------------------------------------------


class TestDecompositionTools:
    """Tests for LLM tool definitions and executors."""

    def test_tool_definitions_structure(self):
        from embodied_ai_architect.llm.decomposition_tools import (
            get_decomposition_tool_definitions,
        )

        defs = get_decomposition_tool_definitions()
        assert len(defs) == 3
        names = {d["name"] for d in defs}
        assert names == {"decompose_mission", "retrieve_research", "formulate_design_space"}
        for d in defs:
            assert "description" in d
            assert "input_schema" in d
            assert d["input_schema"]["type"] == "object"

    def test_decompose_mission_executor(self):
        from embodied_ai_architect.llm.decomposition_tools import (
            create_decomposition_tool_executors,
        )

        executors = create_decomposition_tool_executors()
        result = executors["decompose_mission"](mission_description="Drone perception at 5 m/s")
        data = json.loads(result)
        assert data["platform"] == "drone"
        assert "operators" in data
        assert len(data["operators"]) > 0

    def test_retrieve_research_executor(self):
        from embodied_ai_architect.llm.decomposition_tools import (
            create_decomposition_tool_executors,
        )

        executors = create_decomposition_tool_executors()
        result = executors["retrieve_research"](tags=["drone", "edge"])
        data = json.loads(result)
        assert data["documents_found"] > 0
        assert "context_block" in data

    def test_retrieve_research_no_match(self):
        from embodied_ai_architect.llm.decomposition_tools import (
            create_decomposition_tool_executors,
        )

        executors = create_decomposition_tool_executors()
        result = executors["retrieve_research"](tags=["nonexistent_xyz"])
        data = json.loads(result)
        assert data["documents"] == []
        assert "available_tags" in data

    def test_formulate_design_space_executor(self):
        from embodied_ai_architect.llm.decomposition_tools import (
            create_decomposition_tool_executors,
        )

        executors = create_decomposition_tool_executors()
        result = executors["formulate_design_space"](
            mission_description="Drone perception at 5 m/s",
            power_budget_watts=8.0,
        )
        data = json.loads(result)
        assert "constraints_for_create_soc_design_space" in data
        constraints = data["constraints_for_create_soc_design_space"]
        assert constraints["max_power_watts"] == 8.0
        assert constraints["include_accuracy"] is True

    def test_tools_registered_in_main_tools(self):
        from embodied_ai_architect.llm.tools import get_tool_definitions, create_tool_executors

        defs = get_tool_definitions()
        names = {d["name"] for d in defs}
        assert "decompose_mission" in names
        assert "retrieve_research" in names
        assert "formulate_design_space" in names

        executors = create_tool_executors()
        assert "decompose_mission" in executors
        assert "retrieve_research" in executors
        assert "formulate_design_space" in executors
