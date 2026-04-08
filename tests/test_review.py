"""Tests for human-in-the-loop plan review."""

import pytest

from embodied_ai_architect.graphs.review import (
    PlanReviewInput,
    PlanReviewSnapshot,
    ReviewDecision,
    apply_review_edits,
    build_review_snapshot,
    compute_parallel_groups,
    render_plan_review_rich,
    render_task_graph_ascii,
    render_task_table,
    summarize_constraints,
    validate_modified_graph,
)
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
)
from embodied_ai_architect.graphs.task_graph import TaskGraph, TaskNode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

AVAILABLE_AGENTS = [
    "workload_analyzer",
    "hw_explorer",
    "architecture_composer",
    "ppa_assessor",
    "critic",
    "report_generator",
]

SAMPLE_PLAN = [
    {"id": "t1", "name": "Analyze workload", "agent": "workload_analyzer", "dependencies": []},
    {"id": "t2", "name": "Explore hardware", "agent": "hw_explorer", "dependencies": ["t1"]},
    {
        "id": "t3",
        "name": "Compose architecture",
        "agent": "architecture_composer",
        "dependencies": ["t2"],
    },
    {"id": "t4", "name": "Assess PPA", "agent": "ppa_assessor", "dependencies": ["t3"]},
    {
        "id": "t5",
        "name": "Generate report",
        "agent": "report_generator",
        "dependencies": ["t4"],
    },
]


def _make_graph(plan=None) -> TaskGraph:
    """Build a TaskGraph from a plan list."""
    plan = plan or SAMPLE_PLAN
    graph = TaskGraph()
    nodes = [
        TaskNode(
            id=t["id"],
            name=t["name"],
            agent=t["agent"],
            dependencies=t.get("dependencies", []),
        )
        for t in plan
    ]
    graph.add_tasks(nodes)
    return graph


def _make_state(plan=None, constraints=None):
    """Build a state dict with a task graph."""
    state = create_initial_soc_state(
        goal="Design a drone perception SoC",
        constraints=constraints or DesignConstraints(max_power_watts=5.0, max_latency_ms=33.3),
        use_case="delivery_drone",
        platform="drone",
    )
    graph = _make_graph(plan)
    state["task_graph"] = graph.to_dict()
    return state


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------


class TestRenderTaskGraphAscii:
    def test_empty_graph(self):
        graph = TaskGraph()
        result = render_task_graph_ascii(graph)
        assert "empty" in result.lower()

    def test_single_task(self):
        graph = TaskGraph()
        graph.add_task(TaskNode(id="t1", name="Analyze", agent="workload_analyzer"))
        result = render_task_graph_ascii(graph)
        assert "[t1]" in result
        assert "Analyze" in result
        assert "workload_analyzer" in result

    def test_linear_chain(self):
        graph = _make_graph()
        result = render_task_graph_ascii(graph)
        assert "[t1]" in result
        assert "[t5]" in result
        # All tasks should appear
        for tid in ["t1", "t2", "t3", "t4", "t5"]:
            assert f"[{tid}]" in result

    def test_parallel_tasks(self):
        plan = [
            {"id": "t1", "name": "Analyze", "agent": "workload_analyzer", "dependencies": []},
            {"id": "t2", "name": "Explore A", "agent": "hw_explorer", "dependencies": ["t1"]},
            {"id": "t3", "name": "Explore B", "agent": "critic", "dependencies": ["t1"]},
        ]
        graph = _make_graph(plan)
        result = render_task_graph_ascii(graph)
        assert "[t1]" in result
        assert "[t2]" in result
        assert "[t3]" in result


class TestRenderTaskTable:
    def test_produces_rows(self):
        graph = _make_graph()
        table = render_task_table(graph)
        assert len(table) == 5
        assert table[0]["id"] == "t1"
        assert table[0]["agent"] == "workload_analyzer"

    def test_dependencies_formatted(self):
        graph = _make_graph()
        table = render_task_table(graph)
        # t1 has no deps
        assert table[0]["dependencies"] == "(none)"
        # t2 depends on t1
        t2_row = next(r for r in table if r["id"] == "t2")
        assert "t1" in t2_row["dependencies"]


# ---------------------------------------------------------------------------
# Parallel groups
# ---------------------------------------------------------------------------


class TestComputeParallelGroups:
    def test_empty_graph(self):
        graph = TaskGraph()
        assert compute_parallel_groups(graph) == []

    def test_linear_chain(self):
        graph = _make_graph()
        groups = compute_parallel_groups(graph)
        # Linear chain = 5 sequential groups
        assert len(groups) == 5
        assert groups[0] == ["t1"]
        assert groups[4] == ["t5"]

    def test_parallel_tasks(self):
        plan = [
            {"id": "t1", "name": "Analyze", "agent": "workload_analyzer", "dependencies": []},
            {"id": "t2", "name": "Explore A", "agent": "hw_explorer", "dependencies": ["t1"]},
            {"id": "t3", "name": "Explore B", "agent": "critic", "dependencies": ["t1"]},
            {
                "id": "t4",
                "name": "Merge",
                "agent": "architecture_composer",
                "dependencies": ["t2", "t3"],
            },
        ]
        graph = _make_graph(plan)
        groups = compute_parallel_groups(graph)
        assert len(groups) == 3  # t1, [t2, t3], t4
        assert sorted(groups[1]) == ["t2", "t3"]


# ---------------------------------------------------------------------------
# Constraints summary
# ---------------------------------------------------------------------------


class TestSummarizeConstraints:
    def test_basic(self):
        c = DesignConstraints(max_power_watts=5.0, max_latency_ms=33.3)
        result = summarize_constraints(c)
        assert "5.0" in result
        assert "33.3" in result
        assert "Power" in result
        assert "Latency" in result

    def test_empty(self):
        c = DesignConstraints()
        result = summarize_constraints(c)
        assert "no constraints" in result.lower()


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------


class TestBuildReviewSnapshot:
    def test_builds_snapshot(self):
        state = _make_state()
        snap = build_review_snapshot(state, AVAILABLE_AGENTS)
        assert snap.goal == "Design a drone perception SoC"
        assert snap.use_case == "delivery_drone"
        assert snap.platform == "drone"
        assert len(snap.task_table) == 5
        assert snap.available_agents == sorted(AVAILABLE_AGENTS)
        assert snap.inferred_context["num_tasks"] == 5

    def test_snapshot_serializable(self):
        state = _make_state()
        snap = build_review_snapshot(state, AVAILABLE_AGENTS)
        data = snap.model_dump()
        # Should be JSON-serializable
        import json

        json.dumps(data)
        # Should round-trip
        snap2 = PlanReviewSnapshot(**data)
        assert snap2.goal == snap.goal


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateModifiedGraph:
    def test_valid_graph(self):
        graph = _make_graph()
        errors = validate_modified_graph(graph, AVAILABLE_AGENTS)
        assert errors == []

    def test_unknown_agent(self):
        graph = _make_graph()
        graph.nodes["t1"].agent = "nonexistent_agent"
        errors = validate_modified_graph(graph, AVAILABLE_AGENTS)
        assert len(errors) == 1
        assert "nonexistent_agent" in errors[0]

    def test_empty_graph(self):
        graph = TaskGraph()
        errors = validate_modified_graph(graph, AVAILABLE_AGENTS)
        assert len(errors) == 1
        assert "empty" in errors[0].lower()


# ---------------------------------------------------------------------------
# Edit application
# ---------------------------------------------------------------------------


class TestApplyReviewEdits:
    def test_approve_no_changes(self):
        state = _make_state()
        review = PlanReviewInput(decision=ReviewDecision.APPROVE)
        updates = apply_review_edits(state, review, AVAILABLE_AGENTS)
        assert "task_graph" in updates

    def test_remove_task(self):
        state = _make_state()
        review = PlanReviewInput(
            decision=ReviewDecision.MODIFY,
            tasks_to_remove=["t5"],
        )
        updates = apply_review_edits(state, review, AVAILABLE_AGENTS)
        graph = TaskGraph.from_dict(updates["task_graph"])
        assert "t5" not in graph.nodes
        assert len(graph.nodes) == 4

    def test_add_task(self):
        state = _make_state()
        review = PlanReviewInput(
            decision=ReviewDecision.MODIFY,
            tasks_to_add=[
                {
                    "id": "t6",
                    "name": "Review safety",
                    "agent": "critic",
                    "dependencies": ["t4"],
                }
            ],
        )
        updates = apply_review_edits(state, review, AVAILABLE_AGENTS)
        graph = TaskGraph.from_dict(updates["task_graph"])
        assert "t6" in graph.nodes
        assert graph.nodes["t6"].agent == "critic"

    def test_reassign_agent(self):
        state = _make_state()
        review = PlanReviewInput(
            decision=ReviewDecision.MODIFY,
            agent_reassignments={"t2": "critic"},
        )
        updates = apply_review_edits(state, review, AVAILABLE_AGENTS)
        graph = TaskGraph.from_dict(updates["task_graph"])
        assert graph.nodes["t2"].agent == "critic"

    def test_override_constraints(self):
        state = _make_state()
        review = PlanReviewInput(
            decision=ReviewDecision.MODIFY,
            constraint_overrides={"max_power_watts": 10.0},
        )
        updates = apply_review_edits(state, review, AVAILABLE_AGENTS)
        assert updates["constraints"]["max_power_watts"] == 10.0

    def test_invalid_agent_raises(self):
        state = _make_state()
        review = PlanReviewInput(
            decision=ReviewDecision.MODIFY,
            agent_reassignments={"t1": "fake_agent"},
        )
        with pytest.raises(ValueError, match="fake_agent"):
            apply_review_edits(state, review, AVAILABLE_AGENTS)

    def test_notes_recorded(self):
        state = _make_state()
        review = PlanReviewInput(
            decision=ReviewDecision.APPROVE,
            notes="Adding safety review because this is a flight-critical system",
        )
        updates = apply_review_edits(state, review, AVAILABLE_AGENTS)
        rationale = updates.get("design_rationale", [])
        assert any("safety review" in r for r in rationale)


# ---------------------------------------------------------------------------
# Rich rendering
# ---------------------------------------------------------------------------


class TestRenderPlanReviewRich:
    def test_renders_string(self):
        state = _make_state()
        snap = build_review_snapshot(state, AVAILABLE_AGENTS)
        result = render_plan_review_rich(snap)
        assert "PLAN REVIEW" in result
        assert "drone perception" in result.lower()
        assert "TASK GRAPH" in result
        assert "EXECUTION SCHEDULE" in result
        assert "AVAILABLE AGENTS" in result


# ---------------------------------------------------------------------------
# Issue #29: KPU plan review (preview + overrides)
# ---------------------------------------------------------------------------


def _make_rtl_state():
    """State with rtl_enabled=True so KPU preview activates."""
    state = create_initial_soc_state(
        goal="Design a drone perception SoC",
        constraints=DesignConstraints(max_power_watts=5.0, max_latency_ms=33.3),
        use_case="delivery_drone",
        platform="drone",
        rtl_enabled=True,
    )
    state["task_graph"] = _make_graph().to_dict()
    return state


class TestKPUPreview:
    """Snapshot must include a KPU preview when rtl_enabled, and not otherwise."""

    def test_no_preview_when_rtl_disabled(self):
        state = _make_state()  # rtl_enabled defaults to False
        snap = build_review_snapshot(state, AVAILABLE_AGENTS)
        assert snap.kpu_preview is None
        assert snap.kpu_preview_summary == ""

    def test_preview_present_when_rtl_enabled(self):
        state = _make_rtl_state()
        snap = build_review_snapshot(state, AVAILABLE_AGENTS)
        assert snap.kpu_preview is not None
        assert "Compute tiles" in snap.kpu_preview_summary
        assert "Systolic array" in snap.kpu_preview_summary
        assert "L2 SRAM" in snap.kpu_preview_summary
        assert "NoC" in snap.kpu_preview_summary
        assert "DRAM" in snap.kpu_preview_summary

    def test_preview_carries_preset_fields(self):
        state = _make_rtl_state()
        snap = build_review_snapshot(state, AVAILABLE_AGENTS)
        kpu = snap.kpu_preview
        # Standard KPU config keys produced by create_kpu_config()
        assert "compute_tile" in kpu
        assert "memory_tile" in kpu
        assert "noc" in kpu
        assert "dram" in kpu
        assert "array_rows" in kpu
        assert "array_cols" in kpu

    def test_preview_reflects_existing_overrides(self):
        """If kpu_config_overrides is already on state, preview reflects them."""
        state = _make_rtl_state()
        state["kpu_config_overrides"] = {"compute_tile.array_rows": 32}
        snap = build_review_snapshot(state, AVAILABLE_AGENTS)
        assert snap.kpu_preview["compute_tile"]["array_rows"] == 32
        assert "override" in snap.kpu_preview_summary.lower()

    def test_rich_render_includes_kpu_section(self):
        state = _make_rtl_state()
        snap = build_review_snapshot(state, AVAILABLE_AGENTS)
        rendered = render_plan_review_rich(snap)
        assert "KPU MICRO-ARCHITECTURE" in rendered
        assert "delivery_drone" in rendered
        assert "kpu_overrides" in rendered  # the help line


class TestKPUOverridesInReview:
    """PlanReviewInput.kpu_overrides must flow into state via apply_review_edits."""

    def test_kpu_overrides_recorded_in_updates(self):
        state = _make_rtl_state()
        review = PlanReviewInput(
            decision=ReviewDecision.MODIFY,
            kpu_overrides={
                "compute_tile.array_rows": 8,
                "compute_tile.array_cols": 8,
                "noc.link_width_bits": 512,
            },
        )
        updates = apply_review_edits(state, review, AVAILABLE_AGENTS)
        assert "kpu_config_overrides" in updates
        assert updates["kpu_config_overrides"]["compute_tile.array_rows"] == 8
        assert updates["kpu_config_overrides"]["noc.link_width_bits"] == 512

    def test_kpu_overrides_merge_with_existing(self):
        """A second review pass should add to (not replace) prior overrides."""
        state = _make_rtl_state()
        state["kpu_config_overrides"] = {"dram.num_controllers": 4}
        review = PlanReviewInput(
            decision=ReviewDecision.MODIFY,
            kpu_overrides={"compute_tile.array_rows": 8},
        )
        updates = apply_review_edits(state, review, AVAILABLE_AGENTS)
        merged = updates["kpu_config_overrides"]
        assert merged["dram.num_controllers"] == 4  # preserved
        assert merged["compute_tile.array_rows"] == 8  # added

    def test_kpu_override_summary_in_history(self):
        """The design rationale must mention KPU overrides for traceability."""
        state = _make_rtl_state()
        review = PlanReviewInput(
            decision=ReviewDecision.MODIFY,
            kpu_overrides={"compute_tile.array_rows": 8},
        )
        updates = apply_review_edits(state, review, AVAILABLE_AGENTS)
        rationale = updates.get("design_rationale", [])
        assert any("KPU parameter" in r for r in rationale)

    def test_no_kpu_section_when_overrides_empty(self):
        """Approving without kpu_overrides leaves kpu_config_overrides untouched."""
        state = _make_rtl_state()
        review = PlanReviewInput(decision=ReviewDecision.APPROVE)
        updates = apply_review_edits(state, review, AVAILABLE_AGENTS)
        assert "kpu_config_overrides" not in updates
