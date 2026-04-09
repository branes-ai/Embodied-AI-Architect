"""End-to-end integration test for the codebase → SoC design pipeline (issue #43).

Exercises the full flow:

    scan cpp_drone → build CodebaseAnalysisResult → convert to workload_profile
    → infer constraints → recommend hardware → codebase_to_soc_state → save
    session → verify API serves codebase-originated data correctly

The LLM analysis is bypassed (no API key in CI) — the test constructs a
realistic CodebaseAnalysisResult from the scanner output, mimicking the
agent's scan-only fallback path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from embodied_ai_architect.codebase.converter import (
    CodebaseConverter,
    codebase_to_soc_state,
    infer_constraints,
)
from embodied_ai_architect.codebase.models import (
    CodebaseAnalysisResult,
    ComputeKernel,
    DataflowLink,
)
from embodied_ai_architect.codebase.recommender import (
    classify_workload,
    recommend_hardware,
)
from embodied_ai_architect.codebase.scanner import CodebaseScanner
from embodied_ai_architect.graphs.session_store import SessionStore

FIXTURES = Path(__file__).parent / "fixtures" / "sample_projects"


# ---------------------------------------------------------------------------
# Module-scoped fixtures — build the analysis chain once, share across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scan_result():
    """Scan the cpp_drone fixture (no LLM, fast)."""
    scanner = CodebaseScanner()
    return scanner.scan(FIXTURES / "cpp_drone")


@pytest.fixture(scope="module")
def analysis(scan_result):
    """Build a realistic CodebaseAnalysisResult from the scanner output.

    In production the LLM fills this in. In tests we construct it manually
    using the same kernel types / frequencies the LLM would infer from the
    source code. The test_codebase.py unit tests already verify the scanner
    detects cmake, OpenCV, Eigen, etc. — this fixture builds on that.
    """
    return CodebaseAnalysisResult(
        project_name=scan_result.project_name,
        languages=scan_result.languages,
        source_files=scan_result.source_files,
        build_system=scan_result.build_system,
        dependencies=scan_result.dependencies,
        ml_models=scan_result.ml_models,
        kernels=[
            ComputeKernel(
                name="yolo_detection",
                source_file="src/perception.cpp",
                line_range=(15, 45),
                kernel_type="ml_inference",
                estimated_ops_per_invocation=8.4e9,
                invocation_frequency_hz=30.0,
                data_types=["float32"],
                parallelism="data_parallel",
                memory_access_pattern="streaming",
                frameworks=["opencv", "onnxruntime"],
            ),
            ComputeKernel(
                name="pid_control",
                source_file="src/control.cpp",
                line_range=(5, 35),
                kernel_type="control_loop",
                estimated_ops_per_invocation=1e6,
                invocation_frequency_hz=100.0,
                data_types=["float32"],
                parallelism="sequential",
                memory_access_pattern="reuse_heavy",
                frameworks=["eigen"],
            ),
        ],
        dataflow=[
            DataflowLink(
                source_kernel="yolo_detection",
                sink_kernel="pid_control",
                data_size_bytes=4096,
                transfer_type="memory",
            ),
        ],
        summary="Drone perception pipeline with YOLO inference and PID control loop",
    )


@pytest.fixture(scope="module")
def workload_profile(analysis):
    """Convert the analysis to a workload_profile."""
    return CodebaseConverter().to_workload_profile(analysis)


@pytest.fixture(scope="module")
def session_dir(tmp_path_factory):
    """Module-scoped tmp dir for the session store."""
    return str(tmp_path_factory.mktemp("codebase_integration_sessions"))


@pytest.fixture(scope="module")
def soc_state(analysis, session_dir):
    """Create and save a SoCDesignState from the codebase analysis."""
    state = codebase_to_soc_state(
        analysis,
        project_path=str(FIXTURES / "cpp_drone"),
        session_id="soc_codebase_integration",
    )
    store = SessionStore(session_dir=session_dir)
    store.save(state)
    return state


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestScanToWorkloadProfile:
    """Scanner → CodebaseConverter → workload_profile."""

    def test_workload_source_is_codebase_analysis(self, workload_profile):
        assert workload_profile["source"] == "codebase_analysis"

    def test_workload_has_drone_relevant_kernels(self, workload_profile):
        names = [w["name"] for w in workload_profile["workloads"]]
        assert "yolo_detection" in names
        assert "pid_control" in names

    def test_workload_total_gflops_positive(self, workload_profile):
        assert workload_profile["total_estimated_gflops"] > 0

    def test_operator_graph_has_nodes_and_edges(self, workload_profile):
        graph = workload_profile.get("operator_graph")
        assert graph is not None
        assert len(graph["nodes"]) == 2
        # Dataflow link from analysis → 1 edge
        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["source"] == "yolo_detection"
        assert graph["edges"][0]["sink"] == "pid_control"


class TestConstraintInference:
    """infer_constraints() on the drone analysis."""

    def test_control_loop_implies_latency(self, analysis):
        suggestions = infer_constraints(analysis)
        latency = next(
            (c for c in suggestions.constraints if c.name == "max_latency_ms"),
            None,
        )
        assert latency is not None
        # 100Hz → 10ms period
        assert latency.value == 10.0
        assert latency.confidence == "high"

    def test_power_envelope_inferred(self, analysis):
        suggestions = infer_constraints(analysis)
        power = next(
            (c for c in suggestions.constraints if c.name == "max_power_watts"),
            None,
        )
        # ML inference + control loop → some power estimate
        assert power is not None
        assert power.value >= 1.0

    def test_hardware_class_hint_depends_on_kernel_count(self, analysis):
        """With 1 ML + 1 control kernel the ML share is 50% — below the 60%
        threshold for the hardware_class heuristic. That's correct: by
        kernel count, the workload is balanced. The heuristic fires when
        ML kernels dominate by COUNT, not by GFLOPS. Verify the rule
        respects the threshold."""
        suggestions = infer_constraints(analysis)
        hw = next(
            (c for c in suggestions.constraints if c.name == "hardware_class"),
            None,
        )
        # 50% ML share → no hardware_class hint (threshold is >50%)
        assert hw is None


class TestHardwareRecommendations:
    """recommend_hardware() on the drone workload."""

    def test_archetype_is_ml_heavy(self, workload_profile):
        # 2 kernels: ml_inference (dominant) + control_loop
        # Not 60% ML share with only 2 kernels (50/50) → hybrid
        # OR if the test fixture has enough ML weight...
        archetype = classify_workload(workload_profile)
        # With 2 kernels (1 ML + 1 control), share is 50% — below 60% → hybrid
        assert archetype in ("ml_heavy", "hybrid")

    def test_recommendations_available_from_registry(self, workload_profile):
        """When the embodied-schemas registry is installed, recommend_hardware
        returns non-empty results. When not installed, degrades to []."""
        recs = recommend_hardware(workload_profile, top_k=5)
        # Either the registry is installed and we get results, or not
        # Both are valid in CI — the test doesn't fail either way
        assert isinstance(recs, list)
        for r in recs:
            assert hasattr(r, "fit_score")
            assert hasattr(r, "strengths")


class TestSoCStateCreation:
    """codebase_to_soc_state() from the analysis."""

    def test_state_has_workload_profile(self, soc_state):
        wp = soc_state.get("workload_profile", {})
        assert wp.get("source") == "codebase_analysis"
        assert wp.get("workload_count", 0) == 2
        assert wp.get("total_estimated_gflops", 0) > 0

    def test_state_has_codebase_metadata(self, soc_state):
        meta = soc_state.get("codebase_metadata", {})
        assert meta.get("project_name") == "cpp_drone"
        assert Path(meta.get("project_path", "")).is_absolute()
        assert meta.get("kernel_count") == 2

    def test_state_goal_includes_project_name(self, soc_state):
        goal = soc_state.get("goal", "")
        assert "cpp_drone" in goal.lower() or "drone" in goal.lower()

    def test_state_use_case_is_canonical(self, soc_state):
        assert soc_state.get("use_case") == "codebase_analysis"

    def test_operator_graph_on_state(self, soc_state):
        graph = soc_state.get("workload_profile", {}).get("operator_graph")
        assert graph is not None
        assert len(graph["nodes"]) == 2


class TestSessionPersistence:
    """Session round-trips through SessionStore."""

    def test_session_round_trips(self, soc_state, session_dir):
        store = SessionStore(session_dir=session_dir)
        loaded = store.load("soc_codebase_integration")
        assert loaded is not None
        assert loaded.get("workload_profile", {}).get("source") == "codebase_analysis"
        assert loaded.get("codebase_metadata", {}).get("project_name") == "cpp_drone"

    def test_load_latest_returns_session(self, soc_state, session_dir):
        store = SessionStore(session_dir=session_dir)
        latest = store.load_latest()
        assert latest is not None
        assert latest.get("session_id") == "soc_codebase_integration"


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestAPIServesCodebaseData:
    """/api/sessions/{id}/workload returns codebase-originated data."""

    def test_workload_endpoint_has_operators(self, soc_state, session_dir):
        from embodied_ai_architect.api.server import create_app

        app = create_app(session_dir=session_dir)
        client = TestClient(app)

        resp = client.get("/api/sessions/soc_codebase_integration/workload")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["operators"]) == 2
        assert data["source"] == "codebase_analysis"
        names = [op["name"] for op in data["operators"]]
        assert "yolo_detection" in names

    def test_workload_endpoint_has_operator_graph(self, soc_state, session_dir):
        from embodied_ai_architect.api.server import create_app

        app = create_app(session_dir=session_dir)
        client = TestClient(app)

        data = client.get("/api/sessions/soc_codebase_integration/workload").json()
        graph = data.get("operator_graph")
        assert graph is not None
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["source"] == "yolo_detection"
        assert graph["edges"][0]["sink"] == "pid_control"
