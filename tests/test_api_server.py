"""Tests for the REST API server.

Tests every endpoint for:
- Happy path with expected response structure and values
- 404 for nonexistent sessions
- Empty/minimal data handling
- Response field types and content validation

Requires: pip install fastapi uvicorn httpx
Skipped automatically if fastapi is not installed (core CI).
"""

import pytest

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from embodied_ai_architect.graphs.session_store import SessionStore  # noqa: E402
from embodied_ai_architect.graphs.soc_state import (  # noqa: E402
    DesignConstraints,
    create_initial_soc_state,
)
from embodied_ai_architect.graphs.task_graph import TaskGraph, TaskNode  # noqa: E402

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_dir(tmp_path):
    return tmp_path


def _build_rich_state(session_id="soc_testapi"):
    """Build a session state with data in every field the API reads."""
    state = create_initial_soc_state(
        goal="Test drone SoC for API testing",
        constraints=DesignConstraints(
            max_power_watts=5.0,
            max_latency_ms=33.3,
            max_cost_usd=30.0,
        ),
        use_case="test_drone",
        platform="drone",
        session_id=session_id,
    )

    # PPA metrics with verdicts
    state["ppa_metrics"] = {
        "power_watts": 4.5,
        "latency_ms": 28.0,
        "area_mm2": 45.0,
        "cost_usd": 25.0,
        "verdicts": {"power": "PASS", "latency": "PASS", "cost": "PASS"},
        "cost_breakdown": {
            "die_cost_usd": 8.50,
            "package_cost_usd": 3.00,
            "test_cost_usd": 1.50,
            "nre_per_unit_usd": 12.00,
        },
    }

    # Optimization history
    state["iteration"] = 3
    state["optimization_history"] = [
        {
            "iteration": 0,
            "ppa_snapshot": {"power_watts": 7.0, "latency_ms": 40.0, "cost_usd": 38.0},
            "verdicts": {"power": "FAIL", "latency": "FAIL", "cost": "FAIL"},
        },
        {
            "iteration": 1,
            "ppa_snapshot": {"power_watts": 5.5, "latency_ms": 32.0, "cost_usd": 30.0},
            "verdicts": {"power": "FAIL", "latency": "PASS", "cost": "PASS"},
        },
        {
            "iteration": 2,
            "ppa_snapshot": {"power_watts": 4.5, "latency_ms": 28.0, "cost_usd": 25.0},
            "verdicts": {"power": "PASS", "latency": "PASS", "cost": "PASS"},
        },
    ]

    # Task graph with real nodes
    graph = TaskGraph()
    graph.add_tasks(
        [
            TaskNode(id="t1", name="Analyze workload", agent="workload_analyzer"),
            TaskNode(id="t2", name="Explore hardware", agent="hw_explorer", dependencies=["t1"]),
            TaskNode(
                id="t3", name="Compose arch", agent="architecture_composer", dependencies=["t2"]
            ),
        ]
    )
    # Mark completed
    graph.mark_running("t1")
    graph.mark_completed("t1", result={"summary": "done"})
    graph.mark_running("t2")
    graph.mark_completed("t2", result={"summary": "done"})
    state["task_graph"] = graph.to_dict()

    # Workload profile with operators
    state["workload_profile"] = {
        "source": "constraint_estimation",
        "dominant_op": "detection",
        "total_estimated_gflops": 12.4,
        "total_estimated_memory_mb": 24.0,
        "workloads": [
            {
                "name": "yolo_detector",
                "model_class": "YOLOv8",
                "estimated_gflops": 8.4,
                "estimated_memory_mb": 16.0,
                "scheduling": "concurrent",
            },
            {
                "name": "tracker",
                "model_class": "ByteTrack",
                "estimated_gflops": 2.0,
                "estimated_memory_mb": 4.0,
                "scheduling": "sequential",
            },
            {
                "name": "vio_pipeline",
                "model_class": "stereo_vio",
                "estimated_gflops": 2.0,
                "estimated_memory_mb": 4.0,
                "scheduling": "concurrent",
            },
        ],
    }

    # Pareto points
    state["pareto_points"] = [
        {"power": 4.5, "latency": 28.0, "cost": 25.0, "hardware": "KPU", "dominated": False},
        {"power": 3.0, "latency": 45.0, "cost": 20.0, "hardware": "DSP", "dominated": False},
        {"power": 6.0, "latency": 20.0, "cost": 35.0, "hardware": "GPU", "dominated": False},
        {"power": 7.0, "latency": 35.0, "cost": 40.0, "hardware": "CPU", "dominated": True},
    ]
    state["pareto_results"] = {
        "front": [0, 1, 2],
        "knee_point_index": 0,
    }

    # MOO results with per-variable sensitivity (issue #24).
    # IMPORTANT: this fixture uses the REAL producer format emitted by
    # bayesian_opt._extract_sensitivity — nested as
    # {objective: {variable: {lengthscale, importance}}}.
    # The /sensitivity endpoint normalizes this to {variable: {objective: float}}
    # via normalize_sensitivity(). Tests should exercise this real shape so
    # we catch any consumer/producer drift.
    state["moo_results"] = {
        "total_evaluations": 100,
        "pareto_front": [],
        "sensitivity": {
            "power_watts": {
                "quantization_dtype": {"lengthscale": 0.45, "importance": 0.82},
                "npu_frequency_mhz": {"lengthscale": 0.55, "importance": 0.71},
                "sram_size_kb": {"lengthscale": 2.10, "importance": 0.15},
            },
            "latency_ms": {
                "quantization_dtype": {"lengthscale": 0.62, "importance": 0.65},
                "npu_frequency_mhz": {"lengthscale": 0.70, "importance": 0.58},
                "sram_size_kb": {"lengthscale": 0.50, "importance": 0.72},
            },
            "cost_usd": {
                "quantization_dtype": {"lengthscale": 1.80, "importance": 0.12},
                "npu_frequency_mhz": {"lengthscale": 0.85, "importance": 0.45},
                "sram_size_kb": {"lengthscale": 0.95, "importance": 0.38},
            },
        },
    }

    # Design rationale
    state["design_rationale"] = [
        "[planner] Created task graph with 3 tasks",
        "[workload_analyzer] Analyzed perception workload: 12.4 GFLOPS",
        "[optimizer] Applied quantize_int8: reduced power 20%",
    ]

    # KPU inner-loop slackness inputs (issue #30) — floorplan_estimate +
    # bandwidth_match are produced by the KPU validators when rtl_enabled.
    # The test fixture uses pre-fabricated dicts so the API can read them.
    state["floorplan_estimate"] = {
        "compute_tile": {
            "width_mm": 2.10,
            "height_mm": 2.30,
            "area_mm2": 4.83,
            "sub_blocks": [],
        },
        "memory_tile": {
            "width_mm": 2.00,
            "height_mm": 2.40,
            "area_mm2": 4.80,
            "sub_blocks": [],
        },
        "pitch_matched": True,
        "pitch_ratio_width": 1.05,
        "pitch_ratio_height": 0.96,
        "pitch_tolerance": 0.15,
        "array_width_mm": 6.30,
        "array_height_mm": 6.90,
        "core_area_mm2": 43.5,
        "periphery_area_mm2": 4.7,
        "total_area_mm2": 48.2,
        "die_edge_mm": 7.0,
        "feasible": True,
        "max_die_area_mm2": 100.0,
        "issues": [],
    }
    # Issue #33: full KPU config (used by /api/sessions/{id}/kpu)
    state["kpu_config"] = {
        "name": "swkpu-test",
        "process_nm": 28,
        "array_rows": 3,
        "array_cols": 3,
        "compute_tile": {
            "num_tiles": 4,
            "array_rows": 16,
            "array_cols": 16,
            "vector_lanes": 16,
            "frequency_mhz": 500.0,
            "supported_precisions": ["int8", "fp16"],
            "l2_size_bytes": 262144,
            "l2_num_banks": 8,
            "l2_read_ports": 2,
            "l2_write_ports": 1,
            "l1_size_bytes": 32768,
            "l1_num_banks": 4,
            "num_streamers": 2,
            "streamer_prefetch_depth": 4,
            "streamer_buffer_bytes": 16384,
        },
        "memory_tile": {
            "l3_tile_size_bytes": 524288,
            "l3_num_banks": 4,
            "num_block_movers": 2,
            "block_mover_bw_gbps": 32.0,
            "num_dma_engines": 1,
            "dma_max_transfer_bytes": 1048576,
            "dma_queue_depth": 8,
        },
        "noc": {
            "topology": "mesh_2d",
            "link_width_bits": 256,
            "frequency_mhz": 1000.0,
            "num_routers": 16,
        },
        "dram": {
            "technology": "LPDDR4X",
            "num_controllers": 2,
            "channels_per_controller": 2,
            "bandwidth_per_channel_gbps": 6.4,
            "capacity_gb": 4.0,
        },
    }
    state["bandwidth_match"] = {
        "links": [
            {
                "name": "DRAM -> L3",
                "source": "dram",
                "sink": "l3",
                "available_gbps": 25.6,
                "required_gbps": 12.8,
                "utilization": 0.50,
                "bottleneck": False,
            },
            {
                "name": "L2 -> L1",
                "source": "l2",
                "sink": "l1",
                "available_gbps": 4.8,
                "required_gbps": 4.5,
                "utilization": 0.94,
                "bottleneck": False,
            },
        ],
        "balanced": True,
        "bottleneck_link": None,
        "peak_utilization": 0.94,
        "ingress_gbps": 12.8,
        "egress_gbps": 1.4,
        "compute_demand_gbps": 12.8,
        "issues": [],
    }

    return state


@pytest.fixture
def rich_state(session_dir):
    """Create a fully populated session on disk."""
    store = SessionStore(session_dir=session_dir)
    state = _build_rich_state()
    store.save(state)
    return state


@pytest.fixture
def minimal_state(session_dir):
    """Create a minimal session with almost no data."""
    store = SessionStore(session_dir=session_dir)
    state = create_initial_soc_state(
        goal="Minimal test",
        session_id="soc_minimal",
    )
    store.save(state)
    return state


@pytest.fixture
def client(session_dir, rich_state):
    """Test client with a rich session."""
    from embodied_ai_architect.api.server import create_app

    app = create_app(session_dir=str(session_dir))
    return TestClient(app)


@pytest.fixture
def client_with_both(session_dir, rich_state, minimal_state):
    """Test client with both rich and minimal sessions."""
    from embodied_ai_architect.api.server import create_app

    app = create_app(session_dir=str(session_dir))
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_status(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_reports_session_count(self, client_with_both):
        resp = client_with_both.get("/api/health")
        assert resp.json()["session_count"] == 2

    def test_health_reports_session_dir(self, client):
        data = client.get("/api/health").json()
        assert "session_dir" in data


# ---------------------------------------------------------------------------
# Session list
# ---------------------------------------------------------------------------


class TestSessionList:
    def test_list_returns_array(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_contains_expected_fields(self, client):
        sessions = client.get("/api/sessions").json()
        s = sessions[0]
        assert s["session_id"] == "soc_testapi"
        assert s["platform"] == "drone"
        assert s["use_case"] == "test_drone"
        assert "status" in s
        assert "iteration" in s
        assert "created_at" in s

    def test_list_multiple_sessions(self, client_with_both):
        sessions = client_with_both.get("/api/sessions").json()
        assert len(sessions) == 2
        ids = {s["session_id"] for s in sessions}
        assert "soc_testapi" in ids
        assert "soc_minimal" in ids

    def test_list_empty_store(self, tmp_path):
        from embodied_ai_architect.api.server import create_app

        app = create_app(session_dir=str(tmp_path / "empty"))
        c = TestClient(app)
        assert c.get("/api/sessions").json() == []


# ---------------------------------------------------------------------------
# Session detail
# ---------------------------------------------------------------------------


class TestSessionDetail:
    def test_get_returns_full_state(self, client):
        resp = client.get("/api/sessions/soc_testapi")
        assert resp.status_code == 200
        data = resp.json()
        assert data["goal"] == "Test drone SoC for API testing"
        assert data["platform"] == "drone"
        assert data["use_case"] == "test_drone"

    def test_get_includes_ppa_metrics(self, client):
        data = client.get("/api/sessions/soc_testapi").json()
        ppa = data["ppa_metrics"]
        assert ppa["power_watts"] == 4.5
        assert ppa["latency_ms"] == 28.0
        assert ppa["verdicts"]["power"] == "PASS"

    def test_get_includes_constraints(self, client):
        data = client.get("/api/sessions/soc_testapi").json()
        c = data["constraints"]
        assert c["max_power_watts"] == 5.0
        assert c["max_latency_ms"] == 33.3

    def test_get_includes_design_rationale(self, client):
        data = client.get("/api/sessions/soc_testapi").json()
        assert len(data["design_rationale"]) == 3
        assert "quantize_int8" in data["design_rationale"][2]

    def test_get_nonexistent_returns_404(self, client):
        resp = client.get("/api/sessions/soc_nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_get_minimal_session(self, client_with_both):
        resp = client_with_both.get("/api/sessions/soc_minimal")
        assert resp.status_code == 200
        data = resp.json()
        assert data["goal"] == "Minimal test"
        assert data["ppa_metrics"] == {}


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------


class TestPareto:
    def test_pareto_structure(self, client):
        resp = client.get("/api/sessions/soc_testapi/pareto")
        assert resp.status_code == 200
        data = resp.json()
        assert "points" in data
        assert "front" in data
        assert "knee_point_index" in data
        assert "objectives" in data

    def test_pareto_points_content(self, client):
        data = client.get("/api/sessions/soc_testapi/pareto").json()
        assert len(data["points"]) == 4
        kpu = data["points"][0]
        assert kpu["power"] == 4.5
        assert kpu["hardware"] == "KPU"
        assert kpu["dominated"] is False

    def test_pareto_front_indices(self, client):
        data = client.get("/api/sessions/soc_testapi/pareto").json()
        assert data["front"] == [0, 1, 2]
        assert data["knee_point_index"] == 0

    def test_pareto_objectives(self, client):
        data = client.get("/api/sessions/soc_testapi/pareto").json()
        assert data["objectives"] == ["power_watts", "latency_ms", "cost_usd"]

    def test_pareto_empty_session(self, client_with_both):
        data = client_with_both.get("/api/sessions/soc_minimal/pareto").json()
        assert data["points"] == []

    def test_pareto_404(self, client):
        assert client.get("/api/sessions/soc_nope/pareto").status_code == 404


# ---------------------------------------------------------------------------
# Slackness
# ---------------------------------------------------------------------------


class TestSlackness:
    def test_slackness_returns_array(self, client):
        resp = client.get("/api/sessions/soc_testapi/slackness")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_slackness_has_constraint_fields(self, client):
        data = client.get("/api/sessions/soc_testapi/slackness").json()
        assert len(data) >= 2  # power and latency at minimum
        cs = data[0]
        for field in ("name", "target", "actual", "unit", "verdict", "margin_pct"):
            assert field in cs, f"Missing field: {field}"

    def test_slackness_values_correct(self, client):
        data = client.get("/api/sessions/soc_testapi/slackness").json()
        power = next(cs for cs in data if cs["name"] == "power")
        assert power["target"] == 5.0
        assert power["actual"] == 4.5
        assert power["verdict"] == "PASS"
        assert power["margin_pct"] > 0  # 10% slack

    def test_slackness_empty_session(self, client_with_both):
        data = client_with_both.get("/api/sessions/soc_minimal/slackness").json()
        assert isinstance(data, list)
        # Minimal session has no constraints, so empty or minimal slackness
        assert len(data) == 0

    def test_slackness_404(self, client):
        assert client.get("/api/sessions/soc_nope/slackness").status_code == 404


# ---------------------------------------------------------------------------
# KPU slackness (issue #30)
# ---------------------------------------------------------------------------


class TestKPUSlackness:
    def test_endpoint_returns_200(self, client):
        resp = client.get("/api/sessions/soc_testapi/kpu-slackness")
        assert resp.status_code == 200

    def test_response_has_floorplan_and_bandwidth(self, client):
        data = client.get("/api/sessions/soc_testapi/kpu-slackness").json()
        assert "floorplan" in data
        assert "bandwidth" in data
        assert data["floorplan"] is not None
        assert data["bandwidth"] is not None

    def test_floorplan_fields(self, client):
        data = client.get("/api/sessions/soc_testapi/kpu-slackness").json()
        fp = data["floorplan"]
        assert fp["compute_tile_width_mm"] == 2.10
        assert fp["memory_tile_width_mm"] == 2.00
        assert fp["pitch_matched"] is True
        assert fp["feasible"] is True
        assert fp["total_area_mm2"] == 48.2
        assert fp["max_die_area_mm2"] == 100.0
        assert fp["area_utilization_pct"] == 48.2

    def test_bandwidth_links_classified(self, client):
        data = client.get("/api/sessions/soc_testapi/kpu-slackness").json()
        bw = data["bandwidth"]
        assert bw["balanced"] is True
        assert len(bw["links"]) == 2
        statuses = {link["name"]: link["status"] for link in bw["links"]}
        assert statuses["DRAM -> L3"] == "OK"
        assert statuses["L2 -> L1"] == "TIGHT"  # 94% utilization

    def test_returns_null_when_no_kpu_state(self, client_with_both):
        """Minimal session has no floorplan/bandwidth → both fields None."""
        data = client_with_both.get("/api/sessions/soc_minimal/kpu-slackness").json()
        assert data["floorplan"] is None
        assert data["bandwidth"] is None

    def test_404(self, client):
        assert client.get("/api/sessions/soc_nope/kpu-slackness").status_code == 404

    def test_mixed_cached_and_raw_fallback(self, session_dir, rich_state):
        """CodeRabbit PR #80: per-field fallback when only one half is cached.

        Build a session whose optimization_review_snapshot caches only the
        floorplan, while bandwidth lives only in the raw state. The endpoint
        must return BOTH halves.
        """
        from embodied_ai_architect.api.server import create_app
        from embodied_ai_architect.graphs.session_store import SessionStore

        # Create a fresh state with cached snapshot.kpu_floorplan but no
        # cached snapshot.kpu_bandwidth — bandwidth should come from raw state.
        store = SessionStore(session_dir=session_dir)
        state = store.load("soc_testapi")
        cached_floorplan = {
            "compute_tile_width_mm": 2.5,
            "compute_tile_height_mm": 2.5,
            "memory_tile_width_mm": 2.5,
            "memory_tile_height_mm": 2.5,
            "pitch_ratio_width": 1.0,
            "pitch_ratio_height": 1.0,
            "pitch_tolerance": 0.15,
            "pitch_matched": True,
            "total_area_mm2": 50.0,
            "max_die_area_mm2": 100.0,
            "area_utilization_pct": 50.0,
            "feasible": True,
            "issues": [],
        }
        state["optimization_review_snapshot"] = {
            "kpu_floorplan": cached_floorplan,
            # NOTE: no kpu_bandwidth — must fall back to bandwidth_match
        }
        state["session_id"] = "soc_mixed"
        store.save(state)

        app = create_app(session_dir=str(session_dir))
        client = TestClient(app)

        data = client.get("/api/sessions/soc_mixed/kpu-slackness").json()
        # Cached half came through verbatim
        assert data["floorplan"]["compute_tile_width_mm"] == 2.5
        # Uncached half was extracted from raw bandwidth_match
        assert data["bandwidth"] is not None
        assert len(data["bandwidth"]["links"]) == 2


# ---------------------------------------------------------------------------
# KPU full snapshot endpoint (issue #33)
# ---------------------------------------------------------------------------


class TestKPUEndpoint:
    """Tests for /api/sessions/{id}/kpu — the full KPU snapshot served to
    /architect-drill (issue #33). Combines kpu_config + floorplan + bandwidth."""

    def test_endpoint_returns_200(self, client):
        resp = client.get("/api/sessions/soc_testapi/kpu")
        assert resp.status_code == 200

    def test_response_has_three_sections(self, client):
        data = client.get("/api/sessions/soc_testapi/kpu").json()
        assert "config" in data
        assert "floorplan" in data
        assert "bandwidth" in data
        assert data["config"] is not None
        assert data["floorplan"] is not None
        assert data["bandwidth"] is not None

    def test_config_carries_full_kpu_microarch(self, client):
        """The config field must carry the full KPUMicroArchConfig dump
        so /architect-drill kpu can render compute, memory, NoC, DRAM."""
        data = client.get("/api/sessions/soc_testapi/kpu").json()
        config = data["config"]
        assert config["name"] == "swkpu-test"
        assert config["process_nm"] == 28
        # Compute tile knobs
        assert config["compute_tile"]["array_rows"] == 16
        assert config["compute_tile"]["array_cols"] == 16
        assert config["compute_tile"]["frequency_mhz"] == 500.0
        assert config["compute_tile"]["l2_size_bytes"] == 262144
        # Memory tile knobs
        assert config["memory_tile"]["l3_tile_size_bytes"] == 524288
        # NoC and DRAM
        assert config["noc"]["topology"] == "mesh_2d"
        assert config["noc"]["link_width_bits"] == 256
        assert config["dram"]["technology"] == "LPDDR4X"

    def test_returns_null_config_for_minimal_session(self, client_with_both):
        """A session without kpu_config returns config=null + None slackness."""
        data = client_with_both.get("/api/sessions/soc_minimal/kpu").json()
        assert data["config"] is None
        assert data["floorplan"] is None
        assert data["bandwidth"] is None

    def test_404_for_nonexistent_session(self, client):
        assert client.get("/api/sessions/soc_nope/kpu").status_code == 404


# ---------------------------------------------------------------------------
# Sensitivity (issue #24)
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_sensitivity_structure(self, client):
        resp = client.get("/api/sessions/soc_testapi/sensitivity")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "objectives" in data

    def test_sensitivity_entries_present(self, client):
        data = client.get("/api/sessions/soc_testapi/sensitivity").json()
        assert len(data["entries"]) == 3
        var_names = {e["variable"] for e in data["entries"]}
        assert var_names == {"quantization_dtype", "npu_frequency_mhz", "sram_size_kb"}

    def test_sensitivity_ranked_by_max_impact(self, client):
        """Variables must be sorted by max impact descending."""
        data = client.get("/api/sessions/soc_testapi/sensitivity").json()
        max_impacts = [e["max_impact"] for e in data["entries"]]
        assert max_impacts == sorted(max_impacts, reverse=True)
        # quantization_dtype has max impact 0.82 → should be first
        assert data["entries"][0]["variable"] == "quantization_dtype"
        assert data["entries"][0]["max_impact"] == 0.82

    def test_sensitivity_impacts_dict_contents(self, client):
        data = client.get("/api/sessions/soc_testapi/sensitivity").json()
        first = data["entries"][0]
        assert first["impacts"]["power_watts"] == 0.82
        assert first["impacts"]["latency_ms"] == 0.65

    def test_sensitivity_objectives_listed(self, client):
        data = client.get("/api/sessions/soc_testapi/sensitivity").json()
        # All three objective columns should appear
        assert set(data["objectives"]) == {"power_watts", "latency_ms", "cost_usd"}

    def test_sensitivity_empty_for_minimal_session(self, client_with_both):
        """Sessions without MOO results return empty sensitivity."""
        data = client_with_both.get("/api/sessions/soc_minimal/sensitivity").json()
        assert data["entries"] == []
        assert data["objectives"] == []

    def test_sensitivity_404(self, client):
        assert client.get("/api/sessions/soc_nope/sensitivity").status_code == 404


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_trajectory_returns_array(self, client):
        resp = client.get("/api/sessions/soc_testapi/trajectory")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_trajectory_iteration_count(self, client):
        data = client.get("/api/sessions/soc_testapi/trajectory").json()
        assert len(data) == 3

    def test_trajectory_iteration_order(self, client):
        data = client.get("/api/sessions/soc_testapi/trajectory").json()
        iterations = [d["iteration"] for d in data]
        assert iterations == [0, 1, 2]

    def test_trajectory_ppa_values(self, client):
        data = client.get("/api/sessions/soc_testapi/trajectory").json()
        # First iteration: high power, failing
        assert data[0]["ppa_snapshot"]["power_watts"] == 7.0
        assert data[0]["verdicts"]["power"] == "FAIL"
        # Last iteration: converged
        assert data[2]["ppa_snapshot"]["power_watts"] == 4.5
        assert data[2]["verdicts"]["power"] == "PASS"

    def test_trajectory_shows_convergence(self, client):
        data = client.get("/api/sessions/soc_testapi/trajectory").json()
        powers = [d["ppa_snapshot"]["power_watts"] for d in data]
        assert powers == [7.0, 5.5, 4.5]  # monotonically decreasing

    def test_trajectory_empty_session(self, client_with_both):
        data = client_with_both.get("/api/sessions/soc_minimal/trajectory").json()
        assert data == []

    def test_trajectory_404(self, client):
        assert client.get("/api/sessions/soc_nope/trajectory").status_code == 404


# ---------------------------------------------------------------------------
# Task graph
# ---------------------------------------------------------------------------


class TestTaskGraph:
    def test_taskgraph_structure(self, client):
        resp = client.get("/api/sessions/soc_testapi/taskgraph")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "execution_order" in data
        assert "parallel_groups" in data

    def test_taskgraph_node_count(self, client):
        data = client.get("/api/sessions/soc_testapi/taskgraph").json()
        assert len(data["nodes"]) == 3

    def test_taskgraph_node_fields(self, client):
        data = client.get("/api/sessions/soc_testapi/taskgraph").json()
        node = data["nodes"][0]
        for field in ("id", "name", "agent", "status", "dependencies"):
            assert field in node, f"Missing field: {field}"

    def test_taskgraph_node_status(self, client):
        data = client.get("/api/sessions/soc_testapi/taskgraph").json()
        statuses = {n["id"]: n["status"] for n in data["nodes"]}
        assert statuses["t1"] == "completed"
        assert statuses["t2"] == "completed"
        assert statuses["t3"] == "ready"  # deps satisfied but not run

    def test_taskgraph_dependencies(self, client):
        data = client.get("/api/sessions/soc_testapi/taskgraph").json()
        deps = {n["id"]: n["dependencies"] for n in data["nodes"]}
        assert deps["t1"] == []
        assert deps["t2"] == ["t1"]
        assert deps["t3"] == ["t2"]

    def test_taskgraph_execution_order(self, client):
        data = client.get("/api/sessions/soc_testapi/taskgraph").json()
        assert data["execution_order"] == ["t1", "t2", "t3"]

    def test_taskgraph_parallel_groups(self, client):
        data = client.get("/api/sessions/soc_testapi/taskgraph").json()
        assert len(data["parallel_groups"]) == 3  # linear chain = 3 groups

    def test_taskgraph_empty_session(self, client_with_both):
        data = client_with_both.get("/api/sessions/soc_minimal/taskgraph").json()
        assert data["nodes"] == []

    def test_taskgraph_404(self, client):
        assert client.get("/api/sessions/soc_nope/taskgraph").status_code == 404


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


class TestWorkload:
    def test_workload_structure(self, client):
        resp = client.get("/api/sessions/soc_testapi/workload")
        assert resp.status_code == 200
        data = resp.json()
        assert "operators" in data
        assert "total_gflops" in data
        assert "total_memory_mb" in data
        assert "dominant_op" in data

    def test_workload_operator_count(self, client):
        data = client.get("/api/sessions/soc_testapi/workload").json()
        assert len(data["operators"]) == 3

    def test_workload_operator_fields(self, client):
        data = client.get("/api/sessions/soc_testapi/workload").json()
        op = data["operators"][0]
        assert op["name"] == "yolo_detector"
        assert op["model_class"] == "YOLOv8"
        assert op["gflops"] == 8.4
        assert op["memory_mb"] == 16.0

    def test_workload_totals(self, client):
        data = client.get("/api/sessions/soc_testapi/workload").json()
        assert data["total_gflops"] == 12.4
        assert data["total_memory_mb"] == 24.0
        assert data["dominant_op"] == "detection"

    def test_workload_empty_session(self, client_with_both):
        data = client_with_both.get("/api/sessions/soc_minimal/workload").json()
        assert data["operators"] == []
        assert data["total_gflops"] is None

    def test_workload_404(self, client):
        assert client.get("/api/sessions/soc_nope/workload").status_code == 404


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORS:
    def test_cors_allows_localhost_3000(self, client):
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_cors_rejects_unknown_origin(self, client):
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") != "http://evil.com"


# ---------------------------------------------------------------------------
# Path traversal protection
# ---------------------------------------------------------------------------


class TestPathTraversal:
    """Session ID validation prevents path traversal.

    Note: Starlette normalizes ../.. in URLs before routing, so those
    never reach the handler (they get 404 from the router). Our validation
    catches cases that bypass URL normalization (e.g., encoded dots,
    non-soc prefixes).
    """

    def test_rejects_non_soc_prefix(self, client):
        """IDs without soc_ prefix are rejected with 400."""
        resp = client.get("/api/sessions/malicious_id")
        assert resp.status_code == 400
        assert "Invalid session ID" in resp.json()["detail"]

    def test_rejects_dots_in_id(self, client):
        """IDs containing dots are rejected."""
        resp = client.get("/api/sessions/soc_..test")
        assert resp.status_code == 400

    def test_rejects_spaces(self, client):
        resp = client.get("/api/sessions/soc_test%20injection")
        assert resp.status_code == 400

    def test_rejects_special_characters(self, client):
        for bad_id in ["soc_test;rm", "soc_test$(cmd)", "soc_test&evil"]:
            resp = client.get(f"/api/sessions/{bad_id}")
            assert resp.status_code == 400, f"Should reject: {bad_id}"

    def test_accepts_valid_session_id(self, client):
        resp = client.get("/api/sessions/soc_testapi")
        assert resp.status_code == 200

    def test_accepts_hex_session_id(self, client):
        """soc_fa0074e8 format should be accepted."""
        resp = client.get("/api/sessions/soc_fa0074e8")
        assert resp.status_code == 404  # not found, but format is valid

    def test_accepts_hyphenated_session_id(self, client):
        resp = client.get("/api/sessions/soc_test-run-1")
        assert resp.status_code == 404  # format valid, session doesn't exist

    def test_validation_on_derived_endpoints(self, client):
        """All derived endpoints validate session_id."""
        for suffix in ["/pareto", "/slackness", "/trajectory", "/taskgraph", "/workload"]:
            resp = client.get(f"/api/sessions/bad_id{suffix}")
            assert resp.status_code == 400, f"Validation missing on {suffix}"

    def test_validation_on_stream(self, client):
        resp = client.get("/api/sessions/bad_id/stream")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# SSE Streaming
# ---------------------------------------------------------------------------


class TestSSEStream:
    """Test SSE streaming components.

    The async generator with asyncio.sleep cannot be tested via TestClient
    (it blocks). We test the helper functions directly and the HTTP 404.
    """

    def test_stream_http_404_for_nonexistent(self, client):
        """HTTP endpoint should return 404 for nonexistent session."""
        resp = client.get("/api/sessions/soc_nonexistent/stream")
        assert resp.status_code == 404

    def test_extract_update_fields(self, session_dir, rich_state):
        """_extract_update should return the expected field set."""
        from embodied_ai_architect.api.server import _extract_update

        store = SessionStore(session_dir=session_dir)
        state = store.load("soc_testapi")
        update = _extract_update(state)
        assert update["session_id"] == "soc_testapi"
        assert update["status"] == state.get("status")  # matches whatever fixture sets
        assert update["iteration"] == 3
        assert update["max_iterations"] == 20
        assert update["ppa_metrics"]["power_watts"] == 4.5
        assert update["ppa_metrics"]["latency_ms"] == 28.0
        assert update["verdicts"]["power"] == "PASS"

    def test_extract_update_empty_state(self):
        """_extract_update handles empty state gracefully."""
        from embodied_ai_architect.api.server import _extract_update

        update = _extract_update({})
        assert update["session_id"] == ""
        assert update["status"] == ""
        assert update["iteration"] == 0
        assert update["ppa_metrics"]["power_watts"] is None

    def test_format_sse_state_update(self):
        """SSE format should match the spec: event + data + double newline."""
        from embodied_ai_architect.api.server import _format_sse

        result = _format_sse("state_update", {"status": "ok"})
        assert result == 'event: state_update\ndata: {"status": "ok"}\n\n'

    def test_format_sse_complete(self):
        from embodied_ai_architect.api.server import _format_sse

        result = _format_sse("complete", {"status": "complete", "iteration": 5})
        assert result.startswith("event: complete\n")
        assert "data:" in result
        assert '"iteration": 5' in result
        assert result.endswith("\n\n")

    def test_format_sse_error(self):
        from embodied_ai_architect.api.server import _format_sse

        result = _format_sse("error", {"message": "Session file not found"})
        assert "event: error\n" in result
        assert "Session file not found" in result
