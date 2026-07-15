"""Seam S7 (issue #212): the loop's evaluate_node delegates to the real ppa_assessor,
so its ppa_metrics.verdicts equal the standalone assessor's for the same state."""

from embodied_ai_architect.graphs.design_state import DesignConstraints, DesignState
from embodied_ai_architect.graphs.loop_convergence_graph import evaluate_node
from embodied_ai_architect.graphs.specialists import ppa_assessor
from embodied_ai_architect.graphs.task_graph import TaskNode


def _state() -> DesignState:
    return {
        "constraints": DesignConstraints(
            max_power_watts=5.0,
            max_latency_ms=33.0,
            max_cost_usd=30.0,
            max_area_mm2=50.0,
        ).model_dump(),
        "workload_profile": {"total_gflops": 20.0, "operators": []},
        "ip_blocks": [],
        "selected_architecture": {},
        "hardware_candidates": [],
    }


def test_evaluate_node_verdicts_match_standalone_assessor() -> None:
    state = _state()
    loop_ppa = evaluate_node(state)["ppa_metrics"]
    standalone = ppa_assessor(TaskNode(id="x", name="x", agent="ppa_assessor"), state)[
        "ppa_metrics"
    ]
    assert loop_ppa["verdicts"] == standalone["verdicts"]
    # And the full metric payload matches — it's the same assessor, same state.
    assert loop_ppa == standalone


def test_evaluate_node_produces_pipeline_style_verdicts() -> None:
    """Verdicts are keyed like the rest of the pipeline (power/latency/...), not the
    old bespoke power_watts/latency_ms keys."""
    verdicts = evaluate_node(_state())["ppa_metrics"]["verdicts"]
    assert verdicts  # constraints present -> non-empty
    assert set(verdicts).issubset({"power", "latency", "cost", "area"})
    assert "power_watts" not in verdicts


def test_evaluate_node_writes_only_ppa_metrics() -> None:
    """The node's return dict is confined to the ppa_metrics channel (S1 invariant)."""
    assert set(evaluate_node(_state())) == {"ppa_metrics"}
