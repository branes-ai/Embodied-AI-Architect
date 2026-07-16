"""Seam S8 (issue #214): a SPECIALIST_RETASK delta causes the named specialist to
re-run (filing fresh DesignIssues) and drains `pending_specialist_tasks`."""

from embodied_ai_architect.graphs.design_state import (
    DesignConstraints,
    DesignDelta,
    DesignState,
    DeltaKind,
    MetricAxis,
    open_issues,
)
from embodied_ai_architect.graphs.loop_agents import _run_pending_retasks, optimizer_node


def _hot_state() -> DesignState:
    """A design over the power budget and thermally infeasible (20W -> 160°C)."""
    return {
        "constraints": DesignConstraints(max_power_watts=5.0).model_dump(),
        "ppa_metrics": {"power_watts": 20.0},
        "iteration": 1,
    }


def test_retask_reruns_named_specialist_and_drains_queue() -> None:
    state = _hot_state()
    state["pending_specialist_tasks"] = [{"specialist": "thermal_specialist"}]

    ran = _run_pending_retasks(state)

    assert ran == 1
    assert state["pending_specialist_tasks"] == []  # queue drained
    filed = open_issues(state)
    assert any(
        i.raised_by == "thermal_specialist" and i.metric == MetricAxis.THERMAL for i in filed
    )


def test_unknown_specialist_is_skipped_not_fatal() -> None:
    state = _hot_state()
    state["pending_specialist_tasks"] = [
        {"specialist": "does_not_exist"},
        {"specialist": "ppa_specialist"},
    ]
    ran = _run_pending_retasks(state)
    assert ran == 1  # only the real one ran
    assert state["pending_specialist_tasks"] == []


def test_retask_flows_through_optimizer_node() -> None:
    """End-to-end: a SPECIALIST_RETASK delta enqueues a task, and optimizer_node
    both applies it and re-runs the specialist in the same pass."""
    state = _hot_state()
    retask = DesignDelta(
        kind=DeltaKind.SPECIALIST_RETASK,
        target="thermal_specialist",
        change={"reason": "power just changed — re-check thermals"},
        rationale="config changed",
    )
    state["pending_deltas"] = [retask.model_dump(mode="json")]

    def fake_moo(s: DesignState) -> dict:
        return {"hypervolume_history": s.get("hypervolume_history", []) + [1.0]}

    result = optimizer_node(state, moo_tool=fake_moo)

    assert result["pending_specialist_tasks"] == []  # drained in the return
    assert any(i.get("raised_by") == "thermal_specialist" for i in result.get("open_issues", []))


def test_no_retasks_is_a_noop() -> None:
    assert _run_pending_retasks(_hot_state()) == 0
