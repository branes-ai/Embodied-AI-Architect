"""Seam S12 (issue #217): the single convergence condition — empty backlog OR
critic diminishing-returns OR relative windowed hypervolume plateau (the iteration
cap is the router's job)."""

import json

from embodied_ai_architect.graphs.design_state import (
    AbstractionLevel,
    DesignIssue,
    DesignState,
    MetricAxis,
    Severity,
    has_converged,
)
from embodied_ai_architect.graphs.loop_agents import Critic


def _issue_state(**hv: object) -> DesignState:
    """A state with one open issue (backlog non-empty) so convergence must come
    from a signal other than the empty backlog."""
    issue = DesignIssue(
        metric=MetricAxis.POWER,
        level=AbstractionLevel.SYSTEM,
        severity=Severity.HIGH,
        summary="power failing",
        raised_by="test",
    )
    return {"open_issues": [issue.model_dump(mode="json")], **hv}


# ---------------------------------------------------------------------------
# The three convergence signals
# ---------------------------------------------------------------------------


def test_empty_backlog_converges() -> None:
    assert has_converged({"open_issues": []}) is True


def test_diminishing_returns_converges_even_with_open_issues() -> None:
    state = _issue_state(hypervolume_history=[1.0, 2.0, 4.0])  # still improving
    assert has_converged(state) is False
    state["critic_diminishing_returns"] = True
    assert has_converged(state) is True


def test_relative_windowed_plateau_converges() -> None:
    # Flat frontier over the window -> converged despite an open issue.
    assert has_converged(_issue_state(hypervolume_history=[5.0, 5.0, 5.0])) is True


def test_single_flat_step_does_not_prematurely_converge() -> None:
    # One flat step inside an otherwise-improving run must NOT stop the loop
    # (the old absolute single-step epsilon would have).
    assert has_converged(_issue_state(hypervolume_history=[1.0, 2.0, 2.0])) is False


def test_steadily_improving_frontier_does_not_converge() -> None:
    assert has_converged(_issue_state(hypervolume_history=[1.0, 2.0, 3.0, 4.0])) is False


# ---------------------------------------------------------------------------
# The critic emits the diminishing-returns judgment (LLM path)
# ---------------------------------------------------------------------------


class _Stub:
    def __init__(self, text: str) -> None:
        self.text = text


class _Client:
    def __init__(self, text: str) -> None:
        self._t = text

    def chat(self, messages, system):  # noqa: ANN001
        return _Stub(self._t)


def test_critic_parses_diminishing_returns() -> None:
    payload = json.dumps(
        {
            "converged": False,
            "diminishing_returns": True,
            "issues": [{"metric": "power", "severity": "high", "summary": "stuck"}],
            "deltas": [],
        }
    )
    state = {"ppa_metrics": {"verdicts": {"power": "FAIL"}}}
    verdict = Critic(llm_available=True, llm_client=_Client(payload)).review(state)
    assert verdict.diminishing_returns is True


def test_critic_diminishing_returns_string_false_is_falsey() -> None:
    payload = json.dumps({"diminishing_returns": "false", "issues": [], "deltas": []})
    v = Critic(llm_available=True, llm_client=_Client(payload)).review(
        {"ppa_metrics": {"verdicts": {"power": "PASS"}}}
    )
    assert v.diminishing_returns is False
