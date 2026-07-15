"""Seam S4 (issue #209): Critic._review_with_llm parses a JSON verdict into typed
DesignIssues + DesignDeltas, with graceful fallback to the heuristic path."""

import json
from typing import Any

from embodied_ai_architect.graphs.design_state import (
    AbstractionLevel,
    DeltaKind,
    DesignState,
    MetricAxis,
    Severity,
)
from embodied_ai_architect.graphs.loop_agents import Critic


class _StubResp:
    def __init__(self, text: str) -> None:
        self.text = text


class _StubClient:
    """Returns a fixed text payload regardless of prompt."""

    def __init__(self, text: str) -> None:
        self._text = text

    def chat(self, messages: list[dict[str, Any]], system: str) -> _StubResp:
        return _StubResp(self._text)


class _RaisingClient:
    def chat(self, messages: list[dict[str, Any]], system: str) -> _StubResp:
        raise RuntimeError("boom")


def _state() -> DesignState:
    return {
        "mission_description": "Drone perception at 5 m/s within 5W",
        "platform": "drone",
        "iteration": 1,
        "ppa_metrics": {"verdicts": {"power_watts": "FAIL"}, "power_watts": 6.0},
        "constraints": {"max_power_watts": 5.0},
    }


def _critic(text: str) -> Critic:
    return Critic(llm_available=True, llm_client=_StubClient(text))


# ---------------------------------------------------------------------------


def test_llm_verdict_parses_typed_issues_and_deltas() -> None:
    payload = json.dumps(
        {
            "converged": False,
            "analysis": "power dominated by the detector",
            "research_citations": ["accelerators/stillwater_kpu.md"],
            "issues": [
                {
                    "metric": "power",
                    "level": "system",
                    "severity": "critical",
                    "component": "detector",
                    "summary": "power over budget",
                    "observed_value": 6.0,
                    "target_value": 5.0,
                    "contribution_pct": 40.0,
                }
            ],
            "deltas": [
                {
                    "kind": "design_space_edit",
                    "target": "quantization_dtype",
                    "change": {"value": "int8"},
                    "rationale": "int8 cuts detector power",
                    "addresses_issue": 0,
                }
            ],
        }
    )
    v = _critic(payload).review(_state())
    assert len(v.issues) == 1
    assert v.issues[0].metric == MetricAxis.POWER
    assert v.issues[0].severity == Severity.CRITICAL
    assert len(v.deltas) == 1
    assert v.deltas[0].kind == DeltaKind.DESIGN_SPACE_EDIT
    assert v.deltas[0].typed_change().value == "int8"
    # delta linked to the issue it addresses (both directions)
    assert v.deltas[0].addresses_issue_ids == [v.issues[0].id]
    assert v.deltas[0].id in v.issues[0].delta_ids
    assert v.analysis == "power dominated by the detector"
    assert v.research_citations == ["accelerators/stillwater_kpu.md"]
    assert v.converged is False


def test_malformed_delta_is_skipped_not_fatal() -> None:
    payload = json.dumps(
        {
            "issues": [{"metric": "latency", "severity": "high", "summary": "latency"}],
            "deltas": [
                {"kind": "design_space_edit", "target": "x", "change": {}, "rationale": "r"},
                {
                    "kind": "constraint_relaxation",
                    "target": "max_power_watts",
                    "change": {"to": 6.0},
                    "rationale": "loosen",
                    "addresses_issue": 0,
                },
            ],
        }
    )
    v = _critic(payload).review(_state())
    # first delta (missing required "value") dropped; second survives
    assert len(v.deltas) == 1
    assert v.deltas[0].kind == DeltaKind.CONSTRAINT_RELAXATION


def test_unknown_enum_values_fall_back() -> None:
    payload = json.dumps(
        {
            "issues": [
                {
                    "metric": "quantum_flux",
                    "level": "galactic",
                    "severity": "apocalyptic",
                    "summary": "x",
                }
            ],
            "deltas": [],
        }
    )
    v = _critic(payload).review(_state())
    assert v.issues[0].metric == MetricAxis.POWER  # _to_metric_axis default
    assert v.issues[0].level == AbstractionLevel.SYSTEM
    assert v.issues[0].severity == Severity.MEDIUM


def test_llm_error_falls_back_to_heuristic() -> None:
    critic = Critic(llm_available=True, llm_client=_RaisingClient())
    v = critic.review(_state())
    # heuristic derives a power issue from the failing verdict
    assert any(i.metric == MetricAxis.POWER for i in v.issues)


def test_no_key_uses_heuristic() -> None:
    v = Critic(llm_available=False).review(_state())
    assert any(i.metric == MetricAxis.POWER for i in v.issues)


def test_delta_links_survive_skipped_issue() -> None:
    """addresses_issue must index the ORIGINAL LLM array, not the compacted list.

    Issue 0 is malformed (non-numeric observed_value -> ValidationError -> skipped),
    so `issues` compacts to one element. A delta addressing original index 1 must
    still link to the surviving issue, not fail or slide onto the wrong one.
    """
    payload = json.dumps(
        {
            "issues": [
                {"metric": "power", "summary": "bad", "observed_value": "not-a-number"},
                {"metric": "latency", "severity": "high", "summary": "latency high"},
            ],
            "deltas": [
                {
                    "kind": "design_space_edit",
                    "target": "clock_mhz",
                    "change": {"value": 800},
                    "rationale": "raise clock",
                    "addresses_issue": 1,
                }
            ],
        }
    )
    v = _critic(payload).review(_state())
    assert len(v.issues) == 1  # issue 0 was skipped
    assert v.issues[0].metric == MetricAxis.LATENCY  # the surviving one
    assert v.deltas[0].addresses_issue_ids == [v.issues[0].id]
    assert v.deltas[0].id in v.issues[0].delta_ids


def test_converged_string_false_is_not_truthy() -> None:
    """A JSON string 'false' must not be treated as converged (plain bool() would)."""
    payload = json.dumps({"converged": "false", "issues": [], "deltas": []})
    # no failing verdict in this state so the FAIL-guard doesn't mask the coercion
    state = {"ppa_metrics": {"verdicts": {"power_watts": "PASS"}}}
    v = Critic(llm_available=True, llm_client=_StubClient(payload)).review(state)
    assert v.converged is False


def test_converged_true_rejected_while_constraint_fails() -> None:
    """Even if the LLM says converged, a FAILing verdict forces converged=False."""
    payload = json.dumps({"converged": True, "issues": [], "deltas": []})
    v = _critic(payload).review(_state())  # _state has power_watts FAIL
    assert v.converged is False
