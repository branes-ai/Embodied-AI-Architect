"""Seam S4 (issue #209): Critic._review_with_llm parses a JSON verdict into typed
DesignIssues + DesignDeltas, with graceful fallback to the heuristic path."""

import json

from embodied_ai_architect.graphs.design_state import (
    AbstractionLevel,
    DeltaKind,
    MetricAxis,
    Severity,
)
from embodied_ai_architect.graphs.loop_agents import Critic


class _StubResp:
    def __init__(self, text):
        self.text = text


class _StubClient:
    """Returns a fixed text payload regardless of prompt."""

    def __init__(self, text):
        self._text = text

    def chat(self, messages, system):
        return _StubResp(self._text)


class _RaisingClient:
    def chat(self, messages, system):
        raise RuntimeError("boom")


def _state():
    return {
        "mission_description": "Drone perception at 5 m/s within 5W",
        "platform": "drone",
        "iteration": 1,
        "ppa_metrics": {"verdicts": {"power_watts": "FAIL"}, "power_watts": 6.0},
        "constraints": {"max_power_watts": 5.0},
    }


def _critic(text):
    return Critic(llm_available=True, llm_client=_StubClient(text))


# ---------------------------------------------------------------------------


def test_llm_verdict_parses_typed_issues_and_deltas():
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


def test_malformed_delta_is_skipped_not_fatal():
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


def test_unknown_enum_values_fall_back():
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


def test_llm_error_falls_back_to_heuristic():
    critic = Critic(llm_available=True, llm_client=_RaisingClient())
    v = critic.review(_state())
    # heuristic derives a power issue from the failing verdict
    assert any(i.metric == MetricAxis.POWER for i in v.issues)


def test_no_key_uses_heuristic():
    v = Critic(llm_available=False).review(_state())
    assert any(i.metric == MetricAxis.POWER for i in v.issues)
