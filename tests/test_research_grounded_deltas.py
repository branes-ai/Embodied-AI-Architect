"""Seam S5 (issue #210): critic deltas are grounded in the research library —
retrieval tags vary with the mission's bottlenecks, and deltas carry research refs."""

import json
from typing import Any

from embodied_ai_architect.graphs.design_state import DesignState
from embodied_ai_architect.graphs.loop_agents import Critic, _research_tags_for_state


class _StubResp:
    def __init__(self, text: str) -> None:
        self.text = text


class _StubClient:
    def __init__(self, text: str) -> None:
        self._text = text

    def chat(self, messages: list[dict[str, Any]], system: str) -> _StubResp:
        return _StubResp(self._text)


def _power_state() -> DesignState:
    return {"platform": "drone", "ppa_metrics": {"verdicts": {"power_watts": "FAIL"}}}


def _latency_state() -> DesignState:
    return {"platform": "amr", "ppa_metrics": {"verdicts": {"latency_ms": "FAIL"}}}


# ---------------------------------------------------------------------------
# Retrieval tags vary with mission + bottleneck (not a fixed query)
# ---------------------------------------------------------------------------


def test_research_tags_vary_with_bottleneck() -> None:
    pt = _research_tags_for_state(_power_state())
    lt = _research_tags_for_state(_latency_state())
    # platform carried through
    assert "drone" in pt and "amr" in lt
    # bottleneck-specific tags differ
    assert "quantization" in pt and "quantization" not in lt
    assert "dataflow" in lt and "dataflow" not in pt
    assert set(pt) != set(lt)


def test_research_tags_fall_back_without_bottleneck() -> None:
    tags = _research_tags_for_state({"platform": "edge", "ppa_metrics": {"verdicts": {}}})
    assert tags == ["edge", "efficiency"]


def test_research_tags_include_open_issue_metrics() -> None:
    state: DesignState = {
        "platform": "drone",
        "ppa_metrics": {"verdicts": {}},
        "open_issues": [{"metric": "bandwidth", "summary": "bw", "status": "open"}],
    }
    tags = _research_tags_for_state(state)
    assert "memory" in tags and "noc" in tags  # bandwidth research tags


# ---------------------------------------------------------------------------
# Deltas cite retrieved research
# ---------------------------------------------------------------------------


def test_deltas_carry_research_refs_from_llm() -> None:
    payload = json.dumps(
        {
            "issues": [{"metric": "power", "severity": "critical", "summary": "p"}],
            "deltas": [
                {
                    "kind": "design_space_edit",
                    "target": "quantization_dtype",
                    "change": {"value": "int8"},
                    "rationale": "int8 per the KPU efficiency study",
                    "addresses_issue": 0,
                    "research_refs": [
                        "accelerators/stillwater_kpu.md",
                        "efficiency_studies/quantization_accuracy.md",
                    ],
                }
            ],
        }
    )
    v = Critic(llm_available=True, llm_client=_StubClient(payload)).review(_power_state())
    assert v.deltas[0].research_refs == [
        "accelerators/stillwater_kpu.md",
        "efficiency_studies/quantization_accuracy.md",
    ]


def test_deltas_without_research_refs_default_empty() -> None:
    payload = json.dumps(
        {
            "issues": [{"metric": "power", "severity": "high", "summary": "p"}],
            "deltas": [
                {
                    "kind": "design_space_edit",
                    "target": "x",
                    "change": {"value": 1},
                    "rationale": "r",
                }
            ],
        }
    )
    v = Critic(llm_available=True, llm_client=_StubClient(payload)).review(_power_state())
    assert v.deltas[0].research_refs == []
