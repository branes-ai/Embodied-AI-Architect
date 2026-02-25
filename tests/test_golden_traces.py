"""Tests for golden trace storage and comparison."""

from __future__ import annotations

import json

import pytest

from embodied_ai_architect.graphs.evaluation import RunTrace
from embodied_ai_architect.graphs.golden_traces import (
    TraceComparison,
    compare_traces,
    load_golden_trace,
    save_golden_trace,
)


def _make_trace(**kwargs) -> RunTrace:
    defaults = dict(
        demo_name="test",
        task_graph={
            "nodes": {
                "t1": {"name": "analyze", "agent": "workload_analyzer", "dependencies": []},
                "t2": {"name": "explore", "agent": "hw_explorer", "dependencies": ["t1"]},
            },
        },
        ppa_metrics={
            "power_watts": 5.0,
            "latency_ms": 30.0,
            "cost_usd": 25.0,
            "verdicts": {"power": "PASS", "latency": "PASS"},
        },
        tool_calls=["workload_analyzer", "hw_explorer"],
        iteration_history=[],
    )
    defaults.update(kwargs)
    return RunTrace(**defaults)


class TestSaveLoadGoldenTrace:
    def test_save_and_load_roundtrip(self, tmp_path):
        trace = _make_trace()
        path = tmp_path / "golden" / "test_trace.json"
        saved_path = save_golden_trace(trace, path)
        assert saved_path.exists()
        loaded = load_golden_trace(saved_path)
        assert loaded.demo_name == trace.demo_name
        assert loaded.ppa_metrics == trace.ppa_metrics
        assert loaded.tool_calls == trace.tool_calls

    def test_save_creates_parent_dirs(self, tmp_path):
        trace = _make_trace()
        path = tmp_path / "deep" / "nested" / "trace.json"
        saved_path = save_golden_trace(trace, path)
        assert saved_path.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_golden_trace(tmp_path / "nonexistent.json")

    def test_saved_file_is_valid_json(self, tmp_path):
        trace = _make_trace()
        path = tmp_path / "trace.json"
        save_golden_trace(trace, path)
        data = json.loads(path.read_text())
        assert data["demo_name"] == "test"


class TestCompareTraces:
    def test_identical_traces(self):
        golden = _make_trace()
        current = _make_trace()
        result = compare_traces(current, golden)
        assert isinstance(result, TraceComparison)
        assert result.task_graph_match is True
        assert result.ppa_regression is False
        assert result.iteration_regression is False
        assert result.is_regression is False

    def test_task_graph_mismatch(self):
        golden = _make_trace()
        current = _make_trace(
            task_graph={
                "nodes": {
                    "t1": {"name": "analyze", "agent": "workload_analyzer", "dependencies": []},
                    "t2": {"name": "explore", "agent": "hw_explorer", "dependencies": ["t1"]},
                    "t3": {"name": "extra", "agent": "design_explorer", "dependencies": ["t2"]},
                },
            },
        )
        result = compare_traces(current, golden)
        assert result.task_graph_match is False

    def test_ppa_regression_detected(self):
        golden = _make_trace(
            ppa_metrics={
                "power_watts": 5.0,
                "latency_ms": 30.0,
                "verdicts": {"power": "PASS"},
            }
        )
        current = _make_trace(
            ppa_metrics={
                "power_watts": 6.5,
                "latency_ms": 30.0,  # >10% worse
                "verdicts": {"power": "PASS"},
            }
        )
        result = compare_traces(current, golden)
        assert result.ppa_regression is True
        assert result.is_regression is True

    def test_no_ppa_regression_within_threshold(self):
        golden = _make_trace(ppa_metrics={"power_watts": 5.0, "verdicts": {}})
        current = _make_trace(ppa_metrics={"power_watts": 5.4, "verdicts": {}})
        result = compare_traces(current, golden)
        assert result.ppa_regression is False

    def test_verdict_regression(self):
        golden = _make_trace(ppa_metrics={"verdicts": {"power": "PASS", "latency": "PASS"}})
        current = _make_trace(ppa_metrics={"verdicts": {"power": "PASS", "latency": "FAIL"}})
        result = compare_traces(current, golden)
        assert result.ppa_regression is True

    def test_iteration_regression(self):
        golden = _make_trace(iteration_history=[{"iter": 1}])
        current = _make_trace(iteration_history=[{"iter": 1}, {"iter": 2}, {"iter": 3}])
        result = compare_traces(current, golden)
        assert result.iteration_regression is True

    def test_no_iteration_regression(self):
        golden = _make_trace(iteration_history=[{"iter": 1}, {"iter": 2}])
        current = _make_trace(iteration_history=[{"iter": 1}])
        result = compare_traces(current, golden)
        assert result.iteration_regression is False

    def test_tool_call_diff(self):
        golden = _make_trace(tool_calls=["workload_analyzer", "hw_explorer"])
        current = _make_trace(tool_calls=["workload_analyzer", "design_explorer"])
        result = compare_traces(current, golden)
        assert "hw_explorer" in result.tool_call_diff.get("removed", [])
        assert "design_explorer" in result.tool_call_diff.get("added", [])

    def test_details_populated(self):
        golden = _make_trace()
        current = _make_trace()
        result = compare_traces(current, golden)
        assert len(result.details) > 0

    def test_is_regression_property(self):
        comp = TraceComparison(ppa_regression=True, iteration_regression=False)
        assert comp.is_regression is True

        comp2 = TraceComparison(ppa_regression=False, iteration_regression=True)
        assert comp2.is_regression is True

        comp3 = TraceComparison(ppa_regression=False, iteration_regression=False)
        assert comp3.is_regression is False
