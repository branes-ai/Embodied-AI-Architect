"""Tests for branes swap CLI subcommands."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from embodied_ai_architect.cli import cli
from embodied_ai_architect.cli.commands import swap as swap_mod

# Register swap command on the cli group (normally done in main())
cli.add_command(swap_mod.swap)


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------


def test_estimate_basic(runner):
    result = runner.invoke(cli, ["--quiet", "swap", "estimate", "--area", "50", "--power", "5"])
    assert result.exit_code == 0
    assert "Weight" in result.output


def test_estimate_all_options(runner):
    result = runner.invoke(
        cli,
        [
            "--quiet",
            "swap",
            "estimate",
            "--area",
            "50",
            "--power",
            "5",
            "--process",
            "7",
            "--package",
            "FCBGA",
            "--cooling",
            "active_fan",
            "--enclosure",
            "magnesium",
            "--volume",
            "1000",
            "--layers",
            "6",
            "--connectors",
            "2",
            "--ambient-temp",
            "35",
        ],
    )
    assert result.exit_code == 0
    assert "Weight" in result.output


def test_estimate_json(runner):
    result = runner.invoke(
        cli,
        [
            "--quiet",
            "swap",
            "estimate",
            "--area",
            "50",
            "--power",
            "5",
            "--json-output",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "weight_grams" in data
    assert "volume_cm3" in data
    assert "cost_usd" in data
    assert "thermal" in data


# ---------------------------------------------------------------------------
# bom
# ---------------------------------------------------------------------------


def test_bom_table(runner):
    result = runner.invoke(cli, ["--quiet", "swap", "bom", "--area", "50", "--power", "5"])
    assert result.exit_code == 0
    assert "die" in result.output
    assert "pcb" in result.output
    assert "TOTAL" in result.output


def test_bom_json(runner):
    result = runner.invoke(
        cli,
        [
            "--quiet",
            "swap",
            "bom",
            "--area",
            "50",
            "--power",
            "5",
            "--json-output",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    names = [row["name"] for row in data]
    assert "die" in names
    assert "pcb" in names


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def test_check_pass(runner):
    result = runner.invoke(
        cli,
        [
            "--quiet",
            "swap",
            "check",
            "--area",
            "50",
            "--power",
            "5",
            "--max-weight",
            "500",
            "--max-volume",
            "200",
            "--max-cost",
            "1000",
        ],
    )
    assert result.exit_code == 0
    assert "PASS" in result.output


def test_check_fail(runner):
    result = runner.invoke(
        cli,
        [
            "--quiet",
            "swap",
            "check",
            "--area",
            "50",
            "--power",
            "5",
            "--max-weight",
            "1",  # impossibly tight budget
        ],
    )
    assert result.exit_code == 1


def test_check_json(runner):
    result = runner.invoke(
        cli,
        [
            "--quiet",
            "swap",
            "check",
            "--area",
            "50",
            "--power",
            "5",
            "--max-weight",
            "500",
            "--json-output",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "metrics" in data
    assert "overall_verdict" in data


@patch("embodied_ai_architect.cli.commands.swap._read_spec_budgets")
def test_check_from_spec(mock_read, runner):
    mock_read.return_value = {
        "max_power_watts": 10.0,
        "max_weight_grams": 500.0,
    }
    result = runner.invoke(
        cli,
        [
            "--quiet",
            "swap",
            "check",
            "--area",
            "50",
            "--power",
            "5",
            "--from-spec",
            "my-drone",
        ],
    )
    assert result.exit_code == 0
    mock_read.assert_called_once_with("my-drone")
    assert "PASS" in result.output


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_two_configs(runner):
    result = runner.invoke(
        cli,
        [
            "--quiet",
            "swap",
            "compare",
            "--area",
            "50",
            "--power",
            "5",
            "--process",
            "28",
            "--left",
            "QFN,passive,abs_plastic",
            "--right",
            "FCBGA,active_fan,aluminum",
        ],
    )
    assert result.exit_code == 0
    assert "QFN" in result.output
    assert "FCBGA" in result.output
    assert "Weight" in result.output


def test_compare_json(runner):
    result = runner.invoke(
        cli,
        [
            "--quiet",
            "swap",
            "compare",
            "--area",
            "50",
            "--power",
            "5",
            "--left",
            "BGA,passive,aluminum",
            "--right",
            "FCBGA,liquid,magnesium",
            "--json-output",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "left" in data
    assert "right" in data
    assert "weight_grams" in data["left"]


def test_compare_bad_config(runner):
    result = runner.invoke(
        cli,
        [
            "--quiet",
            "swap",
            "compare",
            "--area",
            "50",
            "--power",
            "5",
            "--left",
            "QFN,passive",  # missing enclosure
            "--right",
            "FCBGA,active_fan,aluminum",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# explore (fast mode, no real optimization engine needed — but tests integration)
# ---------------------------------------------------------------------------


def test_explore_fast(runner):
    """Runs fast-mode explore end-to-end. This exercises the full optimization pipeline."""
    result = runner.invoke(
        cli,
        [
            "--quiet",
            "swap",
            "explore",
            "-g",
            "test SoC",
            "--power",
            "10",
            "--weight",
            "500",
            "--fast",
            "--workers",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "Optimization Complete" in result.output or "pareto_front" in result.output


# ---------------------------------------------------------------------------
# show-front
# ---------------------------------------------------------------------------


def test_show_front_no_result(runner, monkeypatch):
    """Error message when no prior explore result exists."""
    monkeypatch.setattr("embodied_ai_architect.cli.commands.swap._load_swap_result", lambda: None)
    result = runner.invoke(cli, ["--quiet", "swap", "show-front"])
    assert result.exit_code == 0
    assert "No SWaP-C results" in result.output


def test_show_front_with_result(runner, tmp_path, monkeypatch):
    """Shows table when a result file exists."""
    import os
    import tempfile

    # Write a fake result
    fake_result = {
        "pareto_front": [
            {
                "objectives": {
                    "power_watts": 3.5,
                    "latency_ms": 10.0,
                    "area_mm2": 25.0,
                    "cost_usd": 12.0,
                    "weight_grams": 80.0,
                    "volume_cm3": 40.0,
                },
                "design_params": {
                    "process_nm": 28,
                    "package_type": "BGA",
                    "cooling_type": "passive",
                },
            },
        ],
        "knee_point": None,
    }
    result_path = os.path.join(tempfile.gettempdir(), "branes_swap_result.json")
    with open(result_path, "w") as f:
        json.dump(fake_result, f)

    try:
        result = runner.invoke(cli, ["--quiet", "swap", "show-front"])
        assert result.exit_code == 0
        assert "BGA" in result.output or "passive" in result.output
    finally:
        os.unlink(result_path)


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


def test_explain_basic(runner):
    """Shows delta table for two points from a saved result."""
    import os
    import tempfile

    fake_result = {
        "pareto_front": [
            {
                "objectives": {
                    "power_watts": 3.5,
                    "latency_ms": 10.0,
                    "area_mm2": 25.0,
                    "cost_usd": 12.0,
                    "weight_grams": 80.0,
                    "volume_cm3": 40.0,
                },
                "design_params": {
                    "process_nm": 28,
                    "package_type": "BGA",
                    "cooling_type": "passive",
                },
            },
            {
                "objectives": {
                    "power_watts": 5.0,
                    "latency_ms": 8.0,
                    "area_mm2": 35.0,
                    "cost_usd": 18.0,
                    "weight_grams": 120.0,
                    "volume_cm3": 60.0,
                },
                "design_params": {
                    "process_nm": 14,
                    "package_type": "FCBGA",
                    "cooling_type": "active_fan",
                },
            },
        ],
        "knee_point": None,
    }
    result_path = os.path.join(tempfile.gettempdir(), "branes_swap_result.json")
    with open(result_path, "w") as f:
        json.dump(fake_result, f)

    try:
        result = runner.invoke(cli, ["--quiet", "swap", "explain", "--points", "0,1"])
        assert result.exit_code == 0
        assert "Objective Changes" in result.output
        assert "weight_grams" in result.output
        assert "volume_cm3" in result.output
    finally:
        os.unlink(result_path)


def test_explain_no_result(runner, monkeypatch):
    monkeypatch.setattr("embodied_ai_architect.cli.commands.swap._load_swap_result", lambda: None)
    result = runner.invoke(cli, ["--quiet", "swap", "explain", "--points", "0,1"])
    assert result.exit_code == 0
    assert "No SWaP-C results" in result.output


def test_explain_invalid_format(runner):
    """Bad --points format should give an error."""
    import os
    import tempfile

    fake_result = {"pareto_front": [{"objectives": {}, "design_params": {}}]}
    result_path = os.path.join(tempfile.gettempdir(), "branes_swap_result.json")
    with open(result_path, "w") as f:
        json.dump(fake_result, f)

    try:
        result = runner.invoke(cli, ["--quiet", "swap", "explain", "--points", "abc"])
        assert result.exit_code == 0
        assert "Invalid points format" in result.output
    finally:
        os.unlink(result_path)
