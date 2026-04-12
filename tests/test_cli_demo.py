"""Tests for the branes demo CLI command."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from click.testing import CliRunner

from embodied_ai_architect.cli import cli
from embodied_ai_architect.cli.commands import demo as demo_mod


pytestmark = pytest.mark.cli

# Register demo command on the cli group (normally done in main())
cli.add_command(demo_mod.demo)


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# branes demo list
# ---------------------------------------------------------------------------


def test_demo_list(runner):
    result = runner.invoke(cli, ["--quiet", "demo", "list"])
    assert result.exit_code == 0
    assert "soc-designer" in result.output
    assert "dse-pareto" in result.output
    assert "kpu-rtl" in result.output
    assert "full-campaign" in result.output


def test_demo_list_json(runner):
    result = runner.invoke(cli, ["--json", "demo", "list"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 7
    names = {d["name"] for d in data}
    assert "soc-designer" in names
    assert "dse-pareto" in names


# ---------------------------------------------------------------------------
# branes demo info
# ---------------------------------------------------------------------------


def test_demo_info(runner):
    result = runner.invoke(cli, ["--quiet", "demo", "info", "kpu-rtl"])
    assert result.exit_code == 0
    assert "KPU" in result.output
    assert "demo_kpu_rtl.py" in result.output


def test_demo_info_json(runner):
    result = runner.invoke(cli, ["--json", "demo", "info", "kpu-rtl"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["name"] == "kpu-rtl"
    assert data["number"] == 4


def test_demo_info_unknown(runner):
    result = runner.invoke(cli, ["--quiet", "demo", "info", "nonexistent"])
    assert result.exit_code != 0
    assert "Unknown demo" in result.output


# ---------------------------------------------------------------------------
# branes demo run (with mocked execution)
# ---------------------------------------------------------------------------


def _make_mock_module(run_demo_fn=None):
    """Create a mock module with a run_demo function."""
    mod = MagicMock()
    if run_demo_fn is None:
        mod.run_demo = MagicMock(return_value={"status": "complete"})
    else:
        mod.run_demo = run_demo_fn
    return mod


@patch("embodied_ai_architect.cli.commands.demo._load_demo_module")
def test_demo_run_specific(mock_load, runner):
    mock_load.return_value = _make_mock_module()
    result = runner.invoke(cli, ["--quiet", "demo", "run", "dse-pareto"])
    assert result.exit_code == 0
    assert "dse-pareto" in result.output
    mock_load.return_value.run_demo.assert_called_once_with()


@patch("embodied_ai_architect.cli.commands.demo._load_demo_module")
def test_demo_run_unknown(mock_load, runner):
    result = runner.invoke(cli, ["--quiet", "demo", "run", "nonexistent"])
    assert result.exit_code != 0
    assert "Unknown demo" in result.output


@patch("embodied_ai_architect.cli.commands.demo._load_demo_module")
def test_demo_run_json_mode(mock_load, runner):
    mock_load.return_value = _make_mock_module()
    result = runner.invoke(cli, ["--json", "demo", "run", "dse-pareto"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["demo"] == "dse-pareto"
    assert data["success"] is True


@patch("embodied_ai_architect.cli.commands.demo._load_demo_module")
def test_demo_run_quiet_mode(mock_load, runner):
    mock_load.return_value = _make_mock_module()
    result = runner.invoke(cli, ["--quiet", "demo", "run", "dse-pareto"])
    assert result.exit_code == 0
    assert "OK" in result.output


@patch("embodied_ai_architect.cli.commands.demo._load_demo_module")
def test_demo_run_with_args(mock_load, runner):
    mock_mod = _make_mock_module()
    mock_load.return_value = mock_mod
    result = runner.invoke(
        cli,
        ["--quiet", "demo", "run", "soc-optimizer", "--power", "4.0"],
    )
    assert result.exit_code == 0
    mock_mod.run_demo.assert_called_once_with(
        max_power=4.0, max_latency=33.3, max_cost=30.0, max_iterations=10
    )


@patch("embodied_ai_architect.cli.commands.demo._load_demo_module")
def test_demo_run_error_handling(mock_load, runner):
    def failing_demo():
        raise RuntimeError("something broke")

    mock_mod = _make_mock_module()
    mock_mod.run_demo = failing_demo
    mock_load.return_value = mock_mod

    result = runner.invoke(cli, ["--json", "demo", "run", "dse-pareto"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["success"] is False
    assert "RuntimeError" in data["error"]


@patch("embodied_ai_architect.cli.commands.demo._build_kwargs", return_value={})
@patch("embodied_ai_architect.cli.commands.demo._load_demo_module")
def test_demo_run_all(mock_load, mock_kwargs, runner):
    mock_load.return_value = _make_mock_module()
    result = runner.invoke(cli, ["--quiet", "demo", "run", "all"])
    assert result.exit_code == 0
    # Should see all 7 demos listed
    assert "soc-designer" in result.output or "OK" in result.output


@patch("embodied_ai_architect.cli.commands.demo._build_kwargs", return_value={})
@patch("embodied_ai_architect.cli.commands.demo._load_demo_module")
def test_demo_run_all_json(mock_load, mock_kwargs, runner):
    mock_load.return_value = _make_mock_module()
    result = runner.invoke(cli, ["--json", "demo", "run", "all"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["passed"] == 7
    assert data["failed"] == 0
    assert len(data["demos"]) == 7


# ---------------------------------------------------------------------------
# _get_examples_dir
# ---------------------------------------------------------------------------


def test_get_examples_dir():
    from embodied_ai_architect.cli.commands.demo import _get_examples_dir

    # Should resolve from repo root via file path
    examples_dir = _get_examples_dir()
    assert examples_dir.is_dir()
    assert (examples_dir / "demo_dse_pareto.py").exists()
