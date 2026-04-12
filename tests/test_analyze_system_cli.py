"""Tests for the branes analyze-system CLI command group."""

import pytest
from click.testing import CliRunner

from embodied_ai_architect.cli.commands.analyze_group import analyze_lifecycle

pytestmark = pytest.mark.cli


def _make_runner():
    return CliRunner()


_STUB_SUBCOMMANDS = [
    "power",
    "latency",
    "thermal",
    "swap",
    "safety",
    "cost",
    "bandwidth",
    "scheduling",
]


class TestAnalyzeLifecycleStubs:
    @pytest.mark.parametrize("subcmd", _STUB_SUBCOMMANDS)
    def test_stub_exits_zero(self, subcmd):
        runner = _make_runner()
        result = runner.invoke(analyze_lifecycle, [subcmd, "dummy_mission"], obj={})
        assert result.exit_code == 0, result.output
        assert "Coming in a future release" in result.output


class TestAnalyzeLifecycleHelp:
    def test_no_subcommand_shows_help(self):
        runner = _make_runner()
        result = runner.invoke(analyze_lifecycle, [], obj={})
        assert result.exit_code == 0
        # Help should list subcommands
        for subcmd in _STUB_SUBCOMMANDS:
            assert subcmd in result.output
