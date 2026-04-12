"""Tests for the branes platform CLI command group."""

import json

import pytest
from click.testing import CliRunner

from embodied_ai_architect.cli.commands.platform import platform

pytestmark = pytest.mark.cli


def _make_runner():
    return CliRunner()


class TestPlatformList:
    def test_lists_platforms(self):
        runner = _make_runner()
        result = runner.invoke(platform, ["list"], obj={})
        assert result.exit_code == 0, result.output
        assert "Platform Registry" in result.output

    def test_with_category_filter(self):
        runner = _make_runner()
        result = runner.invoke(platform, ["list", "--category", "aerial"], obj={})
        assert result.exit_code == 0, result.output

    def test_json_output(self):
        runner = _make_runner()
        result = runner.invoke(platform, ["list", "--json"], obj={})
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        if data:
            assert "id" in data[0]
            assert "name" in data[0]


class TestPlatformSearch:
    def test_search_delivery_drone(self):
        runner = _make_runner()
        result = runner.invoke(platform, ["search", "delivery drone"], obj={})
        assert result.exit_code == 0, result.output

    def test_search_json(self):
        runner = _make_runner()
        result = runner.invoke(platform, ["search", "drone", "--json"], obj={})
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)


class TestPlatformShow:
    def test_show_known_platform(self):
        runner = _make_runner()
        result = runner.invoke(platform, ["show", "aerial.agricultural_sprayer"], obj={})
        assert result.exit_code == 0, result.output

    def test_show_nonexistent(self):
        runner = _make_runner()
        result = runner.invoke(platform, ["show", "nonexistent_xyz_123"], obj={})
        assert result.exit_code != 0


class TestPlatformCategories:
    def test_lists_categories(self):
        runner = _make_runner()
        result = runner.invoke(platform, ["categories"], obj={})
        assert result.exit_code == 0
        assert "Categories" in result.output
