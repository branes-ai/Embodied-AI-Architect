"""Tests for the branes spec CLI command group."""

import pytest
from click.testing import CliRunner

from embodied_ai_architect.cli.commands.spec import spec
import embodied_ai_architect.mission.store as store_mod

pytestmark = pytest.mark.cli


@pytest.fixture(autouse=True)
def _use_tmp_dirs(tmp_path, monkeypatch):
    """Isolate both SpecStore (cwd-based) and MissionStore to tmp."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(store_mod, "DEFAULT_MISSIONS_DIR", tmp_path / "missions")


def _make_runner():
    return CliRunner()


class TestSpecNew:
    def test_creates_spec(self):
        runner = _make_runner()
        result = runner.invoke(spec, ["new", "test-spec"], obj={})
        assert result.exit_code == 0, result.output
        assert "Created spec" in result.output
        assert "test-spec" in result.output

    def test_creates_spec_with_description(self):
        runner = _make_runner()
        result = runner.invoke(spec, ["new", "my-drone", "-d", "A test drone spec"], obj={})
        assert result.exit_code == 0
        assert "my-drone" in result.output


class TestSpecList:
    def test_empty_list(self):
        runner = _make_runner()
        result = runner.invoke(spec, ["list"], obj={})
        assert result.exit_code == 0
        assert "No specs found" in result.output

    def test_list_after_create(self):
        runner = _make_runner()
        runner.invoke(spec, ["new", "alpha"], obj={})
        result = runner.invoke(spec, ["list"], obj={})
        assert result.exit_code == 0
        assert "alpha" in result.output


class TestSpecShow:
    def test_show_existing(self):
        runner = _make_runner()
        runner.invoke(spec, ["new", "test-spec"], obj={})
        result = runner.invoke(spec, ["show", "test-spec"], obj={})
        assert result.exit_code == 0
        assert "test-spec" in result.output

    def test_show_nonexistent(self):
        runner = _make_runner()
        result = runner.invoke(spec, ["show", "nonexistent"], obj={})
        assert result.exit_code != 0


class TestSpecSet:
    def test_set_field(self):
        runner = _make_runner()
        runner.invoke(spec, ["new", "test-spec"], obj={})
        result = runner.invoke(spec, ["set", "test-spec", "/perception/min_fps", "60"], obj={})
        assert result.exit_code == 0, result.output
        assert "Set" in result.output
        assert "60" in result.output


class TestSpecCommit:
    def test_commit(self):
        runner = _make_runner()
        runner.invoke(spec, ["new", "test-spec"], obj={})
        result = runner.invoke(spec, ["commit", "test-spec", "-m", "initial"], obj={})
        assert result.exit_code == 0, result.output
        assert "Committed" in result.output


class TestSpecHistory:
    def test_no_versions(self):
        runner = _make_runner()
        runner.invoke(spec, ["new", "test-spec"], obj={})
        result = runner.invoke(spec, ["history", "test-spec"], obj={})
        assert result.exit_code == 0
        assert "No committed versions" in result.output

    def test_with_versions(self):
        runner = _make_runner()
        runner.invoke(spec, ["new", "test-spec"], obj={})
        runner.invoke(spec, ["commit", "test-spec", "-m", "v1"], obj={})
        result = runner.invoke(spec, ["history", "test-spec"], obj={})
        assert result.exit_code == 0
        assert "v1" in result.output


class TestSpecDelete:
    def test_delete_field(self):
        runner = _make_runner()
        runner.invoke(spec, ["new", "test-spec"], obj={})
        runner.invoke(spec, ["set", "test-spec", "/perception/min_fps", "60"], obj={})
        result = runner.invoke(spec, ["delete", "test-spec", "/perception/min_fps"], obj={})
        assert result.exit_code == 0, result.output
        assert "Deleted" in result.output
