"""Tests for KPU specialist agents (configurator, validators, optimizer)."""

from embodied_ai_architect.graphs.kpu_config import (
    KPU_PRESETS,
    apply_kpu_overrides,
    create_kpu_config,
)
from embodied_ai_architect.graphs.kpu_specialists import (
    bandwidth_validator,
    floorplan_validator,
    kpu_configurator,
    kpu_optimizer,
)
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
)
from embodied_ai_architect.graphs.task_graph import TaskNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides):
    """Create a default test state for a delivery drone design."""
    state = create_initial_soc_state(
        goal="Design drone SoC",
        constraints=DesignConstraints(max_power_watts=5.0, max_area_mm2=100.0),
        use_case="delivery_drone",
        rtl_enabled=True,
    )
    state.update(overrides)
    return state


def _make_task(agent: str = "kpu_configurator") -> TaskNode:
    """Create a minimal TaskNode for testing."""
    return TaskNode(id="t1", name="Test task", agent=agent)


def _configure_state(state):
    """Run kpu_configurator and apply its state updates, returning the enriched state."""
    task = _make_task("kpu_configurator")
    result = kpu_configurator(task, state)
    for key, value in result.get("_state_updates", {}).items():
        state[key] = value
    return state, result


# ---------------------------------------------------------------------------
# Tests: kpu_configurator
# ---------------------------------------------------------------------------


class TestKPUConfigurator:
    def test_kpu_configurator_generates_config(self):
        """kpu_configurator returns a kpu_config dict with expected keys."""
        state = _make_state()
        task = _make_task("kpu_configurator")

        result = kpu_configurator(task, state)

        assert "kpu_config" in result
        config = result["kpu_config"]
        assert "name" in config
        assert "process_nm" in config
        assert "dram" in config
        assert "noc" in config
        assert "compute_tile" in config
        assert "memory_tile" in config
        assert "array_rows" in config
        assert "array_cols" in config
        assert "summary" in result
        assert "delivery-drone" in config["name"]

    def test_kpu_configurator_state_updates(self):
        """_state_updates contains kpu_config matching the returned kpu_config."""
        state = _make_state()
        task = _make_task("kpu_configurator")

        result = kpu_configurator(task, state)

        assert "_state_updates" in result
        assert "kpu_config" in result["_state_updates"]
        assert result["_state_updates"]["kpu_config"] == result["kpu_config"]


# ---------------------------------------------------------------------------
# Issue #29: architect overrides from plan review
# ---------------------------------------------------------------------------


class TestApplyKPUOverrides:
    """The apply_kpu_overrides helper handles dotted-path merges."""

    def test_no_overrides_returns_input_unchanged(self):
        cfg = KPU_PRESETS["edge_balanced"]
        out = apply_kpu_overrides(cfg, {})
        assert out.compute_tile.array_rows == cfg.compute_tile.array_rows

    def test_top_level_field(self):
        cfg = KPU_PRESETS["edge_balanced"]
        out = apply_kpu_overrides(cfg, {"name": "swkpu-custom"})
        assert out.name == "swkpu-custom"

    def test_nested_field(self):
        cfg = KPU_PRESETS["edge_balanced"]
        out = apply_kpu_overrides(
            cfg,
            {
                "compute_tile.array_rows": 8,
                "compute_tile.array_cols": 8,
                "noc.link_width_bits": 512,
                "dram.num_controllers": 4,
            },
        )
        assert out.compute_tile.array_rows == 8
        assert out.compute_tile.array_cols == 8
        assert out.noc.link_width_bits == 512
        assert out.dram.num_controllers == 4

    def test_unknown_path_silently_ignored(self):
        """Stale overrides from a previous review must not crash dispatch."""
        cfg = KPU_PRESETS["edge_balanced"]
        out = apply_kpu_overrides(
            cfg,
            {
                "compute_tile.array_rows": 8,  # valid
                "nonexistent.field": 42,  # invalid
                "compute_tile.no_such_field": 99,  # invalid leaf
            },
        )
        # Valid override applied
        assert out.compute_tile.array_rows == 8
        # Invalid ones did not raise

    def test_does_not_mutate_input(self):
        cfg = create_kpu_config("delivery_drone", {"max_power_watts": 5.0}, {"gflops": 5.0})
        original_rows = cfg.compute_tile.array_rows
        _ = apply_kpu_overrides(cfg, {"compute_tile.array_rows": 99})
        assert cfg.compute_tile.array_rows == original_rows


class TestKPUConfiguratorRespectsOverrides:
    """kpu_configurator must overlay state['kpu_config_overrides'] on heuristic output."""

    def test_overrides_win_over_heuristic(self):
        state = _make_state()
        # The heuristic for delivery_drone @ 5W gives array_rows=8 typically.
        # Force a different value via overrides — architect's choice must win.
        state["kpu_config_overrides"] = {
            "compute_tile.array_rows": 32,
            "compute_tile.array_cols": 32,
            "noc.link_width_bits": 512,
        }
        task = _make_task("kpu_configurator")
        result = kpu_configurator(task, state)

        cfg = result["kpu_config"]
        assert cfg["compute_tile"]["array_rows"] == 32
        assert cfg["compute_tile"]["array_cols"] == 32
        assert cfg["noc"]["link_width_bits"] == 512

    def test_summary_mentions_overrides(self):
        state = _make_state()
        state["kpu_config_overrides"] = {"compute_tile.array_rows": 16}
        task = _make_task("kpu_configurator")
        result = kpu_configurator(task, state)
        assert "override" in result["summary"].lower()

    def test_no_overrides_works_as_before(self):
        """Backward compatibility: empty/missing overrides → heuristic-only path."""
        state = _make_state()
        # No kpu_config_overrides set
        task = _make_task("kpu_configurator")
        result = kpu_configurator(task, state)
        assert "kpu_config" in result
        # Summary should NOT mention overrides
        assert "override" not in result["summary"].lower()


# ---------------------------------------------------------------------------
# Tests: floorplan_validator
# ---------------------------------------------------------------------------


class TestFloorplanValidator:
    def test_floorplan_validator_pass(self):
        """Floorplan validator returns PASS for a pitch-matched preset config."""
        state = _make_state()
        # Use the drone_minimal preset which is known to be pitch-matched and area-feasible
        state["kpu_config"] = KPU_PRESETS["drone_minimal"].model_dump()
        task = _make_task("floorplan_validator")

        result = floorplan_validator(task, state)

        assert result["verdict"] == "PASS"
        assert "floorplan_estimate" in result
        fp = result["floorplan_estimate"]
        assert fp["feasible"] is True
        assert fp["pitch_matched"] is True
        assert fp["total_area_mm2"] <= 100.0
        assert "_state_updates" in result
        assert "floorplan_estimate" in result["_state_updates"]

    def test_floorplan_validator_skip_no_config(self):
        """Floorplan validator returns SKIP when no kpu_config is in state."""
        state = _make_state()
        # kpu_config is empty dict by default from create_initial_soc_state
        task = _make_task("floorplan_validator")

        result = floorplan_validator(task, state)

        assert result["verdict"] == "SKIP"
        assert "_state_updates" not in result


# ---------------------------------------------------------------------------
# Tests: bandwidth_validator
# ---------------------------------------------------------------------------


class TestBandwidthValidator:
    def test_bandwidth_validator_pass(self):
        """Bandwidth validator returns PASS for a standard drone config."""
        state = _make_state()
        state, _ = _configure_state(state)
        task = _make_task("bandwidth_validator")

        result = bandwidth_validator(task, state)

        assert result["verdict"] == "PASS"
        assert "bandwidth_match" in result
        bw = result["bandwidth_match"]
        assert bw["balanced"] is True
        assert len(bw["links"]) == 4  # dram->l3, l3->l2, l2->l1, l1->compute
        assert "_state_updates" in result
        assert "bandwidth_match" in result["_state_updates"]

    def test_bandwidth_validator_skip_no_config(self):
        """Bandwidth validator returns SKIP when no kpu_config is in state."""
        state = _make_state()
        task = _make_task("bandwidth_validator")

        result = bandwidth_validator(task, state)

        assert result["verdict"] == "SKIP"
        assert "_state_updates" not in result


# ---------------------------------------------------------------------------
# Tests: kpu_optimizer
# ---------------------------------------------------------------------------


class TestKPUOptimizer:
    def test_kpu_optimizer_no_changes(self):
        """When floorplan and bandwidth both pass, optimizer reports no adjustments."""
        state = _make_state()
        # Use drone_minimal preset -- known to pass both floorplan and bandwidth
        state["kpu_config"] = KPU_PRESETS["drone_minimal"].model_dump()

        # Run floorplan and bandwidth validators to populate state
        fp_result = floorplan_validator(_make_task("floorplan_validator"), state)
        for key, value in fp_result.get("_state_updates", {}).items():
            state[key] = value

        bw_result = bandwidth_validator(_make_task("bandwidth_validator"), state)
        for key, value in bw_result.get("_state_updates", {}).items():
            state[key] = value

        # Confirm both passed before testing optimizer
        assert fp_result["verdict"] == "PASS"
        assert bw_result["verdict"] == "PASS"

        task = _make_task("kpu_optimizer")
        result = kpu_optimizer(task, state)

        assert "No adjustments needed" in result["changes"]
        assert "kpu_config" in result

    def test_kpu_optimizer_fixes_bandwidth(self):
        """Optimizer adds a DRAM controller when bandwidth_match shows dram bottleneck."""
        state = _make_state()
        state, configurator_result = _configure_state(state)

        original_controllers = state["kpu_config"]["dram"]["num_controllers"]

        # Inject a failing bandwidth match with dram bottleneck
        state["bandwidth_match"] = {
            "balanced": False,
            "bottleneck_link": "dram_to_l3",
            "peak_utilization": 0.95,
            "links": [],
            "issues": ["DRAM bandwidth bottleneck"],
        }
        # Set floorplan to passing so optimizer only acts on bandwidth
        state["floorplan_estimate"] = {
            "feasible": True,
            "pitch_matched": True,
            "pitch_ratio_width": 1.0,
            "pitch_ratio_height": 1.0,
        }

        task = _make_task("kpu_optimizer")
        result = kpu_optimizer(task, state)

        # Should have added a DRAM controller
        new_controllers = result["kpu_config"]["dram"]["num_controllers"]
        assert new_controllers == original_controllers + 1

        # Verify the change is described
        changes_text = " ".join(result["changes"])
        assert "DRAM controller" in changes_text
        assert "_state_updates" in result
        assert result["_state_updates"]["kpu_config"]["dram"]["num_controllers"] == new_controllers


# ---------------------------------------------------------------------------
# Issue #34: KPU convergence history persistence
# ---------------------------------------------------------------------------


class TestKPUConvergenceHistory:
    """Each KPU specialist must append a per-iteration entry to
    state['kpu_optimization_history'] so the architect can replay the
    inner-loop sequence."""

    def test_configurator_writes_history_entry(self) -> None:
        state = _make_state()
        result = kpu_configurator(_make_task("kpu_configurator"), state)
        history = result["_state_updates"]["kpu_optimization_history"]
        assert len(history) == 1
        entry = history[0]
        assert entry["source"] == "kpu_configurator"
        assert entry["config_name"]
        assert entry["compute_array"]  # systolic dims
        assert "summary" in entry

    def test_floorplan_validator_writes_history_entry(self) -> None:
        state = _make_state()
        state["kpu_config"] = KPU_PRESETS["drone_minimal"].model_dump()
        result = floorplan_validator(_make_task("floorplan_validator"), state)
        history = result["_state_updates"]["kpu_optimization_history"]
        assert len(history) == 1
        entry = history[0]
        assert entry["source"] == "floorplan_validator"
        assert "pitch_matched" in entry
        assert "total_area_mm2" in entry
        assert "floorplan_feasible" in entry

    def test_bandwidth_validator_writes_history_entry(self) -> None:
        state = _make_state()
        state["kpu_config"] = KPU_PRESETS["drone_minimal"].model_dump()
        result = bandwidth_validator(_make_task("bandwidth_validator"), state)
        history = result["_state_updates"]["kpu_optimization_history"]
        assert len(history) == 1
        entry = history[0]
        assert entry["source"] == "bandwidth_validator"
        assert "bandwidth_balanced" in entry
        assert "peak_utilization" in entry

    def test_kpu_optimizer_writes_history_entry_with_changes(self) -> None:
        state = _make_state()
        state["kpu_config"] = KPU_PRESETS["drone_minimal"].model_dump()
        # Force a DRAM bottleneck so the optimizer makes a change
        state["bandwidth_match"] = {
            "balanced": False,
            "bottleneck_link": "dram_to_l3",
            "peak_utilization": 0.95,
            "links": [],
            "issues": ["DRAM bandwidth bottleneck"],
        }
        state["floorplan_estimate"] = {
            "feasible": True,
            "pitch_matched": True,
            "pitch_ratio_width": 1.0,
            "pitch_ratio_height": 1.0,
        }
        result = kpu_optimizer(_make_task("kpu_optimizer"), state)
        history = result["_state_updates"]["kpu_optimization_history"]
        assert len(history) == 1
        entry = history[0]
        assert entry["source"] == "kpu_optimizer"
        assert "changes" in entry
        assert len(entry["changes"]) > 0
        # The change list mentions the DRAM controller bump
        assert any("DRAM controller" in c for c in entry["changes"])

    def test_history_accumulates_across_iterations(self) -> None:
        """Three sequential specialist calls should grow the history to 3."""
        state = _make_state()
        # Iteration 0: configurator
        result0 = kpu_configurator(_make_task("kpu_configurator"), state)
        for k, v in result0["_state_updates"].items():
            state[k] = v
        # Iteration 1: floorplan validator
        result1 = floorplan_validator(_make_task("floorplan_validator"), state)
        for k, v in result1["_state_updates"].items():
            state[k] = v
        # Iteration 2: bandwidth validator
        result2 = bandwidth_validator(_make_task("bandwidth_validator"), state)
        for k, v in result2["_state_updates"].items():
            state[k] = v

        history = state["kpu_optimization_history"]
        assert len(history) == 3
        assert [e["source"] for e in history] == [
            "kpu_configurator",
            "floorplan_validator",
            "bandwidth_validator",
        ]

    def test_history_records_outer_iteration_number(self) -> None:
        """The history entry should carry state['iteration'] so the architect
        can correlate KPU steps with the outer dispatch loop."""
        state = _make_state()
        state["iteration"] = 5
        result = kpu_configurator(_make_task("kpu_configurator"), state)
        entry = result["_state_updates"]["kpu_optimization_history"][0]
        assert entry["iteration"] == 5
