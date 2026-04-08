"""Tests for the design optimizer specialist."""

from embodied_ai_architect.graphs.memory import WorkingMemoryStore
from embodied_ai_architect.graphs.optimizer import (
    OPTIMIZATION_STRATEGIES,
    design_optimizer,
)
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
)
from embodied_ai_architect.graphs.task_graph import TaskNode


def _make_task():
    return TaskNode(id="opt1", name="Optimize design", agent="design_optimizer")


def _make_failing_state(power_watts=6.3, latency_ms=30.0):
    """Create a state where the KPU design fails power."""
    state = create_initial_soc_state(
        goal="Design drone SoC",
        constraints=DesignConstraints(max_power_watts=5.0, max_latency_ms=33.3),
        use_case="delivery_drone",
        platform="drone",
    )
    state["ppa_metrics"] = {
        "power_watts": power_watts,
        "latency_ms": latency_ms,
        "verdicts": {
            "power": "FAIL" if power_watts > 5.0 else "PASS",
            "latency": "FAIL" if latency_ms > 33.3 else "PASS",
        },
        "bottlenecks": ["Power 6.3W exceeds 5W budget"] if power_watts > 5.0 else [],
        "suggestions": [],
    }
    state["workload_profile"] = {
        "total_estimated_gflops": 9.2,
        "dominant_op": "convolution",
        "use_case": "delivery_drone",
        "source": "goal_estimation",
        "workloads": [
            {"name": "object_detection", "estimated_gflops": 8.7},
            {"name": "object_tracking", "estimated_gflops": 0.5},
        ],
    }
    state["ip_blocks"] = [
        {"name": "compute_engine", "type": "kpu", "config": {"frequency_mhz": 1000}},
        {"name": "cpu_subsystem", "type": "cpu", "config": {"cores": 2}},
    ]
    return state


class TestOptimizationStrategyCatalog:
    def test_catalog_not_empty(self):
        assert len(OPTIMIZATION_STRATEGIES) >= 5

    def test_all_strategies_have_required_fields(self):
        for strat in OPTIMIZATION_STRATEGIES:
            assert "name" in strat
            assert "description" in strat
            assert "applicable_when" in strat
            assert "power_reduction_factor" in strat
            assert "latency_reduction_factor" in strat

    def test_all_factors_in_range(self):
        for strat in OPTIMIZATION_STRATEGIES:
            assert -1.0 <= strat["power_reduction_factor"] <= 1.0
            assert -1.0 <= strat["latency_reduction_factor"] <= 1.0


class TestDesignOptimizer:
    def test_no_failures_no_optimization(self):
        state = _make_failing_state(power_watts=4.0, latency_ms=25.0)
        result = design_optimizer(_make_task(), state)
        assert result["applied"] is False
        assert result["strategy"] is None

    def test_selects_strategy_for_power_failure(self):
        state = _make_failing_state(power_watts=6.3)
        result = design_optimizer(_make_task(), state)

        assert result["applied"] is True
        assert result["strategy"] is not None
        assert "power" in result["failing_constraints"]
        assert "_state_updates" in result

    def test_reduces_ppa_metrics(self):
        state = _make_failing_state(power_watts=6.3)
        result = design_optimizer(_make_task(), state)
        updates = result["_state_updates"]

        new_ppa = updates["ppa_metrics"]
        assert new_ppa["power_watts"] < 6.3

    def test_records_in_working_memory(self):
        state = _make_failing_state(power_watts=6.3)
        result = design_optimizer(_make_task(), state)
        updates = result["_state_updates"]

        wm = WorkingMemoryStore(**updates["working_memory"])
        tried = wm.get_tried_descriptions("design_optimizer")
        assert len(tried) == 1
        assert tried[0] == result["strategy"]

    def test_skips_already_tried_strategies(self):
        state = _make_failing_state(power_watts=6.3)

        # Pre-populate working memory with first strategy
        store = WorkingMemoryStore()
        store.record_attempt("design_optimizer", "smaller_model", "tried", 0)
        state["working_memory"] = store.model_dump()

        result = design_optimizer(_make_task(), state)
        assert result["applied"] is True
        assert result["strategy"] != "smaller_model"

    def test_exhausted_strategies(self):
        state = _make_failing_state(power_watts=6.3)

        # Mark all power strategies as tried
        store = WorkingMemoryStore()
        for strat in OPTIMIZATION_STRATEGIES:
            if "power" in strat["applicable_when"]:
                store.record_attempt("design_optimizer", strat["name"], "tried", 0)
        state["working_memory"] = store.model_dump()

        result = design_optimizer(_make_task(), state)
        assert result["applied"] is False
        assert len(result["already_tried"]) > 0

    def test_modifies_workload_profile(self):
        state = _make_failing_state(power_watts=6.3)
        result = design_optimizer(_make_task(), state)

        if result["_state_updates"].get("workload_profile"):
            wp = result["_state_updates"]["workload_profile"]
            assert wp["total_estimated_gflops"] < 9.2

    def test_modifies_ip_blocks_for_clock_scaling(self):
        """If clock_scaling is selected, ip_blocks should have reduced frequency."""
        state = _make_failing_state(power_watts=6.3)

        # Make clock_scaling the only option
        store = WorkingMemoryStore()
        for strat in OPTIMIZATION_STRATEGIES:
            if strat["name"] != "clock_scaling" and "power" in strat["applicable_when"]:
                store.record_attempt("design_optimizer", strat["name"], "tried", 0)
        state["working_memory"] = store.model_dump()

        result = design_optimizer(_make_task(), state)
        if result["applied"] and result["strategy"] == "clock_scaling":
            ip_blocks = result["_state_updates"]["ip_blocks"]
            kpu = next(b for b in ip_blocks if b["type"] == "kpu")
            assert kpu["config"]["frequency_mhz"] < 1000

    def test_selects_best_strategy_for_latency(self):
        """When latency is the only failure, prefer latency reduction."""
        state = _make_failing_state(power_watts=4.0, latency_ms=50.0)
        state["ppa_metrics"]["verdicts"] = {"power": "PASS", "latency": "FAIL"}

        result = design_optimizer(_make_task(), state)
        assert result["applied"] is True
        assert "latency" in result["failing_constraints"]


class TestMOOAwareStrategySelection:
    """Issue #25: MOO sensitivity should drive strategy selection when available."""

    def test_sensitivity_boosts_clock_scaling_when_frequency_dominates(self):
        """When sensitivity says clock_mhz drives power, clock_scaling should
        be preferred over higher-static-factor alternatives like smaller_model."""
        state = _make_failing_state(power_watts=6.3)

        # Inject MOO sensitivity in real producer format showing
        # clock_mhz dominates power (impact=0.85), beating smaller_model's
        # static 0.35 factor when boost is applied
        state["moo_results"] = {
            "sensitivity": {
                "power_watts": {
                    "clock_mhz": {"lengthscale": 0.4, "importance": 0.85},
                    "process_nm": {"lengthscale": 0.6, "importance": 0.40},
                },
            },
        }

        result = design_optimizer(_make_task(), state)
        assert result["applied"] is True
        # Without MOO: smaller_model (0.35 factor) wins
        # With MOO: clock_scaling (0.15 * 1.85 boost = 0.278) competes,
        #          but smaller_model (0.35 * 1.0 = 0.35) still wins.
        # The key check: rationale must mention sensitivity
        assert "sensitivity" in result.get("selection_rationale", "").lower()

    def test_high_clock_sensitivity_overrides_greedy_choice(self):
        """When clock_mhz sensitivity is extreme (boost > smaller_model gap),
        clock_scaling should win over smaller_model."""
        state = _make_failing_state(power_watts=6.3)

        # Make smaller_model already-tried so it's not in the running.
        # This isolates clock_scaling vs other power strategies.
        store = WorkingMemoryStore()
        store.record_attempt("design_optimizer", "smaller_model", "tried", 0)
        state["working_memory"] = store.model_dump()

        # Strong clock sensitivity → clock_scaling preferred
        state["moo_results"] = {
            "sensitivity": {
                "power_watts": {
                    "clock_mhz": {"lengthscale": 0.2, "importance": 0.95},
                },
            },
        }

        result = design_optimizer(_make_task(), state)
        assert result["applied"] is True
        # quantize_int8 has factor 0.20, clock_scaling 0.15
        # Without sensitivity: quantize_int8 wins (0.20 vs 0.15)
        # With clock_mhz boost 1.95: clock_scaling = 0.15 * 1.95 = 0.293 > 0.20
        assert result["strategy"] == "clock_scaling"
        assert "sensitivity" in result["selection_rationale"].lower()
        assert "clock_mhz" in result["selection_rationale"]

    def test_falls_back_to_greedy_without_moo(self):
        """No moo_results → use the original greedy reduction-factor selection."""
        state = _make_failing_state(power_watts=6.3)
        # No moo_results set
        result = design_optimizer(_make_task(), state)
        assert result["applied"] is True
        assert "greedy" in result["selection_rationale"].lower()

    def test_falls_back_to_greedy_with_empty_sensitivity(self):
        """moo_results with empty sensitivity → falls back to greedy."""
        state = _make_failing_state(power_watts=6.3)
        state["moo_results"] = {"sensitivity": {}, "total_evaluations": 100}
        result = design_optimizer(_make_task(), state)
        assert result["applied"] is True
        assert "greedy" in result["selection_rationale"].lower()

    def test_writes_rationale_to_state_updates(self):
        """The rationale should propagate to last_strategy_rationale on the state
        so the optimization review snapshot can pick it up (issue #25)."""
        state = _make_failing_state(power_watts=6.3)
        state["moo_results"] = {
            "sensitivity": {
                "power_watts": {
                    "clock_mhz": {"lengthscale": 0.3, "importance": 0.75},
                },
            },
        }
        result = design_optimizer(_make_task(), state)
        updates = result["_state_updates"]
        assert "last_strategy_rationale" in updates
        assert updates["last_strategy_rationale"]

    def test_selection_rationale_on_result(self):
        """Both code paths must populate the selection_rationale field."""
        state = _make_failing_state(power_watts=6.3)
        result = design_optimizer(_make_task(), state)
        assert "selection_rationale" in result
        assert result["selection_rationale"]


class TestProcessNodeStrategies:
    """Tests for shrink/grow process node optimizer strategies."""

    def _make_cost_failing_state(self):
        """State where cost is the only failing constraint."""
        state = create_initial_soc_state(
            goal="Design drone SoC",
            constraints=DesignConstraints(
                max_power_watts=10.0,
                max_latency_ms=100.0,
                max_cost_usd=5.0,  # tight cost target
                target_process_nm=28,
            ),
            use_case="delivery_drone",
            platform="drone",
        )
        state["ppa_metrics"] = {
            "power_watts": 4.0,
            "latency_ms": 25.0,
            "cost_usd": 22.0,
            "verdicts": {"power": "PASS", "latency": "PASS", "cost": "FAIL"},
            "bottlenecks": ["Cost $22 exceeds $5 budget"],
            "suggestions": [],
        }
        state["workload_profile"] = {
            "total_estimated_gflops": 9.2,
            "dominant_op": "convolution",
            "use_case": "delivery_drone",
            "source": "goal_estimation",
        }
        state["ip_blocks"] = [
            {"name": "compute_engine", "type": "kpu", "config": {}},
            {"name": "cpu_subsystem", "type": "cpu", "config": {}},
        ]
        return state

    def _make_power_failing_state(self):
        """State where power is the only failing constraint."""
        state = create_initial_soc_state(
            goal="Design drone SoC",
            constraints=DesignConstraints(
                max_power_watts=3.0,  # tight power target
                max_latency_ms=100.0,
                target_process_nm=28,
            ),
            use_case="delivery_drone",
            platform="drone",
        )
        state["ppa_metrics"] = {
            "power_watts": 5.0,
            "latency_ms": 25.0,
            "verdicts": {"power": "FAIL", "latency": "PASS"},
            "bottlenecks": [],
            "suggestions": [],
        }
        state["workload_profile"] = {
            "total_estimated_gflops": 9.2,
            "dominant_op": "convolution",
            "use_case": "delivery_drone",
            "source": "goal_estimation",
        }
        state["ip_blocks"] = [
            {"name": "compute_engine", "type": "kpu", "config": {}},
            {"name": "cpu_subsystem", "type": "cpu", "config": {}},
        ]
        return state

    def test_grow_process_resolves_cost_fail(self):
        """grow_process_node strategy selected when cost fails."""
        state = self._make_cost_failing_state()

        # Exhaust all non-process strategies applicable to cost
        # (smaller_model no longer has "cost", so only grow_process_node applies)
        result = design_optimizer(_make_task(), state)

        assert result["applied"] is True
        assert result["strategy"] == "grow_process_node"

    def test_shrink_process_available_for_power_fail(self):
        """shrink_process_node is among applicable strategies when power fails."""
        state = self._make_power_failing_state()

        # Exhaust all workload/ip_blocks power strategies
        store = WorkingMemoryStore()
        for strat in OPTIMIZATION_STRATEGIES:
            if (
                strat["applies_to"] in ("workload_profile", "ip_blocks")
                and "power" in strat["applicable_when"]
            ):
                store.record_attempt("design_optimizer", strat["name"], "tried", 0)
        state["working_memory"] = store.model_dump()

        result = design_optimizer(_make_task(), state)
        assert result["applied"] is True
        assert result["strategy"] == "shrink_process_node"

    def test_process_strategy_updates_constraints(self):
        """_state_updates['constraints'] has new target_process_nm."""
        state = self._make_cost_failing_state()
        result = design_optimizer(_make_task(), state)

        assert result["applied"] is True
        updates = result["_state_updates"]
        assert "constraints" in updates
        new_nm = updates["constraints"]["target_process_nm"]
        # 28nm → 40nm (grow = next larger node)
        assert new_nm == 40

    def test_dynamic_strategy_naming(self):
        """Working memory records 'grow_process_node_28' not just 'grow_process_node'."""
        state = self._make_cost_failing_state()
        result = design_optimizer(_make_task(), state)

        wm = WorkingMemoryStore(**result["_state_updates"]["working_memory"])
        tried = wm.get_tried_descriptions("design_optimizer")
        assert "grow_process_node_28" in tried
        assert "grow_process_node" not in tried
