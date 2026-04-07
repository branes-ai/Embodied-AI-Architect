"""Tests for optimization transparency and human steering."""

from embodied_ai_architect.graphs.optimization_review import (
    OptimizationReviewSnapshot,
    OptimizationSteeringInput,
    SteeringDecision,
    analyze_strategies,
    apply_steering_input,
    build_optimization_review_snapshot,
    compute_constraint_slackness,
    render_optimization_review,
    summarize_trajectory,
)
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_state(
    power=4.0,
    latency=30.0,
    area=None,
    iteration=0,
    verdicts=None,
    trajectory=None,
):
    """Build a state dict with PPA metrics."""
    state = create_initial_soc_state(
        goal="Design a drone perception SoC",
        constraints=DesignConstraints(
            max_power_watts=5.0,
            max_latency_ms=33.3,
            max_cost_usd=30.0,
        ),
        use_case="delivery_drone",
        platform="drone",
    )

    ppa = {
        "power_watts": power,
        "latency_ms": latency,
        "verdicts": verdicts or {"power": "PASS", "latency": "PASS", "cost": "PASS"},
    }
    if area is not None:
        ppa["area_mm2"] = area

    state["ppa_metrics"] = ppa
    state["iteration"] = iteration
    state["optimization_history"] = trajectory or []
    return state


def _make_trajectory(iterations=3):
    """Build a sample optimization trajectory."""
    trajectory = []
    for i in range(iterations):
        trajectory.append(
            {
                "iteration": i,
                "ppa_snapshot": {
                    "power_watts": 6.0 - i * 0.5,
                    "latency_ms": 40.0 - i * 3.0,
                    "cost_usd": 35.0 - i * 2.0,
                },
                "verdicts": {
                    "power": "FAIL" if 6.0 - i * 0.5 > 5.0 else "PASS",
                    "latency": "FAIL" if 40.0 - i * 3.0 > 33.3 else "PASS",
                    "cost": "FAIL" if 35.0 - i * 2.0 > 30.0 else "PASS",
                },
            }
        )
    return trajectory


# ---------------------------------------------------------------------------
# Constraint slackness
# ---------------------------------------------------------------------------


class TestComputeConstraintSlackness:
    def test_all_pass(self):
        state = _make_state(power=4.0, latency=30.0)
        slackness = compute_constraint_slackness(state)
        assert len(slackness) >= 2
        power = next(cs for cs in slackness if cs.name == "power")
        assert power.margin_pct > 0  # 4.0 vs 5.0 = 20% slack
        assert power.verdict == "PASS"

    def test_failing_constraint(self):
        state = _make_state(
            power=7.0,
            latency=30.0,
            verdicts={"power": "FAIL", "latency": "PASS", "cost": "PASS"},
        )
        slackness = compute_constraint_slackness(state)
        power = next(cs for cs in slackness if cs.name == "power")
        assert power.margin_pct < 0  # 7.0 vs 5.0 = -40% violated
        assert power.verdict == "FAIL"

    def test_binding_constraint(self):
        state = _make_state(power=4.8, latency=30.0)
        slackness = compute_constraint_slackness(state)
        power = next(cs for cs in slackness if cs.name == "power")
        # 4.8 vs 5.0 = 4% margin → binding
        assert power.binding is True

    def test_trend_detection(self):
        trajectory = _make_trajectory(3)
        state = _make_state(
            power=5.0,
            latency=34.0,
            verdicts={"power": "PASS", "latency": "FAIL", "cost": "FAIL"},
            trajectory=trajectory,
        )
        slackness = compute_constraint_slackness(state)
        # Power was decreasing (6.0 → 5.5 → 5.0) → improving
        power = next(cs for cs in slackness if cs.name == "power")
        assert power.trend == "improving"


# ---------------------------------------------------------------------------
# Strategy analysis
# ---------------------------------------------------------------------------


class TestAnalyzeStrategies:
    def test_with_failing_constraints(self):
        state = _make_state(
            power=7.0,
            latency=40.0,
            verdicts={"power": "FAIL", "latency": "FAIL"},
        )
        strategies = analyze_strategies(state)
        assert len(strategies) > 0
        available = [s for s in strategies if s.status == "available"]
        assert len(available) > 0

    def test_all_pass_no_applicable(self):
        state = _make_state(
            power=3.0,
            latency=20.0,
            verdicts={"power": "PASS", "latency": "PASS"},
        )
        strategies = analyze_strategies(state)
        available = [s for s in strategies if s.status == "available"]
        assert len(available) == 0

    def test_tried_strategies_excluded(self):
        state = _make_state(
            power=7.0,
            verdicts={"power": "FAIL"},
        )
        # Mark quantize_int8 as tried
        from embodied_ai_architect.graphs.memory import WorkingMemoryStore

        store = WorkingMemoryStore()
        store.record_attempt("design_optimizer", "quantize_int8", "applied", 0)
        state["working_memory"] = store.model_dump()

        strategies = analyze_strategies(state)
        quant = next(s for s in strategies if s.name == "quantize_int8")
        assert quant.status == "tried"


# ---------------------------------------------------------------------------
# Trajectory summary
# ---------------------------------------------------------------------------


class TestSummarizeTrajectory:
    def test_empty(self):
        result = summarize_trajectory([])
        assert "no optimization" in result.lower()

    def test_with_data(self):
        trajectory = _make_trajectory(3)
        result = summarize_trajectory(trajectory)
        assert "iter 0" in result
        assert "iter 2" in result
        assert "P=" in result  # Power metric


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------


class TestBuildOptimizationReviewSnapshot:
    def test_builds_snapshot(self):
        state = _make_state(
            power=6.0,
            latency=35.0,
            verdicts={"power": "FAIL", "latency": "FAIL"},
            trajectory=_make_trajectory(2),
            iteration=2,
        )
        snap = build_optimization_review_snapshot(state)
        assert snap.iteration == 2
        assert not snap.all_pass
        assert snap.most_violated is not None
        assert len(snap.strategies) > 0
        assert len(snap.constraint_slackness) >= 2

    def test_snapshot_serializable(self):
        state = _make_state(
            power=4.0,
            latency=30.0,
            trajectory=_make_trajectory(1),
        )
        snap = build_optimization_review_snapshot(state)
        data = snap.model_dump()
        import json

        json.dumps(data, default=str)
        # Round-trip
        snap2 = OptimizationReviewSnapshot(**data)
        assert snap2.iteration == snap.iteration

    def test_sensitivity_extracted_from_moo_results(self):
        """Issue #24: sensitivity must flow from moo_results into the snapshot."""
        state = _make_state(power=4.0, latency=30.0)
        # Inject a sensitivity dict like the BO layer would produce
        state["moo_results"] = {
            "sensitivity": {
                "quantization_dtype": {
                    "power_watts": 0.82,
                    "latency_ms": 0.65,
                    "cost_usd": 0.12,
                },
                "npu_frequency_mhz": {
                    "power_watts": 0.71,
                    "latency_ms": 0.58,
                    "cost_usd": 0.45,
                },
                "sram_size_kb": {
                    "power_watts": 0.15,
                    "latency_ms": 0.72,
                    "cost_usd": 0.38,
                },
            },
            "total_evaluations": 100,
            "pareto_front": [],
        }
        snap = build_optimization_review_snapshot(state)
        assert snap.sensitivity
        assert "quantization_dtype" in snap.sensitivity
        assert snap.sensitivity["quantization_dtype"]["power_watts"] == 0.82

    def test_sensitivity_empty_when_no_moo(self):
        """When MOO hasn't run, sensitivity should be an empty dict, not crash."""
        state = _make_state()
        snap = build_optimization_review_snapshot(state)
        assert snap.sensitivity == {}


# ---------------------------------------------------------------------------
# Steering application
# ---------------------------------------------------------------------------


class TestApplySteeringInput:
    def test_accept(self):
        state = _make_state()
        steering = OptimizationSteeringInput(decision=SteeringDecision.ACCEPT)
        updates = apply_steering_input(state, steering)
        assert updates["next_action"] == "report"

    def test_stop(self):
        state = _make_state()
        steering = OptimizationSteeringInput(decision=SteeringDecision.STOP)
        updates = apply_steering_input(state, steering)
        assert updates["next_action"] == "report"

    def test_redirect(self):
        state = _make_state()
        steering = OptimizationSteeringInput(
            decision=SteeringDecision.REDIRECT,
            focus_objective="power",
        )
        updates = apply_steering_input(state, steering)
        assert updates["next_action"] == "optimize"
        # Should have stored steering directives
        assert "optimization_steering" in updates

    def test_constraint_relaxation(self):
        state = _make_state()
        steering = OptimizationSteeringInput(
            decision=SteeringDecision.CONTINUE,
            constraint_relaxation={"max_power_watts": 8.0},
        )
        updates = apply_steering_input(state, steering)
        assert updates["constraints"]["max_power_watts"] == 8.0

    def test_notes_recorded(self):
        state = _make_state()
        steering = OptimizationSteeringInput(
            decision=SteeringDecision.CONTINUE,
            notes="Power is too tight, relaxing to see latency tradeoff",
        )
        updates = apply_steering_input(state, steering)
        rationale = updates.get("design_rationale", [])
        assert any("steering" in r.lower() for r in rationale)


# ---------------------------------------------------------------------------
# Rich rendering
# ---------------------------------------------------------------------------


class TestRenderOptimizationReview:
    def test_renders_string(self):
        state = _make_state(
            power=6.0,
            latency=35.0,
            verdicts={"power": "FAIL", "latency": "FAIL"},
            trajectory=_make_trajectory(3),
            iteration=3,
        )
        snap = build_optimization_review_snapshot(state)
        result = render_optimization_review(snap)
        assert "OPTIMIZATION REVIEW" in result
        assert "CONSTRAINT ANALYSIS" in result
        assert "OPTIMIZATION TRAJECTORY" in result
        assert "STRATEGY ANALYSIS" in result
        assert "STEERING OPTIONS" in result
        assert "Iteration 3" in result

    def test_all_pass_banner(self):
        state = _make_state(
            power=4.0,
            latency=30.0,
            verdicts={"power": "PASS", "latency": "PASS", "cost": "PASS"},
        )
        snap = build_optimization_review_snapshot(state)
        result = render_optimization_review(snap)
        assert "ALL CONSTRAINTS PASS" in result
