"""Tests for optimization transparency and human steering."""

from embodied_ai_architect.graphs.optimization_review import (
    OptimizationReviewSnapshot,
    OptimizationSteeringInput,
    SteeringDecision,
    analyze_strategies,
    apply_steering_input,
    build_optimization_review_snapshot,
    compute_constraint_slackness,
    extract_kpu_bandwidth_slackness,
    extract_kpu_floorplan_slackness,
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

    def test_sensitivity_extracted_from_real_producer_format(self):
        """Issue #24: sensitivity from BO layer is in
        {objective: {variable: {importance, lengthscale}}} format. The
        snapshot builder must normalize it to {variable: {objective: float}}.
        """
        state = _make_state(power=4.0, latency=30.0)
        # This is the EXACT shape produced by
        # bayesian_opt._extract_sensitivity (verified against the source).
        state["moo_results"] = {
            "sensitivity": {
                "power_watts": {
                    "quantization_dtype": {"lengthscale": 0.45, "importance": 0.82},
                    "npu_frequency_mhz": {"lengthscale": 0.55, "importance": 0.71},
                    "sram_size_kb": {"lengthscale": 2.10, "importance": 0.15},
                },
                "latency_ms": {
                    "quantization_dtype": {"lengthscale": 0.62, "importance": 0.65},
                    "npu_frequency_mhz": {"lengthscale": 0.70, "importance": 0.58},
                    "sram_size_kb": {"lengthscale": 0.50, "importance": 0.72},
                },
            },
            "total_evaluations": 100,
            "pareto_front": [],
        }
        snap = build_optimization_review_snapshot(state)
        assert snap.sensitivity
        # Normalized to variable-keyed
        assert "quantization_dtype" in snap.sensitivity
        # Each value is now a {objective: float} dict
        assert snap.sensitivity["quantization_dtype"]["power_watts"] == 0.82
        assert snap.sensitivity["quantization_dtype"]["latency_ms"] == 0.65
        assert snap.sensitivity["sram_size_kb"]["latency_ms"] == 0.72

    def test_sensitivity_empty_when_no_moo(self):
        """When MOO hasn't run, sensitivity should be an empty dict, not crash."""
        state = _make_state()
        snap = build_optimization_review_snapshot(state)
        assert snap.sensitivity == {}


class TestNormalizeSensitivity:
    """Direct unit tests for the normalize_sensitivity helper (issue #24)."""

    def test_normalize_real_producer_format(self):
        """The producer format {objective: {variable: {importance,
        lengthscale}}} must be transposed to {variable: {objective: float}}."""
        from embodied_ai_architect.graphs.optimization_review import normalize_sensitivity

        raw = {
            "power_watts": {
                "quantization_dtype": {"lengthscale": 0.45, "importance": 0.82},
                "npu_frequency_mhz": {"lengthscale": 0.55, "importance": 0.71},
            },
            "latency_ms": {
                "quantization_dtype": {"lengthscale": 0.62, "importance": 0.65},
                "npu_frequency_mhz": {"lengthscale": 0.70, "importance": 0.58},
            },
        }
        out = normalize_sensitivity(raw)
        assert set(out.keys()) == {"quantization_dtype", "npu_frequency_mhz"}
        assert out["quantization_dtype"]["power_watts"] == 0.82
        assert out["quantization_dtype"]["latency_ms"] == 0.65
        # Lengthscale must be dropped
        assert "lengthscale" not in out["quantization_dtype"]

    def test_normalize_already_normalized_passes_through(self):
        """If a future producer emits the normalized form directly, the
        helper should accept it as-is."""
        from embodied_ai_architect.graphs.optimization_review import normalize_sensitivity

        raw = {
            "quantization_dtype": {"power_watts": 0.82, "latency_ms": 0.65},
            "sram_size_kb": {"power_watts": 0.15, "latency_ms": 0.72},
        }
        out = normalize_sensitivity(raw)
        assert out == raw

    def test_normalize_objective_keyed_floats_transposed(self):
        """If the input is keyed by objectives but values are flat floats
        (not nested metrics), transpose to variable-keyed."""
        from embodied_ai_architect.graphs.optimization_review import normalize_sensitivity

        raw = {
            "power_watts": {"quantization_dtype": 0.82, "npu_frequency_mhz": 0.71},
            "latency_ms": {"quantization_dtype": 0.65, "npu_frequency_mhz": 0.58},
        }
        out = normalize_sensitivity(raw)
        assert out["quantization_dtype"]["power_watts"] == 0.82
        assert out["npu_frequency_mhz"]["latency_ms"] == 0.58

    def test_normalize_empty(self):
        from embodied_ai_architect.graphs.optimization_review import normalize_sensitivity

        assert normalize_sensitivity(None) == {}
        assert normalize_sensitivity({}) == {}

    def test_normalize_skips_missing_importance(self):
        """If a variable's metrics dict is missing the importance key, skip it."""
        from embodied_ai_architect.graphs.optimization_review import normalize_sensitivity

        raw = {
            "power_watts": {
                "good_var": {"importance": 0.5, "lengthscale": 0.4},
                "bad_var": {"lengthscale": 0.4},  # no importance
            },
        }
        out = normalize_sensitivity(raw)
        assert "good_var" in out
        assert "bad_var" not in out


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


# ---------------------------------------------------------------------------
# Issue #30: KPU inner-loop slackness
# ---------------------------------------------------------------------------


def _make_floorplan_estimate(feasible=True, pitch_matched=True):
    """Sample FloorplanEstimate dict matching the dataclass shape."""
    return {
        "compute_tile": {
            "width_mm": 2.10,
            "height_mm": 2.30,
            "area_mm2": 4.83,
            "sub_blocks": [],
        },
        "memory_tile": {
            "width_mm": 2.00,
            "height_mm": 2.40,
            "area_mm2": 4.80,
            "sub_blocks": [],
        },
        "pitch_matched": pitch_matched,
        "pitch_ratio_width": 1.05 if pitch_matched else 1.30,
        "pitch_ratio_height": 0.96,
        "pitch_tolerance": 0.15,
        "array_width_mm": 6.30,
        "array_height_mm": 6.90,
        "core_area_mm2": 43.5,
        "periphery_area_mm2": 4.7,
        "total_area_mm2": 48.2,
        "die_edge_mm": 7.0,
        "feasible": feasible,
        "max_die_area_mm2": 100.0,
        "issues": [] if feasible else ["Die area exceeds budget"],
    }


def _make_bandwidth_match(balanced=True):
    """Sample BandwidthMatchResult dict matching the validator output."""
    links = [
        {
            "name": "DRAM -> L3",
            "source": "dram",
            "sink": "l3",
            "available_gbps": 25.6,
            "required_gbps": 12.8,
            "utilization": 0.50,
            "bottleneck": False,
        },
        {
            "name": "L3 -> L2",
            "source": "l3",
            "sink": "l2",
            "available_gbps": 16.4,
            "required_gbps": 9.0,
            "utilization": 0.55,
            "bottleneck": False,
        },
        {
            "name": "L2 -> L1",
            "source": "l2",
            "sink": "l1",
            "available_gbps": 4.8,
            "required_gbps": 4.5,
            "utilization": 0.94,
            "bottleneck": not balanced,
        },
        {
            "name": "L1 -> compute",
            "source": "l1",
            "sink": "compute",
            "available_gbps": 3.2,
            "required_gbps": 1.4,
            "utilization": 0.43,
            "bottleneck": False,
        },
    ]
    return {
        "links": links,
        "balanced": balanced,
        "bottleneck_link": None if balanced else "L2 -> L1",
        "peak_utilization": 0.94,
        "ingress_gbps": 12.8,
        "egress_gbps": 1.4,
        "compute_demand_gbps": 12.8,
        "issues": [] if balanced else ["L2 -> L1 saturated at 94%"],
    }


class TestKPUFloorplanSlackness:
    def test_extract_returns_none_when_no_estimate(self):
        state = _make_state()
        assert extract_kpu_floorplan_slackness(state) is None

    def test_extract_populates_all_fields(self):
        state = _make_state()
        state["floorplan_estimate"] = _make_floorplan_estimate(feasible=True)
        fp = extract_kpu_floorplan_slackness(state)
        assert fp is not None
        assert fp.compute_tile_width_mm == 2.10
        assert fp.compute_tile_height_mm == 2.30
        assert fp.memory_tile_width_mm == 2.00
        assert fp.memory_tile_height_mm == 2.40
        assert fp.pitch_matched is True
        assert fp.feasible is True
        assert fp.total_area_mm2 == 48.2
        assert fp.max_die_area_mm2 == 100.0
        assert fp.area_utilization_pct == 48.2

    def test_extract_propagates_failure_state(self):
        state = _make_state()
        state["floorplan_estimate"] = _make_floorplan_estimate(feasible=False)
        fp = extract_kpu_floorplan_slackness(state)
        assert fp.feasible is False
        assert "Die area exceeds budget" in fp.issues


class TestKPUBandwidthSlackness:
    def test_extract_returns_none_when_no_match(self):
        state = _make_state()
        assert extract_kpu_bandwidth_slackness(state) is None

    def test_extract_links_with_status_classification(self):
        state = _make_state()
        state["bandwidth_match"] = _make_bandwidth_match(balanced=True)
        bw = extract_kpu_bandwidth_slackness(state)
        assert bw is not None
        assert len(bw.links) == 4
        # 50% → OK, 55% → OK, 94% → TIGHT, 43% → OK
        statuses = {link.name: link.status for link in bw.links}
        assert statuses["DRAM -> L3"] == "OK"
        assert statuses["L2 -> L1"] == "TIGHT"
        assert statuses["L1 -> compute"] == "OK"
        # utilization is converted from fraction to percent
        l2_link = next(link for link in bw.links if link.name == "L2 -> L1")
        assert l2_link.utilization_pct == 94.0

    def test_extract_marks_bottleneck_when_unbalanced(self):
        state = _make_state()
        state["bandwidth_match"] = _make_bandwidth_match(balanced=False)
        bw = extract_kpu_bandwidth_slackness(state)
        assert bw.balanced is False
        assert bw.bottleneck_link == "L2 -> L1"
        # The bottleneck link should be classified BOTTLENECK regardless of util
        l2_link = next(link for link in bw.links if link.name == "L2 -> L1")
        assert l2_link.bottleneck is True
        assert l2_link.status == "BOTTLENECK"


class TestSnapshotIncludesKPUSlackness:
    def test_snapshot_carries_kpu_floorplan_when_present(self):
        state = _make_state()
        state["floorplan_estimate"] = _make_floorplan_estimate()
        state["bandwidth_match"] = _make_bandwidth_match()
        snap = build_optimization_review_snapshot(state)
        assert snap.kpu_floorplan is not None
        assert snap.kpu_bandwidth is not None
        assert snap.kpu_floorplan.pitch_matched is True
        assert len(snap.kpu_bandwidth.links) == 4

    def test_snapshot_omits_kpu_when_no_state_data(self):
        state = _make_state()
        # No floorplan_estimate / bandwidth_match
        snap = build_optimization_review_snapshot(state)
        assert snap.kpu_floorplan is None
        assert snap.kpu_bandwidth is None


class TestKPURegressionsCodeRabbitPR80:
    """Regression tests for the three CodeRabbit findings on PR #80."""

    def test_status_uses_unrounded_fraction(self):
        """A 0.8496 link must stay OK — rounding to 85.0% must not flip TIGHT."""
        state = _make_state()
        bw_dict = _make_bandwidth_match()
        # Replace one link with a value that crosses the rounding boundary
        bw_dict["links"][2]["utilization"] = 0.8496
        bw_dict["links"][2]["bottleneck"] = False
        state["bandwidth_match"] = bw_dict
        bw = extract_kpu_bandwidth_slackness(state)
        link = bw.links[2]
        # Display percentage rounds to 85.0 ...
        assert link.utilization_pct == 85.0
        # ... but the classifier saw the raw 0.8496 and kept OK
        assert link.status == "OK"

    def test_status_at_exact_threshold(self):
        """0.85 exactly → TIGHT, 1.0 exactly → BOTTLENECK."""
        state = _make_state()
        bw_dict = _make_bandwidth_match()
        bw_dict["links"][2]["utilization"] = 0.85
        bw_dict["links"][2]["bottleneck"] = False
        state["bandwidth_match"] = bw_dict
        bw = extract_kpu_bandwidth_slackness(state)
        assert bw.links[2].status == "TIGHT"

        bw_dict["links"][2]["utilization"] = 1.0
        state["bandwidth_match"] = bw_dict
        bw = extract_kpu_bandwidth_slackness(state)
        assert bw.links[2].status == "BOTTLENECK"

    def test_pitch_zero_preserved(self):
        """Explicit 0.0 pitch ratios from the validator must not be rewritten."""
        state = _make_state()
        fp = _make_floorplan_estimate()
        fp["pitch_ratio_width"] = 0.0  # degenerate case the validator can produce
        fp["pitch_ratio_height"] = 0.0
        fp["pitch_tolerance"] = 0.0
        state["floorplan_estimate"] = fp
        out = extract_kpu_floorplan_slackness(state)
        assert out.pitch_ratio_width == 0.0
        assert out.pitch_ratio_height == 0.0
        assert out.pitch_tolerance == 0.0


class TestRenderKPUSections:
    def test_renders_bandwidth_chain(self):
        state = _make_state()
        state["bandwidth_match"] = _make_bandwidth_match()
        snap = build_optimization_review_snapshot(state)
        out = render_optimization_review(snap)
        assert "KPU BANDWIDTH CHAIN" in out
        assert "DRAM -> L3" in out
        assert "L2 -> L1" in out
        assert "TIGHT" in out  # the 94% link
        assert "GB/s" in out

    def test_renders_floorplan(self):
        state = _make_state()
        state["floorplan_estimate"] = _make_floorplan_estimate()
        snap = build_optimization_review_snapshot(state)
        out = render_optimization_review(snap)
        assert "KPU FLOORPLAN" in out
        assert "Compute tile" in out
        assert "Memory tile" in out
        assert "Pitch ratio" in out
        assert "Total die area" in out
        assert "100" in out  # the budget

    def test_no_kpu_sections_when_state_empty(self):
        state = _make_state()
        snap = build_optimization_review_snapshot(state)
        out = render_optimization_review(snap)
        assert "KPU BANDWIDTH CHAIN" not in out
        assert "KPU FLOORPLAN" not in out


# ---------------------------------------------------------------------------
# Issue #34: KPU convergence history in the snapshot
# ---------------------------------------------------------------------------


class TestKPUConvergenceHistoryInSnapshot:
    def test_snapshot_carries_kpu_history_when_present(self):
        state = _make_state()
        state["kpu_optimization_history"] = [
            {
                "source": "kpu_configurator",
                "iteration": 0,
                "config_name": "swkpu-test",
                "compute_array": "16x16",
                "l2_size_bytes": 262144,
                "summary": "initial sizing",
            },
            {
                "source": "floorplan_validator",
                "iteration": 0,
                "pitch_matched": False,
                "total_area_mm2": 80.0,
                "floorplan_feasible": True,
                "summary": "pitch=FAIL",
            },
            {
                "source": "kpu_optimizer",
                "iteration": 1,
                "config_name": "swkpu-test",
                "compute_array": "12x16",
                "changes": ["Reduced systolic array cols to 12 (pitch match)"],
                "summary": "applied 1 change",
            },
        ]
        snap = build_optimization_review_snapshot(state)
        assert len(snap.kpu_history) == 3
        assert [e["source"] for e in snap.kpu_history] == [
            "kpu_configurator",
            "floorplan_validator",
            "kpu_optimizer",
        ]

    def test_snapshot_kpu_history_empty_by_default(self):
        state = _make_state()
        snap = build_optimization_review_snapshot(state)
        assert snap.kpu_history == []

    def test_render_includes_kpu_history_section(self):
        state = _make_state()
        state["kpu_optimization_history"] = [
            {
                "source": "kpu_configurator",
                "iteration": 0,
                "config_name": "swkpu-test",
                "compute_array": "16x16",
                "summary": "initial sizing",
            },
            {
                "source": "kpu_optimizer",
                "iteration": 1,
                "config_name": "swkpu-test",
                "compute_array": "12x16",
                "pitch_matched": True,
                "total_area_mm2": 48.2,
                "changes": ["Reduced systolic array cols to 12 (pitch match)"],
                "summary": "applied 1 change",
            },
        ]
        snap = build_optimization_review_snapshot(state)
        out = render_optimization_review(snap)
        assert "KPU CONVERGENCE HISTORY" in out
        assert "kpu_configurator" in out
        assert "kpu_optimizer" in out
        assert "16x16" in out
        assert "Reduced systolic array cols" in out

    def test_no_section_when_history_empty(self):
        state = _make_state()
        snap = build_optimization_review_snapshot(state)
        out = render_optimization_review(snap)
        assert "KPU CONVERGENCE HISTORY" not in out
