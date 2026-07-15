"""Seam S6 (issue #211): the loop's Pareto accumulation is monotonic — dominated
points are removed and no non-dominated point is lost across iterations."""

from embodied_ai_architect.graphs.loop_convergence_graph import _merge_pareto


def _pt(power: float, latency: float) -> dict:
    return {"objectives": {"power_watts": power, "latency_ms": latency}}


def _objs(points: list[dict]) -> set[tuple[float, float]]:
    """Full (power, latency) objective vectors — not just power, so a regression that
    corrupts latency while keeping power can't slip through the assertions."""
    return {(p["objectives"]["power_watts"], p["objectives"]["latency_ms"]) for p in points}


def test_first_merge_returns_new_points() -> None:
    merged = _merge_pareto([], [_pt(5.0, 20.0), _pt(4.0, 30.0)])
    assert _objs(merged) == {(5.0, 20.0), (4.0, 30.0)}


def test_dominated_new_point_is_dropped() -> None:
    frontier = _merge_pareto([], [_pt(5.0, 20.0)])  # A
    # B (6,25) is dominated by A on both objectives; C (4,30) is non-dominated.
    merged = _merge_pareto(frontier, [_pt(6.0, 25.0), _pt(4.0, 30.0)])
    assert (5.0, 20.0) in _objs(merged)  # A survived
    assert (4.0, 30.0) in _objs(merged)  # C added
    assert (6.0, 25.0) not in _objs(merged)  # B dominated -> dropped


def test_non_dominated_points_never_lost_when_only_dominated_added() -> None:
    frontier = _merge_pareto([], [_pt(5.0, 20.0), _pt(4.0, 30.0)])
    before = _objs(frontier)
    # Add strictly worse points across three more iterations.
    for worse in (_pt(10.0, 50.0), _pt(9.0, 60.0), _pt(8.0, 40.0)):
        frontier = _merge_pareto(frontier, [worse])
    # Every originally non-dominated point is still present (monotonicity).
    assert before.issubset(_objs(frontier))


def test_new_point_that_dominates_accumulated_removes_it() -> None:
    frontier = _merge_pareto([], [_pt(6.0, 30.0)])  # A
    # D (5,20) dominates A -> A must be removed, D kept.
    merged = _merge_pareto(frontier, [_pt(5.0, 20.0)])
    assert _objs(merged) == {(5.0, 20.0)}


def test_points_keep_engine_native_shape() -> None:
    merged = _merge_pareto([], [_pt(5.0, 20.0)])
    # Consumers (evaluate_node/recommend_node) read point["objectives"][...].
    assert merged[0]["objectives"] == {"power_watts": 5.0, "latency_ms": 20.0}
