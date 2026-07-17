"""Seam S1 (issue #204): DesignState schema audit — reconcile all node writes to channels.

A LangGraph StateGraph merges a node's return dict into state ONLY for keys that
are declared channels on the state schema. Any returned key that is NOT a declared
`DesignState` field is silently dropped at runtime — no error, no stack trace. This
test enforces the invariant: **every key any loop node writes must be a declared
DesignState channel.**

Scope note: the unified Loop Convergence nodes (`loop_agents` /
`loop_convergence_graph`) are audited per-node via the `NODE_WRITES` table below.
The dispatcher loop (`soc_graph`/`dispatcher`/`specialists`) migrated to
`DesignState` in S2b (#206) and is covered structurally by
`test_designstate_is_superset_of_socdesignstate` — because its specialists write
`SoCDesignState` fields, a superset guarantee means none can be dropped. The MOO
loop (`optimization_loop`) migrates under S2a (#205). Extend as nodes migrate.
"""

import ast
import inspect
import textwrap

import pytest

from embodied_ai_architect.graphs.design_state import (
    DesignConstraints,
    DesignState,
    assert_declared_channels,
    create_initial_design_state,
    declared_channels,
    undeclared_keys,
)
from embodied_ai_architect.graphs.loop_agents import (
    critic_node,
    optimizer_node,
)
from embodied_ai_architect.graphs.loop_convergence_graph import (
    evaluate_node,
    recommend_node,
    seed_node,
)
from embodied_ai_architect.graphs.soc_state import SoCDesignState


def test_designstate_is_superset_of_socdesignstate():
    """S2b (#206): rebinding StateGraph(SoCDesignState) -> StateGraph(DesignState) is
    only safe if every SoCDesignState field is a declared DesignState channel — else
    dispatcher/specialist writes to the missing fields would be silently dropped.
    """
    missing = set(SoCDesignState.__annotations__) - declared_channels()
    assert not missing, f"DesignState is missing SoCDesignState channels: {sorted(missing)}"


# ---------------------------------------------------------------------------
# The audit table: node -> fields it may write -> (all must be DesignState channels)
# ---------------------------------------------------------------------------

NODE_WRITES: dict[str, set[str]] = {
    "seed_node": {"status", "design_space_config"},
    "critic_node": {
        "open_issues",
        "pending_deltas",
        "converged",
        "critic_diminishing_returns",
        "analysis",
        "research_citations",
    },
    "optimizer_node": {
        "design_space_config",
        "constraints",
        "pending_specialist_tasks",
        "applied_deltas",
        "open_issues",
        "pending_deltas",
        "iteration",
        # MOO-tool outputs re-emitted through the node's return value:
        "pareto_points",
        "pareto_frontier_history",
        "hypervolume_history",
        "knee_point",
        "sensitivity",
        "atlas",
        "moo_results",
    },
    "evaluate_node": {"ppa_metrics"},
    "recommend_node": {"status", "recommendation", "final_report"},
    # The MOO engine tool merges into state via optimizer_node; its output-key
    # contract must also be channels (mirrors make_moo_engine_tool._run).
    "moo_tool": {
        "pareto_points",
        "pareto_frontier_history",
        "hypervolume_history",
        "knee_point",
        "sensitivity",
        "atlas",
        "moo_results",
    },
}


def _fake_moo_tool(state: DesignState) -> dict:
    """Stand-in for make_moo_engine_tool's real adapter — same output-key contract."""
    knee = {"objectives": {"power_watts": 4.0, "latency_ms": 20.0}}
    return {
        "pareto_points": [knee],
        "pareto_frontier_history": [[knee]],
        "hypervolume_history": list(state.get("hypervolume_history", [])) + [1.0],
        "knee_point": knee,
        "sensitivity": {"power": {"x": 1.0}},
        "atlas": {"coverage": 0.5},
        "moo_results": {"layers_used": ["map_elites"]},
    }


def _representative_state() -> DesignState:
    """A state rich enough to drive every node down its populated path."""
    state = create_initial_design_state(
        "audit fixture",
        constraints=DesignConstraints(max_power_watts=5.0, max_latency_ms=33.0),
    )
    state["llm_available"] = False
    state["ppa_metrics"] = {"verdicts": {"power_watts": "FAIL", "latency_ms": "PASS"}}
    return state


def _run_all_nodes() -> dict[str, dict]:
    """Drive the loop once, capturing each node's return dict."""
    state = _representative_state()
    results: dict[str, dict] = {}

    results["seed_node"] = seed_node(state)
    state.update(results["seed_node"])

    results["critic_node"] = critic_node(state)
    state.update(results["critic_node"])

    results["moo_tool"] = _fake_moo_tool(state)

    results["optimizer_node"] = optimizer_node(state, moo_tool=_fake_moo_tool)
    state.update(results["optimizer_node"])

    results["evaluate_node"] = evaluate_node(state)
    state.update(results["evaluate_node"])

    results["recommend_node"] = recommend_node(state)
    return results


# ---------------------------------------------------------------------------
# Acceptance: the table is valid, and no node writes an undeclared key.
# ---------------------------------------------------------------------------


def test_no_duplicate_channels():
    """DesignState must not double-declare a channel (a duplicate silently shadows).

    `DesignState.__annotations__` is already deduplicated — a later declaration
    overwrites an earlier one before this test could see it — so we inspect the
    class *source* via AST to catch duplicates before Python collapses them.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(DesignState)))
    classdef = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "DesignState"
    )
    names = [
        node.target.id
        for node in classdef.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"DesignState has duplicate channel declarations: {dupes}"


@pytest.mark.parametrize("node,fields", sorted(NODE_WRITES.items()))
def test_audit_table_only_references_declared_channels(node, fields):
    """Every field in the checked-in table must be a declared DesignState channel."""
    extra = fields - declared_channels()
    assert not extra, f"NODE_WRITES['{node}'] references undeclared channels: {sorted(extra)}"


@pytest.mark.parametrize("node", sorted(NODE_WRITES))
def test_nodes_write_only_declared_channels(node):
    """Running each node, every returned key must be a declared channel (S1 invariant)."""
    produced = _run_all_nodes()[node]
    dropped = undeclared_keys(produced)
    assert not dropped, (
        f"{node} returns keys that are NOT DesignState channels and would be "
        f"silently dropped by LangGraph: {sorted(dropped)}"
    )


@pytest.mark.parametrize("node", sorted(NODE_WRITES))
def test_nodes_stay_within_their_declared_table_entry(node):
    """A node must not write keys outside its documented NODE_WRITES entry (drift guard)."""
    produced = set(_run_all_nodes()[node])
    undocumented = produced - NODE_WRITES[node]
    assert not undocumented, (
        f"{node} wrote keys absent from its NODE_WRITES entry: {sorted(undocumented)}. "
        f"Update the audit table (and confirm they are DesignState channels)."
    )


# ---------------------------------------------------------------------------
# The guard helpers themselves.
# ---------------------------------------------------------------------------


def test_undeclared_keys_detects_bad_key():
    assert undeclared_keys({"iteration": 1}) == set()
    assert undeclared_keys({"iteration": 1, "not_a_channel": 9}) == {"not_a_channel"}


def test_assert_declared_channels_raises_on_undeclared():
    assert_declared_channels({"status": "exploring"})  # no raise
    with pytest.raises(KeyError, match="not declared as channels"):
        assert_declared_channels({"bogus_field": 1}, node="critic")
