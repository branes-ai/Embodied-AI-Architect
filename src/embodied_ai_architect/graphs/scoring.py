"""9-dimension scoring functions for the agentic evaluation framework.

Each function scores a single evaluation dimension by comparing a RunTrace
against a GoldStandard, returning a DimensionScore in [0.0, 1.0].

Dimensions:
1. Decomposition — task graph quality vs gold standard
2. PPA Accuracy — estimated vs expected PPA metrics
3. Exploration Efficiency — Pareto points / total designs
4. Reasoning — rationale quality (keyword + structure)
5. Convergence — monotonic improvement rate across iterations
6. Governance — audit log coverage of expected triggers
7. Tool Use — precision/recall vs expected tool calls
8. Adaptability — recovery rate from failures
9. Efficiency — time and cost vs budgets

Usage:
    from embodied_ai_architect.graphs.scoring import score_decomposition
    dim = score_decomposition(trace, gold)
"""

from __future__ import annotations

from typing import Any

from embodied_ai_architect.graphs.evaluation import DimensionScore, GoldStandard, RunTrace


def score_decomposition(trace: RunTrace, gold: GoldStandard) -> DimensionScore:
    """Score task graph decomposition quality.

    Compares node names and dependency edges against the gold standard.
    Score = (node_match_ratio + edge_match_ratio) / 2.
    """
    current_nodes = _extract_node_set(trace.task_graph)
    expected_nodes = _extract_node_set(gold.expected_task_graph)

    if not expected_nodes:
        return DimensionScore(
            dimension="decomposition",
            score=1.0 if current_nodes else 0.5,
            details="No expected task graph defined",
        )

    # Node name overlap (Jaccard-like)
    if current_nodes or expected_nodes:
        intersection = current_nodes & expected_nodes
        union = current_nodes | expected_nodes
        node_score = len(intersection) / len(union) if union else 0.0
    else:
        node_score = 1.0

    # Edge overlap
    current_edges = _extract_edge_set(trace.task_graph)
    expected_edges = _extract_edge_set(gold.expected_task_graph)
    if current_edges or expected_edges:
        edge_intersection = current_edges & expected_edges
        edge_union = current_edges | expected_edges
        edge_score = len(edge_intersection) / len(edge_union) if edge_union else 0.0
    else:
        edge_score = 1.0

    score = (node_score + edge_score) / 2
    details = f"Nodes: {len(current_nodes & expected_nodes)}/{len(expected_nodes)} match, Edges: {edge_score:.2f}"
    return DimensionScore(dimension="decomposition", score=round(score, 3), details=details)


def score_ppa_accuracy(trace: RunTrace, gold: GoldStandard) -> DimensionScore:
    """Score PPA estimation accuracy.

    For each PPA metric, computes |estimated - expected| / expected.
    Tiered scoring: <10% = 1.0, 10-25% = 0.75, 25-50% = 0.5, >50% = 0.25.
    """
    expected_ppa = gold.expected_ppa
    actual_ppa = trace.ppa_metrics

    if not expected_ppa:
        return DimensionScore(
            dimension="ppa_accuracy",
            score=0.75,
            details="No expected PPA defined",
        )

    metrics = ["power_watts", "latency_ms", "area_mm2", "cost_usd"]
    scores: list[float] = []
    detail_parts: list[str] = []

    for m in metrics:
        expected = expected_ppa.get(m)
        actual = actual_ppa.get(m)
        if expected is None or actual is None:
            continue
        if expected == 0:
            scores.append(1.0 if actual == 0 else 0.25)
            continue

        error = abs(actual - expected) / abs(expected)
        if error < 0.10:
            s = 1.0
        elif error < 0.25:
            s = 0.75
        elif error < 0.50:
            s = 0.5
        else:
            s = 0.25
        scores.append(s)
        detail_parts.append(f"{m}: {error:.0%} err -> {s}")

    score = sum(scores) / len(scores) if scores else 0.5
    details = "; ".join(detail_parts) if detail_parts else "No comparable metrics"
    return DimensionScore(dimension="ppa_accuracy", score=round(score, 3), details=details)


def score_exploration_efficiency(trace: RunTrace, gold: GoldStandard) -> DimensionScore:
    """Score exploration efficiency via Pareto analysis.

    Score = non_dominated_points / total_points.
    Falls back to 0.5 if no Pareto exploration was expected.
    """
    if gold.expected_pareto_points <= 0:
        return DimensionScore(
            dimension="exploration_efficiency",
            score=0.75,
            details="No Pareto exploration expected for this demo",
        )

    pareto_points = trace.pareto_points
    total = len(pareto_points)
    if total == 0:
        return DimensionScore(
            dimension="exploration_efficiency",
            score=0.0,
            details="No design points explored",
        )

    non_dominated = sum(1 for p in pareto_points if not p.get("dominated", True))
    ratio = non_dominated / total
    score = min(1.0, ratio * 1.5)  # scale up slightly since some domination is expected

    # Also check against expected count
    if non_dominated >= gold.expected_pareto_points:
        score = min(1.0, score + 0.2)

    details = f"{non_dominated}/{total} non-dominated (expected >= {gold.expected_pareto_points})"
    return DimensionScore(
        dimension="exploration_efficiency",
        score=round(score, 3),
        details=details,
    )


def score_reasoning(trace: RunTrace, gold: GoldStandard) -> DimensionScore:
    """Score reasoning quality from design rationale.

    Checks for: keyword presence, rationale length, structure (agent prefixes).
    """
    rationale = trace.design_rationale
    keywords = gold.rationale_keywords

    if not rationale:
        return DimensionScore(
            dimension="reasoning",
            score=0.25,
            details="No design rationale recorded",
        )

    # Keyword matching (50% weight)
    keyword_score = 0.0
    if keywords:
        text = " ".join(rationale).lower()
        matched = sum(1 for kw in keywords if kw.lower() in text)
        keyword_score = matched / len(keywords)
    else:
        keyword_score = 0.5  # no keywords to check

    # Structure check (25% weight) — expect [agent] prefix pattern
    structured_count = sum(1 for r in rationale if r.startswith("["))
    structure_score = min(1.0, structured_count / max(len(rationale), 1))

    # Length check (25% weight) — reasonable rationale length
    avg_length = sum(len(r) for r in rationale) / max(len(rationale), 1)
    length_score = min(1.0, avg_length / 50)  # expect ~50+ chars average

    score = keyword_score * 0.5 + structure_score * 0.25 + length_score * 0.25
    details = f"Keywords: {keyword_score:.2f}, Structure: {structure_score:.2f}, Length: {length_score:.2f}"
    return DimensionScore(dimension="reasoning", score=round(score, 3), details=details)


def score_convergence(trace: RunTrace, gold: GoldStandard) -> DimensionScore:
    """Score convergence behavior across optimization iterations.

    Measures monotonic improvement rate: fraction of iterations where
    the number of passing constraints increased or stayed the same.
    """
    history = trace.iteration_history
    if len(history) <= 1:
        # Single pass — check if it passed
        verdicts = trace.ppa_metrics.get("verdicts", {})
        all_pass = all(v == "PASS" for v in verdicts.values()) if verdicts else False
        score = 1.0 if all_pass else 0.5
        return DimensionScore(
            dimension="convergence",
            score=score,
            details="Single iteration" + (" — all PASS" if all_pass else ""),
        )

    # Count passing constraints per iteration
    pass_counts = []
    for snapshot in history:
        verdicts = snapshot.get("verdicts", snapshot.get("ppa_metrics", {}).get("verdicts", {}))
        pass_count = sum(1 for v in verdicts.values() if v == "PASS")
        pass_counts.append(pass_count)

    # Monotonic improvement rate
    improvements = 0
    for i in range(1, len(pass_counts)):
        if pass_counts[i] >= pass_counts[i - 1]:
            improvements += 1

    rate = improvements / (len(pass_counts) - 1) if len(pass_counts) > 1 else 0.0

    # Bonus for finishing with all PASS
    final_verdicts = trace.ppa_metrics.get("verdicts", {})
    all_pass = all(v == "PASS" for v in final_verdicts.values()) if final_verdicts else False
    if all_pass:
        rate = min(1.0, rate + 0.2)

    details = f"Improvement rate: {rate:.2f} over {len(history)} iterations"
    return DimensionScore(dimension="convergence", score=round(rate, 3), details=details)


def score_governance(trace: RunTrace, gold: GoldStandard) -> DimensionScore:
    """Score governance compliance.

    Checks that all expected governance triggers appear in the audit log.
    """
    expected_triggers = gold.governance_triggers
    audit_log = trace.audit_log

    if not expected_triggers:
        return DimensionScore(
            dimension="governance",
            score=1.0,
            details="No governance triggers expected",
        )

    if not audit_log:
        return DimensionScore(
            dimension="governance",
            score=0.0,
            details=f"No audit log but {len(expected_triggers)} triggers expected",
        )

    # Check each expected trigger against audit log actions
    audit_actions = [entry.get("action", "") for entry in audit_log]
    audit_text = " ".join(audit_actions).lower()

    matched = 0
    for trigger in expected_triggers:
        if trigger.lower() in audit_text:
            matched += 1

    score = matched / len(expected_triggers)
    details = f"Governance triggers: {matched}/{len(expected_triggers)} matched"
    return DimensionScore(dimension="governance", score=round(score, 3), details=details)


def score_tool_use(trace: RunTrace, gold: GoldStandard) -> DimensionScore:
    """Score tool use precision and recall.

    Precision = expected tools called / total tools called.
    Recall = expected tools called / expected tools.
    Score = F1 of precision and recall.
    """
    expected_tools = set(gold.expected_tool_calls)
    actual_tools = set(trace.tool_calls)

    if not expected_tools:
        return DimensionScore(
            dimension="tool_use",
            score=0.75,
            details="No expected tool calls defined",
        )

    if not actual_tools:
        return DimensionScore(
            dimension="tool_use",
            score=0.0,
            details="No tools called",
        )

    true_positives = len(expected_tools & actual_tools)
    precision = true_positives / len(actual_tools) if actual_tools else 0.0
    recall = true_positives / len(expected_tools) if expected_tools else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    details = f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} ({true_positives} matched)"
    return DimensionScore(dimension="tool_use", score=round(f1, 3), details=details)


def score_adaptability(trace: RunTrace, gold: GoldStandard) -> DimensionScore:
    """Score adaptability: recovery rate from failures.

    Score = recoveries / failures. No failures = perfect score.
    """
    failures = trace.failures
    recoveries = trace.recoveries

    if failures == 0:
        return DimensionScore(
            dimension="adaptability",
            score=1.0,
            details="No failures encountered — perfect adaptability",
        )

    recovery_rate = min(1.0, recoveries / failures)
    details = f"Recovered {recoveries}/{failures} failures"
    return DimensionScore(dimension="adaptability", score=round(recovery_rate, 3), details=details)


def score_efficiency(trace: RunTrace, gold: GoldStandard) -> DimensionScore:
    """Score efficiency: time and cost vs budgets.

    Time score and cost score are averaged. Each compares actual vs budget
    with a tiered scale.
    """
    scores: list[float] = []
    detail_parts: list[str] = []

    # Time efficiency
    if gold.max_duration_seconds > 0 and trace.duration_seconds > 0:
        time_ratio = trace.duration_seconds / gold.max_duration_seconds
        if time_ratio <= 0.5:
            t_score = 1.0
        elif time_ratio <= 1.0:
            t_score = 0.75
        elif time_ratio <= 2.0:
            t_score = 0.5
        else:
            t_score = 0.25
        scores.append(t_score)
        detail_parts.append(f"Time: {trace.duration_seconds:.1f}s/{gold.max_duration_seconds:.0f}s")

    # Cost efficiency
    if gold.max_cost_tokens > 0 and trace.cost_tokens > 0:
        cost_ratio = trace.cost_tokens / gold.max_cost_tokens
        if cost_ratio <= 0.5:
            c_score = 1.0
        elif cost_ratio <= 1.0:
            c_score = 0.75
        elif cost_ratio <= 2.0:
            c_score = 0.5
        else:
            c_score = 0.25
        scores.append(c_score)
        detail_parts.append(f"Tokens: {trace.cost_tokens}/{gold.max_cost_tokens}")

    # Human interventions
    if gold.max_human_interventions >= 0:
        if trace.human_interventions <= gold.max_human_interventions:
            h_score = 1.0
        else:
            excess = trace.human_interventions - gold.max_human_interventions
            h_score = max(0.0, 1.0 - excess * 0.2)
        scores.append(h_score)
        detail_parts.append(f"HITL: {trace.human_interventions}/{gold.max_human_interventions}")

    score = sum(scores) / len(scores) if scores else 0.5
    details = "; ".join(detail_parts) if detail_parts else "No efficiency metrics available"
    return DimensionScore(dimension="efficiency", score=round(score, 3), details=details)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_node_set(task_graph_dict: dict[str, Any]) -> set[str]:
    """Extract agent names from a serialized TaskGraph."""
    nodes = task_graph_dict.get("nodes", {})
    return {node.get("agent", "") for node in nodes.values() if node.get("agent")}


def _extract_edge_set(task_graph_dict: dict[str, Any]) -> set[tuple[str, str]]:
    """Extract dependency edges as (dependency_agent, node_agent) tuples."""
    nodes = task_graph_dict.get("nodes", {})
    edges: set[tuple[str, str]] = set()
    for node in nodes.values():
        node_agent = node.get("agent", "")
        for dep_id in node.get("dependencies", []):
            dep_node = nodes.get(dep_id, {})
            dep_agent = dep_node.get("agent", "")
            if dep_agent and node_agent:
                edges.add((dep_agent, node_agent))
    return edges
