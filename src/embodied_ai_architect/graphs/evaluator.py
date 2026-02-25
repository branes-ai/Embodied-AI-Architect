"""AgenticEvaluator — 9-dimension evaluation framework for SoC design sessions.

Runs all 9 scoring functions against a RunTrace, computes weighted composite
scores, and produces Scorecards.

Usage:
    from embodied_ai_architect.graphs.evaluator import AgenticEvaluator

    evaluator = AgenticEvaluator(gold_standards=ALL_GOLD_STANDARDS)
    scorecard = evaluator.evaluate_run(trace)
    print(scorecard.summary())
"""

from __future__ import annotations

from typing import Any

from embodied_ai_architect.graphs.evaluation import (
    DimensionScore,
    GoldStandard,
    RunTrace,
    Scorecard,
)
from embodied_ai_architect.graphs.scoring import (
    score_adaptability,
    score_convergence,
    score_decomposition,
    score_efficiency,
    score_exploration_efficiency,
    score_governance,
    score_ppa_accuracy,
    score_reasoning,
    score_tool_use,
)

# Default weights from master implementation plan
DEFAULT_WEIGHTS = {
    "decomposition": 0.15,
    "ppa_accuracy": 0.20,
    "exploration_efficiency": 0.10,
    "reasoning": 0.15,
    "convergence": 0.10,
    "governance": 0.10,
    "tool_use": 0.10,
    "adaptability": 0.05,
    "efficiency": 0.05,
}

# All 9 scorers in order
SCORERS = [
    score_decomposition,
    score_ppa_accuracy,
    score_exploration_efficiency,
    score_reasoning,
    score_convergence,
    score_governance,
    score_tool_use,
    score_adaptability,
    score_efficiency,
]

# Default passing threshold
DEFAULT_PASSING_THRESHOLD = 0.75


class AgenticEvaluator:
    """Evaluates agentic SoC design runs across 9 dimensions.

    Args:
        gold_standards: Mapping of demo_name -> GoldStandard.
        weights: Optional override of dimension weights. Defaults to plan values.
        passing_threshold: Minimum composite score to pass. Default 0.75.
    """

    def __init__(
        self,
        gold_standards: dict[str, GoldStandard] | None = None,
        weights: dict[str, float] | None = None,
        passing_threshold: float = DEFAULT_PASSING_THRESHOLD,
    ) -> None:
        self.gold_standards = gold_standards or {}
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.passing_threshold = passing_threshold

    def evaluate_run(self, trace: RunTrace) -> Scorecard:
        """Evaluate a single run trace against its gold standard.

        Args:
            trace: The run trace to evaluate.

        Returns:
            Scorecard with all 9 dimension scores and composite.

        Raises:
            KeyError: If no gold standard exists for the trace's demo_name.
        """
        gold = self.gold_standards.get(trace.demo_name)
        if gold is None:
            raise KeyError(
                f"No gold standard for demo '{trace.demo_name}'. "
                f"Available: {list(self.gold_standards.keys())}"
            )

        dimensions: list[DimensionScore] = []
        for scorer in SCORERS:
            dim_score = scorer(trace, gold)
            # Apply weight
            dim_score.weight = self.weights.get(dim_score.dimension, 0.0)
            dimensions.append(dim_score)

        # Compute weighted composite
        total_weight = sum(d.weight for d in dimensions)
        if total_weight > 0:
            composite = sum(d.score * d.weight for d in dimensions) / total_weight
        else:
            composite = sum(d.score for d in dimensions) / len(dimensions) if dimensions else 0.0

        return Scorecard(
            demo_name=trace.demo_name,
            dimensions=dimensions,
            composite_score=round(composite, 4),
            passed=composite >= self.passing_threshold,
        )

    def evaluate_all(self, traces: dict[str, RunTrace]) -> dict[str, Scorecard]:
        """Evaluate multiple run traces.

        Args:
            traces: Mapping of demo_name -> RunTrace.

        Returns:
            Mapping of demo_name -> Scorecard.
        """
        results = {}
        for demo_name, trace in traces.items():
            if not trace.demo_name:
                trace.demo_name = demo_name
            try:
                results[demo_name] = self.evaluate_run(trace)
            except KeyError:
                # Skip demos without gold standards
                pass
        return results

    def capture_run_trace(
        self,
        initial_state: dict[str, Any],
        final_state: dict[str, Any],
        demo_name: str,
        duration_seconds: float = 0.0,
    ) -> RunTrace:
        """Extract a RunTrace from initial/final state diff.

        Convenience method for building traces from dispatcher output.

        Args:
            initial_state: State before the run.
            final_state: State after the run.
            demo_name: Name of the demo.
            duration_seconds: Wall-clock duration.

        Returns:
            Populated RunTrace.
        """
        # Extract tool calls from history
        history = final_state.get("history", [])
        tool_calls = []
        for entry in history:
            agent = entry.get("agent", "")
            if agent and "Completed task" in entry.get("action", ""):
                tool_calls.append(agent)

        # Count failures and recoveries from task graph
        task_graph = final_state.get("task_graph", {})
        nodes = task_graph.get("nodes", {})
        failures = sum(1 for n in nodes.values() if n.get("status") == "failed")
        sum(1 for n in nodes.values() if n.get("status") == "completed")
        # A recovery is a task that completed after a previous failure in the same agent
        failed_agents = {n.get("agent") for n in nodes.values() if n.get("status") == "failed"}
        completed_agents = {
            n.get("agent") for n in nodes.values() if n.get("status") == "completed"
        }
        recoveries = len(failed_agents & completed_agents)

        # Count human interventions from audit log
        audit_log = final_state.get("audit_log", [])
        human_interventions = sum(1 for entry in audit_log if entry.get("human_approved", False))

        # Cost tracking
        cost_tokens = sum(entry.get("cost_tokens", 0) for entry in audit_log)

        return RunTrace(
            demo_name=demo_name,
            task_graph=task_graph,
            ppa_metrics=final_state.get("ppa_metrics", {}),
            iteration_history=final_state.get("optimization_history", []),
            tool_calls=tool_calls,
            audit_log=audit_log,
            failures=failures,
            recoveries=recoveries,
            duration_seconds=duration_seconds,
            cost_tokens=cost_tokens,
            human_interventions=human_interventions,
            design_rationale=final_state.get("design_rationale", []),
            pareto_points=final_state.get("pareto_results", {}).get("front", []),
            history=history,
        )
