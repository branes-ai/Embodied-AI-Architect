"""Working memory for specialist agents in the SoC design loop.

Tracks what each agent has tried, decided, and discovered across iterations.
This prevents the optimizer from re-trying failed strategies and enables
agents to learn from previous attempts within a design session.

Working memory is serialized into SoCDesignState["working_memory"] via
model_dump() and reconstructed via WorkingMemoryStore(**data).

Usage:
    from embodied_ai_architect.graphs.memory import WorkingMemoryStore

    store = WorkingMemoryStore()
    store.record_attempt(
        agent_name="design_optimizer",
        description="INT8 quantization",
        outcome="Power reduced from 6.3W to 5.1W — still FAIL",
        iteration=1,
    )

    mem = store.get_agent_memory("design_optimizer")
    already_tried = [t["description"] for t in mem.things_tried]
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentWorkingMemory(BaseModel):
    """Per-agent working memory across optimization iterations."""

    agent_name: str
    decisions_made: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    constraints_discovered: list[str] = Field(default_factory=list)
    things_tried: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of {description, outcome, iteration} dicts",
    )
    iteration_notes: dict[int, str] = Field(
        default_factory=dict,
        description="Iteration number -> freeform notes",
    )


class WorkingMemoryStore(BaseModel):
    """Collection of per-agent working memories.

    Serializes to/from dict for TypedDict compatibility with SoCDesignState.
    """

    agents: dict[str, AgentWorkingMemory] = Field(default_factory=dict)

    def get_agent_memory(self, agent_name: str) -> AgentWorkingMemory:
        """Get or create working memory for an agent."""
        if agent_name not in self.agents:
            self.agents[agent_name] = AgentWorkingMemory(agent_name=agent_name)
        return self.agents[agent_name]

    def record_attempt(
        self,
        agent_name: str,
        description: str,
        outcome: str,
        iteration: int,
    ) -> None:
        """Record an attempt by an agent.

        Args:
            agent_name: Which agent made the attempt.
            description: What was tried.
            outcome: What happened (success/failure summary).
            iteration: Optimization iteration number.
        """
        mem = self.get_agent_memory(agent_name)
        mem.things_tried.append(
            {
                "description": description,
                "outcome": outcome,
                "iteration": iteration,
            }
        )

    def record_decision(self, agent_name: str, decision: str) -> None:
        """Record a decision made by an agent."""
        mem = self.get_agent_memory(agent_name)
        mem.decisions_made.append(decision)

    def record_constraint(self, agent_name: str, constraint: str) -> None:
        """Record a constraint discovered by an agent."""
        mem = self.get_agent_memory(agent_name)
        mem.constraints_discovered.append(constraint)

    def get_tried_descriptions(self, agent_name: str) -> list[str]:
        """Get list of descriptions of things already tried by an agent."""
        mem = self.get_agent_memory(agent_name)
        return [t["description"] for t in mem.things_tried]
