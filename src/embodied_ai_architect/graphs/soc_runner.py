"""High-level runner for the SoC design optimization loop.

Wraps the compiled LangGraph StateGraph with sensible defaults,
optional checkpointing, and convenient run/resume methods.

Supports two modes:
- Batch: run() executes the full design loop and returns the final state.
- Interactive: start() + step() for human-in-the-loop review at plan and
  optimization stages.

Usage:
    from embodied_ai_architect.graphs.soc_runner import SoCDesignRunner

    # Batch mode
    runner = SoCDesignRunner()
    result = runner.run(
        goal="Design a drone SoC: <5W, <33ms, <$30",
        use_case="delivery_drone",
        platform="drone",
        constraints=DesignConstraints(max_power_watts=5.0, max_latency_ms=33.3),
    )

    # Interactive mode
    runner = SoCDesignRunner(human_review=True, optimization_review=True)
    status, state = runner.start(
        goal="Design a drone SoC: <5W, <33ms, <$30",
        constraints=DesignConstraints(max_power_watts=5.0, max_latency_ms=33.3),
    )
    # status == "review_plan" — display state["review_snapshot"]
    status, state = runner.step(review_input={"decision": "approve"})
    # status == "review_optimization" — display state["optimization_review_snapshot"]
    status, state = runner.step(steering_input={"decision": "continue"})
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from embodied_ai_architect.graphs.governance import GovernanceGuard, GovernancePolicy
from embodied_ai_architect.graphs.planner import PlannerNode
from embodied_ai_architect.graphs.session_store import SessionStore
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    SoCDesignState,
    create_initial_soc_state,
    get_iteration_summary,
)
from embodied_ai_architect.graphs.specialists import create_default_dispatcher

logger = logging.getLogger(__name__)


class SoCDesignRunner:
    """High-level interface for running the SoC design optimization loop.

    Wraps build_soc_design_graph with sensible defaults and provides
    run(), start(), step(), and get_state_history() methods.
    """

    def __init__(
        self,
        static_plan: Optional[list[dict[str, Any]]] = None,
        llm: Any = None,
        governance: Optional[GovernancePolicy] = None,
        experience_db_path: Optional[str] = None,
        checkpointer: Any = None,
        recursion_limit: int = 50,
        human_review: bool = False,
        optimization_review: bool = False,
        session_dir: Optional[str] = None,
    ) -> None:
        """Initialize the runner.

        Args:
            static_plan: Pre-built plan for deterministic mode.
            llm: LLM client for dynamic planning. One of static_plan/llm required.
            governance: Governance policy for iteration/budget limits.
            experience_db_path: Path to experience SQLite DB. None = default location.
            checkpointer: LangGraph checkpointer for save/resume.
            recursion_limit: LangGraph recursion limit for the graph.
            human_review: Enable plan review node (interrupt before dispatch).
            optimization_review: Enable optimization transparency in evaluate node.
            session_dir: Override session storage directory (for tests/isolation).
                None uses the default user-level session dir.
        """
        self._static_plan = static_plan
        self._llm = llm
        self._governance_policy = governance
        self._experience_db_path = experience_db_path
        self._checkpointer = checkpointer
        self._recursion_limit = recursion_limit
        self._human_review = human_review
        self._optimization_review = optimization_review
        self._compiled_graph = None
        self._state_history: list[dict[str, Any]] = []
        self._current_state: Optional[dict[str, Any]] = None
        self._config: dict[str, Any] = {}
        self._session_store = SessionStore(session_dir=session_dir)

    def _build_graph(self) -> Any:
        """Build and compile the LangGraph StateGraph."""
        from embodied_ai_architect.graphs.soc_graph import build_soc_design_graph

        # Create planner
        if self._static_plan is not None:
            planner = PlannerNode(static_plan=self._static_plan)
        elif self._llm is not None:
            planner = PlannerNode(llm=self._llm)
        else:
            raise ValueError("Either static_plan or llm must be provided")

        # Create dispatcher
        dispatcher = create_default_dispatcher()

        # Create governance
        governance = None
        if self._governance_policy is not None:
            governance = GovernanceGuard(self._governance_policy)

        # Create experience cache
        experience_cache = None
        if self._experience_db_path is not None:
            from embodied_ai_architect.graphs.experience import ExperienceCache

            experience_cache = ExperienceCache(db_path=self._experience_db_path)

        return build_soc_design_graph(
            dispatcher=dispatcher,
            planner=planner,
            governance=governance,
            experience_cache=experience_cache,
            checkpointer=self._checkpointer,
            human_review=self._human_review,
            optimization_review=self._optimization_review,
        )

    def run(
        self,
        goal: str,
        constraints: Optional[DesignConstraints] = None,
        use_case: str = "",
        platform: str = "",
        max_iterations: int = 20,
        session_id: Optional[str] = None,
        governance_dict: Optional[dict] = None,
        rtl_enabled: bool = False,
        rtl_area_feedback: bool = False,
        enable_moo: bool = True,
    ) -> SoCDesignState:
        """Run a complete SoC design session (batch mode, no review).

        Args:
            goal: Natural language design objective.
            constraints: Design constraints.
            use_case: Application type.
            platform: Platform type.
            max_iterations: Maximum optimization iterations.
            session_id: Optional session identifier.
            governance_dict: Optional governance policy dict.
            rtl_enabled: Enable KPU + floorplan + bandwidth + RTL pipeline.
            rtl_area_feedback: Enable RTL→KPU area feedback loop (issue #31).
            enable_moo: Schedule moo_explorer in default plan (default True).

        Returns:
            Final SoCDesignState after optimization completes.
        """
        state = create_initial_soc_state(
            goal=goal,
            constraints=constraints,
            use_case=use_case,
            platform=platform,
            max_iterations=max_iterations,
            session_id=session_id,
            governance=governance_dict,
            rtl_enabled=rtl_enabled,
            rtl_area_feedback=rtl_area_feedback,
            enable_moo=enable_moo,
        )

        graph = self._build_graph()
        self._compiled_graph = graph

        config = {"recursion_limit": self._recursion_limit}
        if self._checkpointer is not None:
            config["configurable"] = {"thread_id": state.get("session_id", "default")}

        logger.info("Starting SoC design session: %s", state.get("session_id"))
        result = graph.invoke(state, config=config)

        self._state_history.append(dict(result))
        self._session_store.save(result)
        logger.info("Session complete: %s", get_iteration_summary(result))

        return result

    # -------------------------------------------------------------------
    # Interactive mode: start() + step()
    # -------------------------------------------------------------------

    def start(
        self,
        goal: str,
        constraints: Optional[DesignConstraints] = None,
        use_case: str = "",
        platform: str = "",
        max_iterations: int = 20,
        session_id: Optional[str] = None,
        governance_dict: Optional[dict] = None,
    ) -> tuple[str, SoCDesignState]:
        """Start an interactive design session.

        Returns at the first interrupt point (plan_review or evaluate).

        Returns:
            (status, state) where status is one of:
            - "review_plan": paused at plan review, display state["review_snapshot"]
            - "review_optimization": paused at evaluate, display
              state["optimization_review_snapshot"]
            - "complete": design finished
            - "running": still executing (no interrupt hit)
        """
        state = create_initial_soc_state(
            goal=goal,
            constraints=constraints,
            use_case=use_case,
            platform=platform,
            max_iterations=max_iterations,
            session_id=session_id,
            governance=governance_dict,
        )
        state["human_review_enabled"] = True

        graph = self._build_graph()
        self._compiled_graph = graph

        self._config = {"recursion_limit": self._recursion_limit}
        if self._checkpointer is not None:
            self._config["configurable"] = {"thread_id": state.get("session_id", "default")}

        logger.info("Starting interactive SoC design session: %s", state.get("session_id"))

        result = graph.invoke(state, config=self._config)
        self._current_state = dict(result)
        self._session_store.save(result)

        return self._classify_state(result)

    def step(
        self,
        review_input: Optional[dict[str, Any]] = None,
        steering_input: Optional[dict[str, Any]] = None,
    ) -> tuple[str, SoCDesignState]:
        """Advance the interactive session past a review point.

        Provide review_input when status was "review_plan", or
        steering_input when status was "review_optimization".

        Args:
            review_input: PlanReviewInput dict (decision, tasks_to_add, etc.)
            steering_input: OptimizationSteeringInput dict (decision, focus, etc.)

        Returns:
            (status, state) — same as start().
        """
        if self._compiled_graph is None or self._current_state is None:
            raise RuntimeError("No active session. Call start() first.")

        state = dict(self._current_state)

        if review_input is not None:
            state["review_input"] = review_input

        if steering_input is not None:
            state["optimization_steering"] = steering_input

        result = self._compiled_graph.invoke(state, config=self._config)
        self._current_state = dict(result)
        self._session_store.save(result)

        status, _ = self._classify_state(result)

        if status == "complete":
            self._state_history.append(dict(result))
            logger.info("Session complete: %s", get_iteration_summary(result))

        return status, result

    def _classify_state(self, state: SoCDesignState) -> tuple[str, SoCDesignState]:
        """Classify the current state into a human-readable status."""
        design_status = state.get("status", "")

        if design_status in ("complete", "failed"):
            return "complete", state

        # Check if we're at plan review
        if state.get("review_snapshot") and design_status == "reviewing":
            return "review_plan", state

        # Check if we're at optimization review
        if state.get("optimization_review_snapshot") and design_status == "optimizing":
            return "review_optimization", state

        return "running", state

    def get_current_state(self) -> Optional[SoCDesignState]:
        """Get the current state of the interactive session."""
        return self._current_state

    def get_state_history(self) -> list[dict[str, Any]]:
        """Get the history of final states from all runs."""
        return list(self._state_history)
