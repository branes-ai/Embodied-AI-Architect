"""FastAPI server for design session data access.

Thin read-only layer over SessionStore. All endpoints are GET — the API
does not modify sessions. Design execution happens through the CLI or
SoCDesignRunner.

Usage:
    from embodied_ai_architect.api.server import create_app

    app = create_app()
    # Run with: uvicorn embodied_ai_architect.api.server:app
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from embodied_ai_architect.graphs.session_store import SessionStore

logger = logging.getLogger(__name__)

_store: Optional[SessionStore] = None


def get_store() -> SessionStore:
    """Get or create the global SessionStore."""
    global _store
    if _store is None:
        _store = SessionStore()
    return _store


def create_app(
    cors_origins: list[str] | None = None,
    session_dir: str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        cors_origins: Allowed CORS origins. Defaults to localhost dev ports.
        session_dir: Override session storage directory.

    Returns:
        Configured FastAPI app.
    """
    global _store

    app = FastAPI(
        title="Branes Architect API",
        description="Design session data for the Branes Embodied AI Architect platform",
        version="0.1.0",
    )

    # CORS
    if cors_origins is None:
        cors_origins = [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # Session store
    if session_dir:
        _store = SessionStore(session_dir=session_dir)
    else:
        _store = SessionStore()

    # Register routes
    app.include_router(_build_router())

    return app


def _build_router():
    """Build the API router with all endpoints."""
    from fastapi import APIRouter

    router = APIRouter(prefix="/api")

    @router.get("/health")
    async def health():
        """Server health check."""
        store = get_store()
        return {
            "status": "ok",
            "session_dir": str(store.session_dir),
            "session_count": len(store.list_sessions()),
        }

    @router.get("/sessions")
    async def list_sessions():
        """List all saved design sessions."""
        store = get_store()
        return store.list_sessions()

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get full session state."""
        store = get_store()
        state = store.load(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return state

    @router.get("/sessions/{session_id}/pareto")
    async def get_pareto(session_id: str):
        """Get Pareto frontier data for visualization."""
        state = _load_or_404(session_id)
        pareto_points = state.get("pareto_points", [])
        pareto_results = state.get("pareto_results", {})

        # Also extract from optimization review snapshot if available
        opt_snap = state.get("optimization_review_snapshot", {})

        return {
            "points": pareto_points,
            "front": pareto_results.get("front", []),
            "knee_point_index": pareto_results.get("knee_point_index"),
            "objectives": ["power_watts", "latency_ms", "cost_usd"],
            "pareto_front_size": opt_snap.get("pareto_front_size", len(pareto_points)),
            "hypervolume": opt_snap.get("hypervolume"),
        }

    @router.get("/sessions/{session_id}/slackness")
    async def get_slackness(session_id: str):
        """Get constraint slackness analysis."""
        state = _load_or_404(session_id)

        # Try optimization review snapshot first (pre-computed)
        opt_snap = state.get("optimization_review_snapshot", {})
        if opt_snap.get("constraint_slackness"):
            return opt_snap["constraint_slackness"]

        # Fall back to computing from state
        try:
            from embodied_ai_architect.graphs.optimization_review import (
                compute_constraint_slackness,
            )

            slackness = compute_constraint_slackness(state)
            return [cs.model_dump() for cs in slackness]
        except Exception:
            return []

    @router.get("/sessions/{session_id}/trajectory")
    async def get_trajectory(session_id: str):
        """Get optimization trajectory (PPA history across iterations)."""
        state = _load_or_404(session_id)
        return state.get("optimization_history", [])

    @router.get("/sessions/{session_id}/taskgraph")
    async def get_taskgraph(session_id: str):
        """Get task graph structure for DAG visualization."""
        state = _load_or_404(session_id)
        task_graph = state.get("task_graph", {"nodes": {}})
        nodes = task_graph.get("nodes", {})

        # Compute execution order and parallel groups
        try:
            from embodied_ai_architect.graphs.task_graph import TaskGraph
            from embodied_ai_architect.graphs.review import compute_parallel_groups

            graph = TaskGraph.from_dict(task_graph)
            execution_order = graph.execution_order()
            parallel_groups = compute_parallel_groups(graph)
        except Exception:
            execution_order = list(nodes.keys())
            parallel_groups = []

        return {
            "nodes": [
                {
                    "id": tid,
                    "name": node.get("name", ""),
                    "agent": node.get("agent", ""),
                    "status": node.get("status", "pending"),
                    "dependencies": node.get("dependencies", []),
                }
                for tid, node in nodes.items()
            ],
            "execution_order": execution_order,
            "parallel_groups": parallel_groups,
        }

    @router.get("/sessions/{session_id}/workload")
    async def get_workload(session_id: str):
        """Get per-operator workload breakdown."""
        state = _load_or_404(session_id)
        wp = state.get("workload_profile", {})

        operators = []
        for w in wp.get("workloads", []):
            operators.append(
                {
                    "name": w.get("name", "unknown"),
                    "model_class": w.get("model_class", ""),
                    "gflops": w.get("estimated_gflops"),
                    "memory_mb": w.get("estimated_memory_mb"),
                    "scheduling": w.get("scheduling", ""),
                }
            )

        return {
            "operators": operators,
            "total_gflops": wp.get("total_estimated_gflops"),
            "total_memory_mb": wp.get("total_estimated_memory_mb"),
            "dominant_op": wp.get("dominant_op", ""),
            "source": wp.get("source", ""),
        }

    return router


def _load_or_404(session_id: str) -> dict[str, Any]:
    """Load a session or raise 404."""
    store = get_store()
    state = store.load(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return state


# Module-level app for `uvicorn embodied_ai_architect.api.server:app`
app = create_app()
