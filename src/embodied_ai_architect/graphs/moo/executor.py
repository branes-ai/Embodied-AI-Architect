"""Concurrent evaluation backends for the MOO engine.

Provides a protocol for evaluation backends and a local thread-pool
implementation for analytical model evaluation.

Usage:
    from embodied_ai_architect.graphs.moo.executor import LocalThreadExecutor

    executor = LocalThreadExecutor(evaluator, max_workers=8)
    results = executor.submit_batch(param_list)
    executor.shutdown()
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Protocol, runtime_checkable

from embodied_ai_architect.graphs.moo.evaluator import DesignEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


@runtime_checkable
class EvalBackend(Protocol):
    """Protocol for evaluation backends."""

    def submit_batch(self, param_list: list[dict[str, Any]]) -> list[EvaluationResult]: ...

    def shutdown(self) -> None: ...


class LocalThreadExecutor:
    """Thread-pool executor for analytical model evaluations.

    Uses ThreadPoolExecutor since evaluations are CPU-bound but release
    the GIL during numpy operations.
    """

    def __init__(
        self,
        evaluator: DesignEvaluator,
        max_workers: int = 8,
    ):
        self.evaluator = evaluator
        self.max_workers = max_workers
        self._pool: ThreadPoolExecutor | None = None

    def _get_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._pool

    def submit_batch(self, param_list: list[dict[str, Any]]) -> list[EvaluationResult]:
        """Evaluate a batch of designs concurrently.

        Results are returned in the same order as the input.
        """
        if not param_list:
            return []

        # For very small batches, evaluate sequentially
        if len(param_list) <= 2:
            return [self.evaluator.evaluate(p) for p in param_list]

        pool = self._get_pool()
        future_to_idx = {}
        for i, params in enumerate(param_list):
            future = pool.submit(self.evaluator.evaluate, params)
            future_to_idx[future] = i

        results: list[EvaluationResult | None] = [None] * len(param_list)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning("Evaluation %d failed: %s", idx, e)
                results[idx] = EvaluationResult(
                    design_params=param_list[idx],
                    objectives={
                        "power_watts": 1e6,
                        "latency_ms": 1e6,
                        "area_mm2": 1e6,
                        "cost_usd": 1e6,
                    },
                    feasible=False,
                    metadata={"error": str(e)},
                )

        return results  # type: ignore[return-value]

    def shutdown(self) -> None:
        """Shut down the thread pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
