"""Drift detection and monitoring.

Monitors model performance over time and detects when accuracy
degrades beyond acceptable thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from .metrics import MetricType, ValidationResult

console = Console()


class DriftStatus(str, Enum):
    """Drift detection status."""

    STABLE = "stable"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class DriftReport:
    """Report on model drift detection."""

    model_id: str
    status: DriftStatus
    metric: MetricType
    baseline_value: float
    current_value: float
    delta: float
    delta_percent: float
    threshold_warning: float = 0.05  # 5% degradation
    threshold_critical: float = 0.10  # 10% degradation
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Determine status based on delta
        if self.delta_percent <= -self.threshold_critical:
            self.status = DriftStatus.CRITICAL
            self.recommendations = [
                "Model performance has degraded significantly",
                "Consider retraining on recent data",
                "Check for data distribution shift",
                "Review input preprocessing pipeline",
            ]
        elif self.delta_percent <= -self.threshold_warning:
            self.status = DriftStatus.WARNING
            self.recommendations = [
                "Model performance is declining",
                "Monitor closely for further degradation",
                "Consider collecting more validation data",
            ]
        else:
            self.status = DriftStatus.STABLE
            self.recommendations = []

    def summary(self) -> str:
        """Generate human-readable summary."""
        status_icons = {
            DriftStatus.STABLE: "[green]✓ STABLE[/green]",
            DriftStatus.WARNING: "[yellow]⚠ WARNING[/yellow]",
            DriftStatus.CRITICAL: "[red]✗ CRITICAL[/red]",
            DriftStatus.UNKNOWN: "[dim]? UNKNOWN[/dim]",
        }

        lines = [
            f"Drift Report: {status_icons[self.status]}",
            f"Model: {self.model_id}",
            f"Metric: {self.metric.value}",
            f"Baseline: {self.baseline_value:.1%}",
            f"Current: {self.current_value:.1%}",
            f"Delta: {self.delta_percent:+.1%}",
        ]

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


class DriftMonitor:
    """Monitors model performance drift over time.

    Tracks validation results and detects when performance
    degrades beyond configured thresholds.
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        warning_threshold: float = 0.05,
        critical_threshold: float = 0.10,
    ):
        """Initialize drift monitor.

        Args:
            storage_path: Path to store validation history
            warning_threshold: Threshold for warning (5% default)
            critical_threshold: Threshold for critical (10% default)
        """
        self.storage_path = storage_path or Path.home() / ".cache" / "branes" / "drift"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def record(self, result: ValidationResult) -> None:
        """Record a validation result for drift tracking.

        Args:
            result: Validation result to record
        """
        history_file = self._get_history_file(result.model_id)

        # Load existing history
        history = self._load_history(history_file)

        # Add new result
        history.append(
            {
                "timestamp": result.timestamp or datetime.now().isoformat(),
                "dataset": result.dataset,
                "passed": result.passed,
                "metrics": {
                    m.metric.value: {
                        "value": m.value,
                        "threshold": m.threshold,
                        "passed": m.passed,
                    }
                    for m in result.metrics
                },
            }
        )

        # Save history
        self._save_history(history_file, history)

        console.print(f"[dim]Recorded validation for {result.model_id}[/dim]")

    def check_drift(
        self,
        model_id: str,
        metric: MetricType = MetricType.MAP50,
        baseline_count: int = 5,
    ) -> Optional[DriftReport]:
        """Check for performance drift.

        Compares recent performance against baseline.

        Args:
            model_id: Model to check
            metric: Metric to monitor
            baseline_count: Number of initial results to use as baseline

        Returns:
            DriftReport or None if insufficient history
        """
        history_file = self._get_history_file(model_id)
        history = self._load_history(history_file)

        if len(history) < baseline_count + 1:
            console.print(
                f"[dim]Insufficient history for drift detection "
                f"({len(history)}/{baseline_count + 1} samples)[/dim]"
            )
            return None

        # Get metric values
        metric_key = metric.value
        values = []
        for entry in history:
            metric_data = entry.get("metrics", {}).get(metric_key)
            if metric_data:
                values.append(metric_data["value"])

        if len(values) < baseline_count + 1:
            return None

        # Compute baseline (first N results)
        baseline_values = values[:baseline_count]
        baseline_avg = sum(baseline_values) / len(baseline_values)

        # Current value (most recent)
        current_value = values[-1]

        # Compute delta
        delta = current_value - baseline_avg
        delta_percent = delta / baseline_avg if baseline_avg != 0 else 0

        return DriftReport(
            model_id=model_id,
            status=DriftStatus.UNKNOWN,  # Will be set in __post_init__
            metric=metric,
            baseline_value=baseline_avg,
            current_value=current_value,
            delta=delta,
            delta_percent=delta_percent,
            threshold_warning=self.warning_threshold,
            threshold_critical=self.critical_threshold,
        )

    def get_history(
        self,
        model_id: str,
        limit: int = 10,
    ) -> list[dict]:
        """Get validation history for a model.

        Args:
            model_id: Model identifier
            limit: Maximum entries to return

        Returns:
            List of validation records
        """
        history_file = self._get_history_file(model_id)
        history = self._load_history(history_file)
        return history[-limit:]

    def show_history(self, model_id: str, limit: int = 10) -> None:
        """Display validation history table."""
        history = self.get_history(model_id, limit)

        if not history:
            console.print(f"[dim]No history for {model_id}[/dim]")
            return

        table = Table(title=f"Validation History: {model_id}")
        table.add_column("Timestamp")
        table.add_column("Dataset")
        table.add_column("Status")

        # Add metric columns from first entry
        metric_keys = list(history[0].get("metrics", {}).keys())[:4]
        for key in metric_keys:
            table.add_column(key, justify="right")

        for entry in history:
            status = "[green]✓[/green]" if entry.get("passed") else "[red]✗[/red]"
            ts = entry.get("timestamp", "")[:19]  # Truncate to datetime

            row = [ts, entry.get("dataset", ""), status]

            for key in metric_keys:
                metric_data = entry.get("metrics", {}).get(key, {})
                value = metric_data.get("value")
                if value is not None:
                    if key in ("latency_ms",):
                        row.append(f"{value:.1f}ms")
                    elif key in ("throughput_fps",):
                        row.append(f"{value:.1f}")
                    else:
                        row.append(f"{value:.1%}")
                else:
                    row.append("N/A")

            table.add_row(*row)

        console.print(table)

    def clear_history(self, model_id: str) -> None:
        """Clear validation history for a model."""
        history_file = self._get_history_file(model_id)
        if history_file.exists():
            history_file.unlink()
            console.print(f"[dim]Cleared history for {model_id}[/dim]")

    def list_models(self) -> list[str]:
        """List models with validation history."""
        models = []
        for f in self.storage_path.glob("*.json"):
            models.append(f.stem)
        return sorted(models)

    def _get_history_file(self, model_id: str) -> Path:
        """Get history file path for a model."""
        safe_id = model_id.replace("/", "_").replace(":", "_")
        return self.storage_path / f"{safe_id}.json"

    def _load_history(self, path: Path) -> list[dict]:
        """Load history from file."""
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return []

    def _save_history(self, path: Path, history: list[dict]) -> None:
        """Save history to file."""
        with open(path, "w") as f:
            json.dump(history, f, indent=2)


# Module-level monitor instance
_monitor: Optional[DriftMonitor] = None


def get_monitor() -> DriftMonitor:
    """Get the global drift monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = DriftMonitor()
    return _monitor


def record_validation(result: ValidationResult) -> None:
    """Record a validation result for drift tracking."""
    get_monitor().record(result)


def check_drift(
    model_id: str,
    metric: MetricType = MetricType.MAP50,
) -> Optional[DriftReport]:
    """Check for performance drift."""
    return get_monitor().check_drift(model_id, metric)
