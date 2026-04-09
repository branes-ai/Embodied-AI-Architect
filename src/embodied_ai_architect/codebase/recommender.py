"""Hardware target recommender from a codebase workload profile (issue #39).

Given a `workload_profile` produced by `CodebaseConverter.to_workload_profile`,
this module classifies the workload into one of five archetypes and scores
every `HardwareEntry` in the embodied-schemas registry by fit. The output is
a ranked list of recommendations with strengths/weaknesses callouts that the
CLI and chat tool render directly.

Usage:
    from embodied_ai_architect.codebase.recommender import recommend_hardware

    recs = recommend_hardware(workload_profile, top_k=5)
    for r in recs:
        print(r.name, r.fit_score, r.strengths)

The recommender does NOT require the embodied-schemas registry to be
loadable — when it isn't (e.g. in a stripped-down test env), `recommend_
hardware` returns an empty list and the caller decides what to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# Workload archetype labels — exposed so callers can introspect
WORKLOAD_ARCHETYPES = (
    "ml_heavy",
    "control_heavy",
    "signal_heavy",
    "hybrid",
    "io_heavy",
)


@dataclass
class HardwareRecommendation:
    """One hardware target ranked against the workload."""

    id: str
    name: str
    vendor: str
    fit_score: float  # 0.0–1.0; higher = better fit
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    compute_match: float = 0.0
    memory_match: float = 0.0
    power_fit: float = 0.0
    cost_score: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "vendor": self.vendor,
            "fit_score": round(self.fit_score, 3),
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
            "compute_match": round(self.compute_match, 3),
            "memory_match": round(self.memory_match, 3),
            "power_fit": round(self.power_fit, 3),
            "cost_score": round(self.cost_score, 3),
            "notes": self.notes,
        }


def classify_workload(workload_profile: dict[str, Any]) -> str:
    """Classify a workload profile into one of the five archetypes.

    The classifier looks at:
      - dominant kernel type across all sub-workloads
      - share of each kernel type
      - total GFLOPS magnitude

    Returns one of WORKLOAD_ARCHETYPES.
    """
    workloads = workload_profile.get("workloads", []) or []
    if not workloads:
        return "hybrid"

    # Tally kernel types
    type_counts: dict[str, int] = {}
    for w in workloads:
        ktype = w.get("kernel_type", "general_compute")
        type_counts[ktype] = type_counts.get(ktype, 0) + 1
    total = sum(type_counts.values())
    if total == 0:
        return "hybrid"

    shares = {k: v / total for k, v in type_counts.items()}

    # Single-archetype dominance (>= 60% of kernels)
    if shares.get("ml_inference", 0.0) >= 0.6:
        return "ml_heavy"
    if shares.get("control_loop", 0.0) >= 0.6:
        return "control_heavy"
    if shares.get("signal_processing", 0.0) >= 0.6:
        return "signal_heavy"
    if shares.get("io_bound", 0.0) >= 0.6:
        return "io_heavy"

    # No single dominant type → hybrid
    return "hybrid"


def _safe_get(obj: Any, *path: str, default: Any = None) -> Any:
    """Walk a dotted attribute path on a HardwareEntry, returning default on miss."""
    cur = obj
    for key in path:
        if cur is None:
            return default
        cur = getattr(cur, key, None)
    return cur if cur is not None else default


def _compute_score(workload_gflops: float, hw_peak_tops: float | None) -> float:
    """Score the compute match between workload and hardware (0.0–1.0).

    Higher is better. Hardware that meets the workload exactly scores 1.0;
    over-provisioned (10x+) drops to ~0.6 (wasted area/cost); under-provisioned
    drops fast.
    """
    if hw_peak_tops is None or hw_peak_tops <= 0:
        return 0.0
    if workload_gflops <= 0:
        # No workload signal — credit any hardware as "good enough"
        return 0.5
    hw_gflops = hw_peak_tops * 1000  # TOPS → GFLOPS
    ratio = hw_gflops / workload_gflops
    if ratio < 1.0:
        # Under-provisioned: linear drop from 0.5 at ratio=1 to 0 at ratio=0
        return max(0.0, 0.5 * ratio)
    if ratio <= 2.0:
        # Sweet spot: 1× to 2× margin
        return 1.0
    if ratio <= 10.0:
        # Comfortably over-provisioned: gradual decline
        return 1.0 - 0.04 * (ratio - 2.0)  # 1.0 → 0.68 across 2x..10x
    # Wildly over-provisioned (>10×): minimum 0.5
    return 0.5


def _memory_score(workload_memory_mb: float, hw_memory_gb: float | None) -> float:
    """Score the memory match. Hardware needs ≥ workload memory."""
    if hw_memory_gb is None or hw_memory_gb <= 0:
        return 0.0
    hw_memory_mb = hw_memory_gb * 1024
    if workload_memory_mb <= 0:
        return 0.5
    ratio = hw_memory_mb / workload_memory_mb
    if ratio < 1.0:
        return max(0.0, 0.5 * ratio)
    if ratio <= 4.0:
        return 1.0
    # Over-provisioned but still useful
    return max(0.5, 1.0 - 0.05 * (ratio - 4.0))


def _power_score(power_envelope_watts: float, hw_tdp_watts: float | None) -> float:
    """Score the power fit. Hardware tdp must be ≤ envelope.

    The envelope comes from the workload-side power inference (issue #38) —
    callers can pass it as `inferred_power_watts`. When unknown, we score
    purely on TDP (lower is better, capped sensibly).
    """
    if hw_tdp_watts is None or hw_tdp_watts <= 0:
        return 0.5
    if power_envelope_watts <= 0:
        # No envelope known — prefer lower TDP, but don't penalize sharply
        return max(0.0, min(1.0, 1.0 - hw_tdp_watts / 100.0))
    if hw_tdp_watts <= power_envelope_watts:
        # Within budget — score by headroom (closer to budget = better
        # utilization, but well below is also fine)
        return 1.0
    # Over budget: linear drop
    over_ratio = hw_tdp_watts / power_envelope_watts
    return max(0.0, 1.0 - 0.5 * (over_ratio - 1.0))


def _cost_score(hw_cost_usd: float | None) -> float:
    """Cost penalty: cheaper is better, capped at $1000."""
    if hw_cost_usd is None or hw_cost_usd <= 0:
        return 0.5
    if hw_cost_usd <= 50:
        return 1.0
    if hw_cost_usd >= 1000:
        return 0.2
    # Linear interpolation between $50 and $1000
    return 1.0 - 0.8 * (hw_cost_usd - 50) / 950


def _archetype_compute_paradigm_bonus(archetype: str, hw: Any) -> float:
    """Small bonus when hardware paradigm matches the archetype."""
    paradigm = _safe_get(hw, "compute_paradigm", default=None)
    paradigm_str = str(paradigm).lower() if paradigm else ""
    htype = str(_safe_get(hw, "hardware_type", default="")).lower()

    if archetype == "ml_heavy":
        if "npu" in paradigm_str or "tensor" in paradigm_str:
            return 0.15
        if "gpu" in paradigm_str or "gpu" in htype:
            return 0.10
        if "tpu" in paradigm_str or "kpu" in paradigm_str:
            return 0.10
    elif archetype == "signal_heavy":
        if "dsp" in paradigm_str or "dsp" in htype:
            return 0.15
    elif archetype == "control_heavy":
        if "cpu" in paradigm_str or "mcu" in htype or "mcu" in paradigm_str:
            return 0.10
    elif archetype == "io_heavy":
        # I/O bound — prefer hardware with good interconnect / low latency
        return 0.0
    return 0.0


def _strengths_and_weaknesses(
    archetype: str,
    workload_gflops: float,
    workload_memory_mb: float,
    power_envelope_watts: float,
    hw: Any,
) -> tuple[list[str], list[str]]:
    """Build human-readable strength and weakness callouts for one HW entry."""
    strengths: list[str] = []
    weaknesses: list[str] = []

    cap = _safe_get(hw, "capabilities", default=None)
    pwr = _safe_get(hw, "power", default=None)

    peak_tops = _safe_get(cap, "peak_tops_int8", default=None)
    memory_gb = _safe_get(cap, "memory_gb", default=None)
    tdp = _safe_get(pwr, "tdp_watts", default=None)
    cost = _safe_get(hw, "cost_usd", default=None)
    quant = _safe_get(cap, "quantization_support", default=[]) or []
    int4 = _safe_get(cap, "int4_support", default=False)
    sparse = _safe_get(cap, "sparse_acceleration", default=False)

    # Compute strengths
    if peak_tops and peak_tops * 1000 >= workload_gflops:
        strengths.append(f"{peak_tops:.1f} TOPS INT8")
    elif peak_tops:
        weaknesses.append(f"only {peak_tops:.1f} TOPS (workload needs more)")

    # Memory
    if memory_gb and memory_gb * 1024 >= workload_memory_mb:
        strengths.append(f"{memory_gb:.0f}GB memory")
    elif memory_gb:
        weaknesses.append(f"{memory_gb:.0f}GB memory (tight for workload)")

    # Power
    if tdp:
        if power_envelope_watts > 0 and tdp > power_envelope_watts:
            weaknesses.append(f"{tdp:.0f}W TDP exceeds {power_envelope_watts:.0f}W envelope")
        elif tdp <= 5:
            strengths.append(f"{tdp:.0f}W TDP")
        else:
            strengths.append(f"{tdp:.0f}W TDP")

    # Cost
    if cost:
        if cost <= 100:
            strengths.append(f"${cost:.0f}")
        elif cost >= 500:
            weaknesses.append(f"${cost:.0f} cost")

    # Quantization / accel features
    if "int8" in [q.lower() for q in quant]:
        strengths.append("INT8 quantization")
    if int4:
        strengths.append("INT4 quantization")
    if sparse:
        strengths.append("sparse acceleration")

    # Archetype-specific callouts
    if archetype == "ml_heavy":
        if not peak_tops:
            weaknesses.append("no INT8 TOPS spec — uncertain ML fit")

    return strengths, weaknesses


def _score_hardware_entry(
    hw: Any,
    workload_gflops: float,
    workload_memory_mb: float,
    power_envelope_watts: float,
    archetype: str,
) -> HardwareRecommendation:
    """Score a single HardwareEntry against the workload."""
    cap = _safe_get(hw, "capabilities", default=None)
    pwr = _safe_get(hw, "power", default=None)

    peak_tops = _safe_get(cap, "peak_tops_int8", default=None)
    memory_gb = _safe_get(cap, "memory_gb", default=None)
    tdp = _safe_get(pwr, "tdp_watts", default=None)
    cost = _safe_get(hw, "cost_usd", default=None)

    compute = _compute_score(workload_gflops, peak_tops)
    memory = _memory_score(workload_memory_mb, memory_gb)
    power = _power_score(power_envelope_watts, tdp)
    cost_s = _cost_score(cost)

    # Weighted aggregate. Compute and power are the dominant factors;
    # memory and cost are secondary tiebreakers. The archetype paradigm
    # bonus nudges the right hardware family to the top.
    base = 0.40 * compute + 0.20 * memory + 0.25 * power + 0.15 * cost_s
    bonus = _archetype_compute_paradigm_bonus(archetype, hw)
    fit = min(1.0, base + bonus)

    strengths, weaknesses = _strengths_and_weaknesses(
        archetype, workload_gflops, workload_memory_mb, power_envelope_watts, hw
    )

    return HardwareRecommendation(
        id=str(getattr(hw, "id", "unknown")),
        name=str(getattr(hw, "name", "unknown")),
        vendor=str(getattr(hw, "vendor", "unknown")),
        fit_score=fit,
        strengths=strengths,
        weaknesses=weaknesses,
        compute_match=compute,
        memory_match=memory,
        power_fit=power,
        cost_score=cost_s,
    )


def _load_hardware_entries() -> list[Any]:
    """Load all hardware entries from the embodied-schemas registry.

    Returns an empty list when the registry isn't available so the caller
    can degrade gracefully.
    """
    try:
        from embodied_schemas import Registry

        registry = Registry.load()
        return list(registry.hardware.values())
    except Exception:
        return []


def recommend_hardware(
    workload_profile: dict[str, Any],
    top_k: int = 4,
    power_envelope_watts: float = 0.0,
    candidates: Iterable[Any] | None = None,
) -> list[HardwareRecommendation]:
    """Rank hardware targets by fit against a workload profile (issue #39).

    Args:
        workload_profile: Dict produced by `CodebaseConverter.to_workload_profile`.
        top_k: Maximum number of recommendations to return.
        power_envelope_watts: Optional power budget hint (e.g. from the
            issue #38 inference layer). When 0, scoring is purely on
            workload-vs-hardware capability without a hard budget.
        candidates: Optional explicit list of `HardwareEntry`-like objects
            to score (used by tests). When None, loads from the
            embodied-schemas registry.

    Returns:
        Ranked list of `HardwareRecommendation` (highest fit_score first),
        truncated to `top_k`. Returns an empty list when no candidates are
        available (e.g. registry not installed).
    """
    workload_gflops = float(workload_profile.get("total_estimated_gflops", 0.0) or 0.0)
    workload_memory_mb = float(workload_profile.get("total_estimated_memory_mb", 0.0) or 0.0)
    archetype = classify_workload(workload_profile)

    entries = list(candidates) if candidates is not None else _load_hardware_entries()
    if not entries:
        return []

    scored = [
        _score_hardware_entry(
            hw,
            workload_gflops,
            workload_memory_mb,
            power_envelope_watts,
            archetype,
        )
        for hw in entries
    ]
    scored.sort(key=lambda r: r.fit_score, reverse=True)
    return scored[:top_k]
