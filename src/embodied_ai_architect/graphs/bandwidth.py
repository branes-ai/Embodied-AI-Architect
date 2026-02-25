"""Bandwidth matching analysis for KPU memory hierarchy.

Validates that data flow is balanced from DRAM through L3->L2->L1->compute.
Each interface must provide enough bandwidth to feed the next level.

Usage:
    from embodied_ai_architect.graphs.bandwidth import check_bandwidth_match

    result = check_bandwidth_match(kpu_config, workload, arithmetic_intensity=50.0)
    assert result.balanced
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class BandwidthLink(BaseModel):
    """One link in the bandwidth chain."""

    name: str
    source: str
    sink: str
    available_gbps: float
    required_gbps: float
    utilization: float
    bottleneck: bool


class BandwidthMatchResult(BaseModel):
    """Result of bandwidth matching analysis."""

    links: list[BandwidthLink] = Field(default_factory=list)
    balanced: bool = True
    bottleneck_link: Optional[str] = None
    peak_utilization: float = 0.0
    ingress_gbps: float = 0.0
    egress_gbps: float = 0.0
    compute_demand_gbps: float = 0.0
    issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Bandwidth estimation
# ---------------------------------------------------------------------------


def _compute_demand_gbps(config, workload: dict[str, Any], arithmetic_intensity: float) -> float:
    """Compute bandwidth demand from workload characteristics.

    Args:
        config: KPUMicroArchConfig.
        workload: Workload profile dict.
        arithmetic_intensity: FLOPs/byte from roofline analysis.
            Higher = compute-bound (less bandwidth needed).
            Lower = memory-bound (more bandwidth needed).
    """
    gflops = workload.get("total_estimated_gflops", workload.get("estimated_gflops", 5.0))

    if arithmetic_intensity > 0:
        # Demand = compute_throughput / arithmetic_intensity
        # peak TOPS in GFLOPS = peak_tops * 1000
        peak_gflops = config.peak_tops_int8 * 1000
        effective_gflops = min(gflops, peak_gflops)
        demand_gb_s = effective_gflops / arithmetic_intensity
    else:
        # Default: assume moderate arithmetic intensity (~10 FLOPs/byte)
        demand_gb_s = gflops / 10.0

    return demand_gb_s


def _l3_bandwidth_gbps(config) -> float:
    """Estimate L3 -> L2 bandwidth (via NoC and block movers)."""
    # Block movers transfer from L3 to L2 via NoC
    noc_bw = config.noc.link_bandwidth_gbps * min(config.noc.num_routers, 4)
    block_mover_bw = (
        config.num_memory_tiles
        * config.memory_tile.num_block_movers
        * config.memory_tile.block_mover_bw_gbps
    )
    # Effective is minimum of NoC capacity and block mover capacity
    return min(noc_bw, block_mover_bw)


def _l2_bandwidth_gbps(config) -> float:
    """Estimate L2 -> L1 bandwidth (streamers within compute tile)."""
    ct = config.compute_tile
    # Each streamer can deliver from L2 to L1
    # Assume each read port delivers at L2 bank bandwidth
    bank_bw_gbps = 16.0  # ~16 GB/s per SRAM bank at typical frequencies
    port_bw = ct.l2_read_ports * bank_bw_gbps
    streamer_bw = ct.num_streamers * bank_bw_gbps
    per_tile = min(port_bw, streamer_bw)
    return per_tile * config.num_compute_tiles


def _l1_bandwidth_gbps(config) -> float:
    """Estimate L1 -> compute bandwidth."""
    ct = config.compute_tile
    # L1 skew buffer: each bank delivers at high rate
    bank_bw_gbps = 32.0  # L1 is faster, closer to compute
    return ct.l1_num_banks * bank_bw_gbps * config.num_compute_tiles


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def check_bandwidth_match(
    config,
    workload: dict[str, Any],
    arithmetic_intensity: float = 0.0,
    bottleneck_threshold: float = 0.85,
) -> BandwidthMatchResult:
    """Check ingress/egress bandwidth matching through memory hierarchy.

    Chain:
    1. DRAM -> L3 (memory controllers)
    2. L3 -> L2 (NoC + block movers)
    3. L2 -> L1 (streamers within tile)
    4. L1 -> compute (skew buffer to systolic array)

    Args:
        config: KPUMicroArchConfig instance.
        workload: Workload profile dict.
        arithmetic_intensity: FLOPs/byte (0 = auto-estimate).
        bottleneck_threshold: Utilization above which a link is a bottleneck.

    Returns:
        BandwidthMatchResult with balance verdict.
    """
    issues = []
    links = []

    # Compute demand
    demand_gbps = _compute_demand_gbps(config, workload, arithmetic_intensity)

    # Scale demand through hierarchy (each level needs fraction of total)
    # DRAM feeds everything, so it sees full demand
    # L3->L2 sees most demand (some reuse in L3 reduces it)
    # L2->L1 sees less (L2 caching reduces demand)
    # L1->compute sees least (L1 has highest hit rate)
    l3_reuse_factor = 0.7  # 70% of DRAM demand reaches L3->L2
    l2_reuse_factor = 0.5  # 50% of L3 demand reaches L2->L1
    l1_reuse_factor = 0.3  # 30% of L2 demand reaches L1->compute

    # Link 1: DRAM -> L3
    dram_available = config.total_dram_bandwidth_gbps
    dram_required = demand_gbps
    dram_util = dram_required / dram_available if dram_available > 0 else 1.0
    dram_bottleneck = dram_util > bottleneck_threshold
    links.append(
        BandwidthLink(
            name="dram_to_l3",
            source="DRAM",
            sink="L3",
            available_gbps=round(dram_available, 1),
            required_gbps=round(dram_required, 1),
            utilization=round(dram_util, 3),
            bottleneck=dram_bottleneck,
        )
    )
    if dram_bottleneck:
        issues.append(
            f"DRAM bandwidth bottleneck: {dram_util:.0%} utilization. "
            "Consider: upgrade DRAM (LPDDR5/HBM2E) or add controllers."
        )

    # Link 2: L3 -> L2 (NoC)
    l3_available = _l3_bandwidth_gbps(config)
    l3_required = demand_gbps * l3_reuse_factor
    l3_util = l3_required / l3_available if l3_available > 0 else 1.0
    l3_bottleneck = l3_util > bottleneck_threshold
    links.append(
        BandwidthLink(
            name="l3_to_l2",
            source="L3",
            sink="L2",
            available_gbps=round(l3_available, 1),
            required_gbps=round(l3_required, 1),
            utilization=round(l3_util, 3),
            bottleneck=l3_bottleneck,
        )
    )
    if l3_bottleneck:
        issues.append(
            f"L3->L2 (NoC) bandwidth bottleneck: {l3_util:.0%} utilization. "
            "Consider: widen NoC links or add block movers."
        )

    # Link 3: L2 -> L1 (streamers)
    l2_available = _l2_bandwidth_gbps(config)
    l2_required = demand_gbps * l3_reuse_factor * l2_reuse_factor
    l2_util = l2_required / l2_available if l2_available > 0 else 1.0
    l2_bottleneck = l2_util > bottleneck_threshold
    links.append(
        BandwidthLink(
            name="l2_to_l1",
            source="L2",
            sink="L1",
            available_gbps=round(l2_available, 1),
            required_gbps=round(l2_required, 1),
            utilization=round(l2_util, 3),
            bottleneck=l2_bottleneck,
        )
    )
    if l2_bottleneck:
        issues.append(
            f"L2->L1 (streamer) bandwidth bottleneck: {l2_util:.0%} utilization. "
            "Consider: add read ports or increase L2 banks."
        )

    # Link 4: L1 -> compute
    l1_available = _l1_bandwidth_gbps(config)
    l1_required = demand_gbps * l3_reuse_factor * l2_reuse_factor * l1_reuse_factor
    l1_util = l1_required / l1_available if l1_available > 0 else 1.0
    l1_bottleneck = l1_util > bottleneck_threshold
    links.append(
        BandwidthLink(
            name="l1_to_compute",
            source="L1",
            sink="Compute",
            available_gbps=round(l1_available, 1),
            required_gbps=round(l1_required, 1),
            utilization=round(l1_util, 3),
            bottleneck=l1_bottleneck,
        )
    )
    if l1_bottleneck:
        issues.append(
            f"L1->Compute bandwidth bottleneck: {l1_util:.0%} utilization. "
            "Consider: increase L1 banks or add read ports."
        )

    # Overall verdict
    balanced = not any(link.bottleneck for link in links)
    peak_util = max(link.utilization for link in links) if links else 0.0
    bottleneck_link = None
    if not balanced:
        worst = max(links, key=lambda link: link.utilization)
        bottleneck_link = worst.name

    return BandwidthMatchResult(
        links=links,
        balanced=balanced,
        bottleneck_link=bottleneck_link,
        peak_utilization=round(peak_util, 3),
        ingress_gbps=round(dram_available, 1),
        egress_gbps=round(dram_available, 1),
        compute_demand_gbps=round(demand_gbps, 1),
        issues=issues,
    )
