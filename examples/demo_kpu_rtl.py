#!/usr/bin/env python3
"""Demo 4: KPU Micro-architecture Configuration + RTL Generation.

Runs the full four-level flow:
  Outer Loop → KPU Config → Floorplan Check → Bandwidth Check → RTL Generation

Shows:
1. KPU micro-architecture sizing for a delivery drone perception workload
2. Checkerboard floorplan estimation with pitch matching
3. Bandwidth matching validation through memory hierarchy
4. Template-based RTL generation with EDA lint + synthesis

Usage:
    python examples/demo_kpu_rtl.py
"""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    create_initial_soc_state,
    get_task_graph,
)
from embodied_ai_architect.graphs.planner import PlannerNode
from embodied_ai_architect.graphs.session_store import SessionStore
from embodied_ai_architect.graphs.soc_runner import SoCDesignRunner

# ---------------------------------------------------------------------------
# Demo config
# ---------------------------------------------------------------------------

DEMO_4_GOAL = (
    "Design a KPU-based SoC for a delivery drone perception pipeline that must: "
    "run camera vision processing through a dedicated ISP, "
    "object detection and multi-object tracking at 30fps, "
    "with full RTL generation for silicon fabrication. "
    "Configure the KPU micro-architecture (systolic array, memory tiles, NoC), "
    "validate checkerboard floorplan pitch matching and bandwidth hierarchy, "
    "then generate synthesizable RTL for all sub-components targeting a 12nm process."
)

DEMO_4_CONSTRAINTS = DesignConstraints(
    max_power_watts=5.0,
    max_latency_ms=33.3,
    max_cost_usd=30.0,
    max_area_mm2=100.0,
    target_volume=100_000,
)

DEMO_4_PLAN = [
    {
        "id": "t1",
        "name": "Analyze perception workload (detection + tracking at 30fps)",
        "agent": "workload_analyzer",
        "dependencies": [],
    },
    {
        "id": "t2",
        "name": "Enumerate feasible hardware under constraints",
        "agent": "hw_explorer",
        "dependencies": ["t1"],
    },
    {
        "id": "t3",
        "name": "Compose SoC architecture with KPU compute engine",
        "agent": "architecture_composer",
        "dependencies": ["t2"],
    },
    {
        "id": "t4",
        "name": "Configure KPU micro-architecture for drone workload",
        "agent": "kpu_configurator",
        "dependencies": ["t3"],
    },
    {
        "id": "t5",
        "name": "Validate floorplan (checkerboard pitch matching)",
        "agent": "floorplan_validator",
        "dependencies": ["t4"],
    },
    {
        "id": "t6",
        "name": "Validate bandwidth matching through memory hierarchy",
        "agent": "bandwidth_validator",
        "dependencies": ["t4"],
    },
    {
        "id": "t7",
        "name": "Generate RTL for KPU sub-components",
        "agent": "rtl_generator",
        "dependencies": ["t5", "t6"],
    },
    {
        "id": "t8",
        "name": "Assess PPA from RTL synthesis results",
        "agent": "rtl_ppa_assessor",
        "dependencies": ["t7"],
    },
    {
        "id": "t9",
        "name": "Review design for risks and improvements",
        "agent": "critic",
        "dependencies": ["t8"],
    },
    {
        "id": "t10",
        "name": "Generate final design report",
        "agent": "report_generator",
        "dependencies": ["t9"],
    },
]

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

W = 72


def banner(text: str) -> None:
    print(f"\n{'=' * W}")
    print(f"  {text}")
    print(f"{'=' * W}")


def section(text: str) -> None:
    print(f"\n--- {text} {'-' * max(0, W - len(text) - 5)}")


def kv(key: str, value: str, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{key:.<35s} {value}")


def verdict_str(v: str) -> str:
    if v == "PASS":
        return "PASS"
    elif v == "FAIL":
        return "** FAIL **"
    return v


# ---------------------------------------------------------------------------
# Main demo flow
# ---------------------------------------------------------------------------


def run_demo() -> None:
    banner("Demo 4: KPU Micro-architecture + RTL Generation")
    print("\n  Goal:")
    print(textwrap.fill(DEMO_4_GOAL, width=W - 4, initial_indent="    ", subsequent_indent="    "))
    print("  Mode: Static Plan via SoCDesignRunner (issue #35)")

    section("Constraints")
    cd = DEMO_4_CONSTRAINTS.model_dump(exclude_none=True)
    for k, v in cd.items():
        if k == "custom" and not v:
            continue
        kv(k.replace("_", " ").title(), str(v))

    # ---- Show the static plan (purely informational) ----
    section("Plan Preview")
    planner = PlannerNode(static_plan=DEMO_4_PLAN)
    preview_state = create_initial_soc_state(
        goal=DEMO_4_GOAL,
        constraints=DEMO_4_CONSTRAINTS,
        use_case="delivery_drone",
        platform="drone",
        rtl_enabled=True,
    )
    preview_state = {**preview_state, **planner(preview_state)}
    graph = get_task_graph(preview_state)
    print(f"\n  Task graph ({len(graph.nodes)} tasks):\n")
    for tid in graph.execution_order():
        task = graph.nodes[tid]
        deps = ", ".join(task.dependencies) if task.dependencies else "(none)"
        print(f"    [{tid}] {task.name}")
        print(f"         agent: {task.agent}  deps: {deps}")

    # ---- Run via SoCDesignRunner so the session is persisted ----
    section("Executing Pipeline (SoCDesignRunner)")
    t0 = time.perf_counter()

    runner = SoCDesignRunner(static_plan=DEMO_4_PLAN)

    # SoCDesignRunner.run() doesn't yet plumb rtl_enabled through its
    # signature (would be a small follow-up); for now, we build the initial
    # state ourselves and invoke the compiled graph directly. This still
    # writes the session via the runner's SessionStore.
    initial_state = create_initial_soc_state(
        goal=DEMO_4_GOAL,
        constraints=DEMO_4_CONSTRAINTS,
        use_case="delivery_drone",
        platform="drone",
        rtl_enabled=True,
    )
    runner._compiled_graph = runner._build_graph()
    state = runner._compiled_graph.invoke(
        initial_state, config={"recursion_limit": runner._recursion_limit}
    )
    runner._session_store.save(state)

    exec_time = time.perf_counter() - t0
    print(f"\n  Pipeline completed in {state.get('iteration', 0)} iterations, " f"{exec_time:.2f}s")
    print(f"  Session saved: {state.get('session_id')}")
    print(f"  Inspect with: branes session show {state.get('session_id')}")
    print("  Or:           branes session show --latest")

    # ---- KPU Results ----
    _print_kpu_results(state)

    # ---- PPA Summary ----
    _print_ppa_summary(state)

    # ---- Verify session round-trips ----
    section("Session Persistence Verification")
    store = SessionStore()
    loaded = store.load(state.get("session_id", ""))
    if loaded is None:
        print("  ! Session not found in store")
    else:
        kv("kpu_config present", str(bool(loaded.get("kpu_config"))))
        kv("floorplan present", str(bool(loaded.get("floorplan_estimate"))))
        kv("bandwidth present", str(bool(loaded.get("bandwidth_match"))))
        kv("RTL synth present", str(bool(loaded.get("rtl_synthesis_results"))))
        kv("KPU history len", str(len(loaded.get("kpu_optimization_history", []))))

    banner("Demo 4 Complete")
    print(f"\n  Total time: {exec_time:.2f}s  |  Status: {state.get('status')}")
    print(f"  RTL enabled: {state.get('rtl_enabled', False)}")
    print()


def _print_kpu_results(state: dict) -> None:
    """Print KPU-specific results."""
    kpu = state.get("kpu_config", {})
    if kpu:
        section("KPU Configuration")
        ct = kpu.get("compute_tile", {})
        mt = kpu.get("memory_tile", {})
        dram = kpu.get("dram", {})
        noc = kpu.get("noc", {})
        kv("Process Node", f"{kpu.get('process_nm', '?')}nm")
        kv("Checkerboard", f"{kpu.get('array_rows', '?')}x{kpu.get('array_cols', '?')}")
        kv("Systolic Array", f"{ct.get('array_rows', '?')}x{ct.get('array_cols', '?')}")
        kv("L2/tile", f"{ct.get('l2_size_bytes', 0) // 1024}KB")
        kv("L1/tile", f"{ct.get('l1_size_bytes', 0) // 1024}KB")
        kv("Streamers/tile", str(ct.get("num_streamers", "?")))
        kv("L3/mem tile", f"{mt.get('l3_tile_size_bytes', 0) // 1024}KB")
        kv("Block Movers/mem tile", str(mt.get("num_block_movers", "?")))
        kv("DRAM", f"{dram.get('technology', '?')} {dram.get('num_controllers', '?')} ctrl")
        kv("NoC", f"{noc.get('topology', '?')} {noc.get('link_width_bits', '?')}-bit")

    fp = state.get("floorplan_estimate", {})
    if fp:
        section("Floorplan Check")
        ct = fp.get("compute_tile", {})
        mt = fp.get("memory_tile", {})
        kv(
            "Compute Tile",
            f"{ct.get('width_mm', 0):.2f} x {ct.get('height_mm', 0):.2f}mm "
            f"= {ct.get('area_mm2', 0):.2f}mm2",
        )
        kv(
            "Memory Tile",
            f"{mt.get('width_mm', 0):.2f} x {mt.get('height_mm', 0):.2f}mm "
            f"= {mt.get('area_mm2', 0):.2f}mm2",
        )
        kv(
            "Pitch Match W",
            f"{fp.get('pitch_ratio_width', 0):.2f} "
            f"({'OK' if abs(fp.get('pitch_ratio_width', 1) - 1) < 0.15 else 'MISMATCH'})",
        )
        kv(
            "Pitch Match H",
            f"{fp.get('pitch_ratio_height', 0):.2f} "
            f"({'OK' if abs(fp.get('pitch_ratio_height', 1) - 1) < 0.15 else 'MISMATCH'})",
        )
        kv("Total Die Area", f"{fp.get('total_area_mm2', 0):.1f}mm2")
        kv("Feasible", verdict_str("PASS" if fp.get("feasible") else "FAIL"))

    bw = state.get("bandwidth_match", {})
    if bw:
        section("Bandwidth Check")
        for link in bw.get("links", []):
            name = link.get("name", "?")
            avail = link.get("available_gbps", 0)
            req = link.get("required_gbps", 0)
            util = link.get("utilization", 0)
            bottleneck = " ** BOTTLENECK" if link.get("bottleneck") else ""
            kv(name, f"{avail:.1f} avail, {req:.1f} req ({util:.0%}){bottleneck}")
        kv("Balanced", verdict_str("PASS" if bw.get("balanced") else "FAIL"))

    rtl_synth = state.get("rtl_synthesis_results", {})
    if rtl_synth:
        section("RTL Generation")
        for name, result in rtl_synth.items():
            cells = result.get("area_cells", 0)
            ok = "PASS" if result.get("success") else "FAIL"
            kv(name, f"{cells:>6} cells ({ok})")
        total_cells = sum(r.get("area_cells", 0) for r in rtl_synth.values() if r.get("success"))
        print(f"\n  Total cells: {total_cells}")


def _print_ppa_summary(state: dict) -> None:
    """Print PPA summary."""
    ppa = state.get("ppa_metrics", {})
    if ppa:
        section("PPA Summary")
        kv("Process Node", f"{ppa.get('process_nm', 28)}nm")
        if ppa.get("power_watts") is not None:
            kv("Power", f"{ppa['power_watts']:.1f}W")
        if ppa.get("area_mm2") is not None:
            kv("Area", f"{ppa['area_mm2']:.1f}mm2")
        if ppa.get("latency_ms") is not None:
            kv("Latency", f"{ppa['latency_ms']:.1f}ms")
        if ppa.get("cost_usd") is not None:
            kv("Cost", f"${ppa['cost_usd']:.0f}")

        cost_bd = ppa.get("cost_breakdown")
        if cost_bd:
            print()
            kv("Die Cost", f"${cost_bd.get('die_cost_usd', 0):.2f}")
            kv("Package", f"${cost_bd.get('package_cost_usd', 0):.2f}")
            kv("Test", f"${cost_bd.get('test_cost_usd', 0):.2f}")
            kv("NRE/unit", f"${cost_bd.get('nre_per_unit_usd', 0):.2f}")
            kv("Yield", f"{cost_bd.get('yield_percent', 0):.1f}%")
            kv("Dies/wafer", str(cost_bd.get("dies_per_wafer", 0)))

        verdicts = ppa.get("verdicts", {})
        if verdicts:
            print()
            for k, v in verdicts.items():
                kv(f"  {k}", verdict_str(v))


if __name__ == "__main__":
    run_demo()
