"""CLI client for the graphs MCP server.

Provides human-friendly access to the quantitative estimators in the
graphs repo without needing to construct JSON-RPC requests.

Usage:
    branes mcp tools                           # List available tools
    branes mcp hardware                        # List all hardware targets
    branes mcp hardware --type gpu             # Filter by device type
    branes mcp hardware --query orin           # Fuzzy search
    branes mcp specs jetson_orin_nano          # Detailed hardware specs
    branes mcp analyze resnet18 jetson_orin_nano
    branes mcp analyze yolov8n h100_sxm5 --precision int8
    branes mcp latency resnet50 jetson_orin_nano --thermal 15W
    branes mcp energy resnet18 h100_sxm5 --power-gating
    branes mcp memory yolov8n jetson_orin_nano
    branes mcp compare resnet18 jetson_orin_nano h100_sxm5 coral_edge_tpu
    branes mcp server                          # Start MCP server (stdio)
    branes mcp server --sse --port 8100        # Start MCP server (HTTP)
"""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _get_server():
    """Import and return the graphs MCP server module."""
    try:
        from graphs.mcp.server import execute_mcp_tool, get_mcp_tool_definitions

        return execute_mcp_tool, get_mcp_tool_definitions
    except ImportError:
        console.print(
            "[red]Error:[/red] graphs package not found.\n"
            "Install it or set PYTHONPATH to the graphs/src directory:\n"
            "  export PYTHONPATH=/path/to/graphs/src"
        )
        sys.exit(1)


def _call(tool_name: str, args: dict, json_output: bool = False) -> dict:
    """Call an MCP tool and return parsed result."""
    execute_mcp_tool, _ = _get_server()
    result_json = execute_mcp_tool(tool_name, args)
    data = json.loads(result_json)
    if json_output:
        console.print_json(result_json)
        sys.exit(0)
    if "error" in data:
        console.print(f"[red]Error:[/red] {data['error']}")
        sys.exit(1)
    return data


# ---------------------------------------------------------------------------
# Command group
# ---------------------------------------------------------------------------


@click.group()
@click.pass_context
def mcp(ctx):
    """Graphs quantitative estimators — roofline, energy, memory analysis.

    \b
    Client for the graphs MCP server.  Calls kernel-level performance
    estimators from the graphs repo directly.

    \b
    Quick start:
      branes mcp hardware                          # browse 50+ targets
      branes mcp analyze resnet18 jetson_orin_nano  # full analysis
      branes mcp compare yolov8n jetson_orin_nano h100_sxm5
    """
    pass


# ---------------------------------------------------------------------------
# tools — list available MCP tools
# ---------------------------------------------------------------------------


@mcp.command("tools")
@click.pass_context
def list_tools(ctx):
    """List available MCP tools and their descriptions."""
    _, get_defs = _get_server()
    tools = get_defs()

    table = Table(title="Graphs MCP Tools", show_lines=True)
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Required Args", style="yellow")

    for t in tools:
        required = t["input_schema"].get("required", [])
        table.add_row(t["name"], t["description"], ", ".join(required))

    console.print(table)


# ---------------------------------------------------------------------------
# hardware — list / search hardware catalog
# ---------------------------------------------------------------------------


@mcp.command("hardware")
@click.option(
    "--type",
    "device_type",
    type=str,
    default=None,
    help="Filter: cpu, gpu, dsp, tpu, kpu, accelerator",
)
@click.option("--query", "-q", type=str, default=None, help="Fuzzy search (e.g. 'orin', 'jetson')")
@click.pass_context
def list_hardware(ctx, device_type, query):
    """List available hardware targets."""
    json_output = ctx.obj.get("json", False)
    args = {}
    if device_type:
        args["device_type"] = device_type
    if query:
        args["query"] = query

    data = _call("list_hardware", args, json_output)

    table = Table(title=f"Hardware Targets ({len(data)} found)")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Vendor")
    table.add_column("Model")
    table.add_column("Type", style="yellow")
    table.add_column("TDP (W)", justify="right")
    table.add_column("Memory (GB)", justify="right")
    table.add_column("Cal.", justify="center")

    for item in data:
        tdp = str(item.get("tdp_watts") or "-")
        mem = str(item.get("memory_gb") or "-")
        cal = "[green]yes[/green]" if item.get("calibrated") else "[dim]no[/dim]"
        table.add_row(
            item["id"],
            item.get("vendor", ""),
            item.get("model", ""),
            item.get("device_type", ""),
            tdp,
            mem,
            cal,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# specs — detailed hardware profile
# ---------------------------------------------------------------------------


@mcp.command("specs")
@click.argument("hardware_id")
@click.pass_context
def get_specs(ctx, hardware_id):
    """Show detailed specifications for a hardware target."""
    json_output = ctx.obj.get("json", False)
    data = _call("get_hardware_specs", {"hardware_id": hardware_id}, json_output)

    console.print(
        Panel(f"[bold cyan]{data['vendor']} {data['model']}[/bold cyan]", subtitle=data["id"])
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Architecture", str(data.get("architecture") or "-"))
    table.add_row("Device Type", data.get("device_type", "-"))
    table.add_row("Category", data.get("product_category") or "-")
    table.add_row("Compute Units", str(data.get("compute_units") or "-"))
    table.add_row("Memory", f"{data.get('memory_gb') or '-'} GB")
    table.add_row("TDP", f"{data.get('tdp_watts') or '-'} W")
    table.add_row("Bandwidth", f"{data.get('peak_bandwidth_gbps') or '-'} GB/s")
    table.add_row("Base Clock", f"{data.get('base_clock_mhz') or '-'} MHz")
    table.add_row("Boost Clock", f"{data.get('boost_clock_mhz') or '-'} MHz")
    table.add_row("Effective Peak", f"{data.get('effective_peak_gflops', 0):.1f} GFLOPS")
    table.add_row("Effective BW", f"{data.get('effective_bandwidth_gbps', 0):.1f} GB/s")
    table.add_row("Calibrated", "[green]yes[/green]" if data.get("calibrated") else "[dim]no[/dim]")

    console.print(table)

    peaks = data.get("theoretical_peaks_gflops", {})
    if peaks:
        console.print()
        ptable = Table(title="Theoretical Peaks (GFLOPS)")
        ptable.add_column("Precision", style="cyan")
        ptable.add_column("GFLOPS", justify="right")
        for prec, val in sorted(peaks.items()):
            ptable.add_row(prec, f"{val:,.1f}")
        console.print(ptable)

    power_profiles = data.get("power_profiles")
    if power_profiles:
        console.print()
        pptable = Table(title="Power Profiles")
        pptable.add_column("Mode", style="cyan")
        pptable.add_column("Details")
        if isinstance(power_profiles, dict):
            for mode, details in power_profiles.items():
                pptable.add_row(mode, json.dumps(details, indent=2))
        elif isinstance(power_profiles, list):
            for i, entry in enumerate(power_profiles):
                label = (
                    entry.get("name", entry.get("mode", str(i)))
                    if isinstance(entry, dict)
                    else str(i)
                )
                pptable.add_row(label, json.dumps(entry, indent=2))
        console.print(pptable)

    if data.get("calibration"):
        console.print()
        cal = data["calibration"]
        console.print(
            Panel(
                f"Date: {cal.get('date', '-')}\n"
                f"Measured GFLOPS: {cal.get('best_measured_gflops', 0):.1f}\n"
                f"Measured BW: {cal.get('measured_bandwidth_gbps', 0):.1f} GB/s",
                title="Calibration Data",
            )
        )


# ---------------------------------------------------------------------------
# analyze — full unified analysis
# ---------------------------------------------------------------------------


@mcp.command("analyze")
@click.argument("model_name", required=False, default=None)
@click.argument("hardware_name", required=False, default=None)
@click.option("--mission", "mission_id", default=None, help="Load model/hardware from a mission")
@click.option("--batch", type=int, default=1, show_default=True, help="Batch size")
@click.option(
    "--precision",
    "-p",
    type=click.Choice(["fp32", "fp16", "bf16", "int8", "int4"]),
    default="fp16",
    show_default=True,
)
@click.option("--thermal", type=str, default=None, help="Thermal profile (e.g. '15W')")
@click.pass_context
def analyze_model(ctx, model_name, hardware_name, mission_id, batch, precision, thermal):
    """Run full roofline + energy + memory analysis."""
    if mission_id:
        from embodied_ai_architect.mission.store import MissionStore

        store = MissionStore()
        mission = store.load(mission_id)
        if not mission:
            console.print(f"[red]Mission '{mission_id}' not found.[/red]")
            ctx.exit(1)
            return
        if not model_name:
            models = getattr(mission, "selected_models", None) or []
            if models:
                model_name = models[0]
        if not hardware_name:
            compute = getattr(mission, "selected_compute", None) or []
            if compute:
                hardware_name = compute[0]

    if not model_name or not hardware_name:
        console.print("[red]model_name and hardware_name are required (or use --mission).[/red]")
        ctx.exit(1)
        return

    json_output = ctx.obj.get("json", False)
    args = {
        "model_name": model_name,
        "hardware_name": hardware_name,
        "batch_size": batch,
        "precision": precision,
    }
    if thermal:
        args["thermal_profile"] = thermal

    data = _call("analyze_model", args, json_output)

    console.print(
        Panel(
            f"[bold]{model_name}[/bold] on [cyan]{hardware_name}[/cyan]  "
            f"batch={batch}  precision={precision}",
            title="Unified Analysis",
        )
    )

    # Top-level metrics
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    for key, label, unit in [
        ("total_latency_ms", "Latency", "ms"),
        ("throughput_fps", "Throughput", "FPS"),
        ("total_energy_mj", "Energy", "mJ"),
        ("peak_memory_mb", "Peak Memory", "MB"),
        ("average_utilization_pct", "Utilization", "%"),
    ]:
        val = data.get(key)
        if val is not None:
            table.add_row(label, f"{val:,.2f} {unit}")
    console.print(table)

    # Roofline breakdown
    roofline = data.get("roofline")
    if roofline:
        console.print()
        rt = Table(title="Roofline Breakdown")
        rt.add_column("Metric", style="bold")
        rt.add_column("Value", justify="right")
        rt.add_row("Compute Time", f"{roofline.get('compute_time_ms', 0):.3f} ms")
        rt.add_row("Memory Time", f"{roofline.get('memory_time_ms', 0):.3f} ms")
        rt.add_row("FLOPS Utilization", f"{roofline.get('avg_flops_utilization', 0):.1%}")
        rt.add_row("BW Utilization", f"{roofline.get('avg_bandwidth_utilization', 0):.1%}")
        rt.add_row("Compute-bound ops", str(roofline.get("num_compute_bound", 0)))
        rt.add_row("Memory-bound ops", str(roofline.get("num_memory_bound", 0)))
        console.print(rt)

    # Energy breakdown
    energy = data.get("energy")
    if energy:
        console.print()
        et = Table(title="Energy Breakdown")
        et.add_column("Component", style="bold")
        et.add_column("Value", justify="right")
        et.add_row("Total Energy", f"{energy.get('total_energy_mj', 0):.3f} mJ")
        et.add_row("Compute", f"{energy.get('compute_energy_j', 0) * 1000:.3f} mJ")
        et.add_row("Memory", f"{energy.get('memory_energy_j', 0) * 1000:.3f} mJ")
        et.add_row("Static", f"{energy.get('static_energy_j', 0) * 1000:.3f} mJ")
        et.add_row("Average Power", f"{energy.get('average_power_w', 0):.2f} W")
        et.add_row("Efficiency", f"{energy.get('efficiency', 0):.1%}")
        console.print(et)

    # Memory summary
    memory = data.get("memory")
    if memory:
        console.print()
        mt = Table(title="Memory Summary")
        mt.add_column("Metric", style="bold")
        mt.add_column("Value", justify="right")
        mt.add_row("Peak Memory", f"{memory.get('peak_memory_mb', 0):.2f} MB")
        mt.add_row("Activations", f"{memory.get('activation_memory_bytes', 0) / 1e6:.2f} MB")
        mt.add_row("Weights", f"{memory.get('weight_memory_bytes', 0) / 1e6:.2f} MB")
        mt.add_row("Workspace", f"{memory.get('workspace_memory_bytes', 0) / 1e6:.2f} MB")
        fits = "[green]yes[/green]" if memory.get("fits_on_device") else "[red]no[/red]"
        mt.add_row("Fits on Device", fits)
        console.print(mt)


# ---------------------------------------------------------------------------
# latency — roofline-only analysis
# ---------------------------------------------------------------------------


@mcp.command("latency")
@click.argument("model_name", required=False, default=None)
@click.argument("hardware_name", required=False, default=None)
@click.option("--mission", "mission_id", default=None, help="Load model/hardware from a mission")
@click.option("--batch", type=int, default=1, show_default=True)
@click.option(
    "--precision",
    "-p",
    type=click.Choice(["fp32", "fp16", "bf16", "int8", "int4"]),
    default="fp16",
    show_default=True,
)
@click.option("--thermal", type=str, default=None, help="Thermal profile (e.g. '15W')")
@click.pass_context
def estimate_latency(ctx, model_name, hardware_name, mission_id, batch, precision, thermal):
    """Predict inference latency using roofline model."""
    if mission_id:
        from embodied_ai_architect.mission.store import MissionStore

        store = MissionStore()
        mission = store.load(mission_id)
        if not mission:
            console.print(f"[red]Mission '{mission_id}' not found.[/red]")
            ctx.exit(1)
            return
        if not model_name:
            models = getattr(mission, "selected_models", None) or []
            if models:
                model_name = models[0]
        if not hardware_name:
            compute = getattr(mission, "selected_compute", None) or []
            if compute:
                hardware_name = compute[0]

    if not model_name or not hardware_name:
        console.print("[red]model_name and hardware_name are required (or use --mission).[/red]")
        ctx.exit(1)
        return

    json_output = ctx.obj.get("json", False)
    args = {
        "model_name": model_name,
        "hardware_name": hardware_name,
        "batch_size": batch,
        "precision": precision,
    }
    if thermal:
        args["thermal_profile"] = thermal

    data = _call("estimate_latency", args, json_output)

    console.print(
        Panel(
            f"Roofline latency: [bold]{model_name}[/bold] on [cyan]{hardware_name}[/cyan]",
            title="Latency Estimate",
        )
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total Latency", f"{data.get('total_latency', 0) * 1000:.3f} ms")
    table.add_row("Compute Time", f"{data.get('total_compute_time', 0) * 1000:.3f} ms")
    table.add_row("Memory Time", f"{data.get('total_memory_time', 0) * 1000:.3f} ms")
    table.add_row("FLOPS Util", f"{data.get('average_flops_utilization', 0):.1%}")
    table.add_row("BW Util", f"{data.get('average_bandwidth_utilization', 0):.1%}")
    table.add_row("Compute-bound", str(data.get("num_compute_bound", 0)))
    table.add_row("Memory-bound", str(data.get("num_memory_bound", 0)))
    console.print(table)

    # Top subgraphs
    latencies = data.get("latencies", [])
    if latencies:
        console.print()
        lt = Table(title=f"Top {len(latencies)} Subgraphs by Latency")
        lt.add_column("Subgraph", style="cyan")
        lt.add_column("Latency (ms)", justify="right")
        lt.add_column("Bottleneck", style="yellow")
        lt.add_column("Util", justify="right")
        for sg in latencies:
            lt.add_row(
                sg.get("subgraph_name", sg.get("subgraph_id", "?")),
                f"{sg.get('actual_latency', 0) * 1000:.3f}",
                sg.get("bottleneck", "-"),
                f"{sg.get('flops_utilization', 0):.1%}",
            )
        console.print(lt)


# ---------------------------------------------------------------------------
# energy — energy-only analysis
# ---------------------------------------------------------------------------


@mcp.command("energy")
@click.argument("model_name")
@click.argument("hardware_name")
@click.option("--batch", type=int, default=1, show_default=True)
@click.option(
    "--precision",
    "-p",
    type=click.Choice(["fp32", "fp16", "bf16", "int8", "int4"]),
    default="fp16",
    show_default=True,
)
@click.option("--power-gating", is_flag=True, help="Enable power gating")
@click.option("--thermal", type=str, default=None, help="Thermal profile")
@click.pass_context
def estimate_energy(ctx, model_name, hardware_name, batch, precision, power_gating, thermal):
    """Estimate energy consumption with component breakdown."""
    json_output = ctx.obj.get("json", False)
    args = {
        "model_name": model_name,
        "hardware_name": hardware_name,
        "batch_size": batch,
        "precision": precision,
        "power_gating_enabled": power_gating,
    }
    if thermal:
        args["thermal_profile"] = thermal

    data = _call("estimate_energy", args, json_output)

    console.print(
        Panel(
            f"Energy estimate: [bold]{model_name}[/bold] on [cyan]{hardware_name}[/cyan]",
            title="Energy Analysis",
        )
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total Energy", f"{data.get('total_energy_mj', 0):.3f} mJ")
    table.add_row("Compute Energy", f"{data.get('compute_energy_j', 0) * 1000:.3f} mJ")
    table.add_row("Memory Energy", f"{data.get('memory_energy_j', 0) * 1000:.3f} mJ")
    table.add_row("Static Energy", f"{data.get('static_energy_j', 0) * 1000:.3f} mJ")
    table.add_row("Avg Power", f"{data.get('average_power_w', 0):.2f} W")
    table.add_row("Efficiency", f"{data.get('average_efficiency', 0):.1%}")
    table.add_row("Wasted", f"{data.get('wasted_energy_percent', 0):.1f}%")
    console.print(table)


# ---------------------------------------------------------------------------
# memory — memory-only analysis
# ---------------------------------------------------------------------------


@mcp.command("memory")
@click.argument("model_name")
@click.argument("hardware_name")
@click.option("--batch", type=int, default=1, show_default=True)
@click.option(
    "--precision",
    "-p",
    type=click.Choice(["fp32", "fp16", "bf16", "int8", "int4"]),
    default="fp16",
    show_default=True,
)
@click.pass_context
def estimate_memory(ctx, model_name, hardware_name, batch, precision):
    """Analyse peak memory usage and activation timeline."""
    json_output = ctx.obj.get("json", False)
    args = {
        "model_name": model_name,
        "hardware_name": hardware_name,
        "batch_size": batch,
        "precision": precision,
    }

    data = _call("estimate_memory", args, json_output)

    console.print(
        Panel(
            f"Memory estimate: [bold]{model_name}[/bold] on [cyan]{hardware_name}[/cyan]",
            title="Memory Analysis",
        )
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Peak Memory", f"{data.get('peak_memory_mb', 0):.2f} MB")
    table.add_row("Activations", f"{data.get('activation_memory_bytes', 0) / 1e6:.2f} MB")
    table.add_row("Weights", f"{data.get('weight_memory_bytes', 0) / 1e6:.2f} MB")
    table.add_row("Workspace", f"{data.get('workspace_memory_bytes', 0) / 1e6:.2f} MB")
    fits = "[green]yes[/green]" if data.get("fits_on_device") else "[red]no[/red]"
    table.add_row("Fits on Device", fits)
    table.add_row(
        "Fits in L2", "[green]yes[/green]" if data.get("fits_in_l2_cache") else "[dim]no[/dim]"
    )
    table.add_row("Peak at Step", str(data.get("peak_at_step", "-")))
    table.add_row("Peak Subgraph", data.get("peak_at_subgraph_name", "-"))
    console.print(table)

    suggestions = data.get("optimization_suggestions", [])
    if suggestions:
        console.print()
        console.print("[bold]Optimization Suggestions:[/bold]")
        for s in suggestions:
            console.print(f"  - {s}")


# ---------------------------------------------------------------------------
# compare — multi-hardware comparison
# ---------------------------------------------------------------------------


@mcp.command("compare")
@click.argument("model_name")
@click.argument("hardware_ids", nargs=-1, required=True)
@click.option("--batch", type=int, default=1, show_default=True)
@click.option(
    "--precision",
    "-p",
    type=click.Choice(["fp32", "fp16", "bf16", "int8", "int4"]),
    default="fp16",
    show_default=True,
)
@click.option(
    "--sort",
    "sort_by",
    type=click.Choice(["latency", "energy", "memory"]),
    default="latency",
    show_default=True,
)
@click.option("--thermal", type=str, default=None, help="Thermal profile applied to all targets (e.g. '15W')")
@click.pass_context
def compare_hardware(ctx, model_name, hardware_ids, batch, precision, sort_by, thermal):
    """Compare model performance across multiple hardware targets."""
    json_output = ctx.obj.get("json", False)
    args = {
        "model_name": model_name,
        "hardware_list": list(hardware_ids),
        "batch_size": batch,
        "precision": precision,
        "sort_by": sort_by,
    }
    if thermal:
        args["thermal_profile"] = thermal

    data = _call("compare_hardware", args, json_output)

    results = data.get("results", [])
    errors = data.get("errors", [])

    if results:
        table = Table(title=f"Comparison: {model_name} ({precision})")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Hardware", style="cyan")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Throughput", justify="right")
        table.add_column("Energy (mJ)", justify="right")
        table.add_column("Memory (MB)", justify="right")
        table.add_column("Util (%)", justify="right")

        for i, row in enumerate(results, 1):
            table.add_row(
                str(i),
                row["hardware"],
                f"{row.get('latency_ms', 0):.3f}",
                f"{row.get('throughput_fps', 0):.1f} FPS",
                f"{row.get('energy_mj', 0):.3f}",
                f"{row.get('peak_memory_mb', 0):.2f}",
                f"{row.get('utilization_pct', 0):.1f}",
            )
        console.print(table)

    if errors:
        console.print()
        for e in errors:
            console.print(f"[yellow]Warning:[/yellow] {e['hardware']}: {e['error']}")


# ---------------------------------------------------------------------------
# server — start MCP server
# ---------------------------------------------------------------------------


@mcp.command("server")
@click.option(
    "--sse", is_flag=True, help="Use SSE/HTTP transport (requires mcp, starlette, uvicorn)"
)
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address")
@click.option("--port", type=int, default=8100, show_default=True, help="Port")
@click.pass_context
def start_server(ctx, sse, host, port):
    """Start the graphs MCP server."""
    try:
        from graphs.mcp.transport import main as transport_main
    except ImportError:
        console.print(
            "[red]Error:[/red] graphs.mcp.transport not found.\n"
            "Set PYTHONPATH to the graphs/src directory."
        )
        sys.exit(1)

    # Build argv for the argparse-based transport
    import sys as _sys

    argv = []
    if sse:
        argv.extend(["--sse", "--host", host, "--port", str(port)])
    _sys.argv = ["branes-mcp-server"] + argv
    transport_main()
