"""RTL Synthesis Tool — gate-level netlist and PPA metrics.

Uses Yosys with mock fallback. Enhanced with process_nm for area scaling.
Timeout is configurable and defaults to a complexity-based estimate.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Sentinel: caller wants auto-detection based on RTL complexity
AUTO_TIMEOUT = -1


@dataclass
class SynthesisResult:
    """Results from RTL synthesis."""

    success: bool
    area_cells: int = 0
    area_um2: float = 0.0
    num_wires: int = 0
    num_cells: int = 0
    cell_counts: dict[str, int] = field(default_factory=dict)
    netlist_path: Optional[str] = None
    log_excerpt: str = ""
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    source: str = "yosys"  # "yosys" | "mock" — lets callers detect mock data

    def to_dict(self) -> dict:
        return asdict(self)


class RTLSynthesisTool:
    """Synthesize RTL to gate-level netlist using Yosys.

    Falls back to mock synthesis if Yosys is not available or times out.

    Args:
        work_dir: Working directory for intermediate files.
        process_nm: Technology node for area scaling.
        timeout: Seconds before Yosys is killed.
            * positive int — fixed timeout
            * AUTO_TIMEOUT (-1, default) — estimated from RTL complexity
    """

    def __init__(
        self,
        work_dir: Optional[Path] = None,
        process_nm: int = 28,
        timeout: int = AUTO_TIMEOUT,
    ):
        self.work_dir = (work_dir or Path(tempfile.mkdtemp())).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.process_nm = process_nm
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Complexity-based timeout estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_timeout(rtl_source: str) -> int:
        """Estimate a reasonable Yosys timeout from RTL complexity.

        Heuristic tiers:
          - simple  (<100 lines, <5 always blocks, no memories): 30 s
          - medium  (<500 lines, no inferred memories):         120 s
          - complex (everything else):                          300 s
        """
        lines = len(rtl_source.splitlines())
        always_blocks = rtl_source.count("always")
        # Inferred memories: reg [...] name [...]
        memory_decls = len(re.findall(r"reg\s*\[.*?\]\s*\w+\s*\[", rtl_source))

        if lines < 100 and always_blocks < 5 and memory_decls == 0:
            return 30
        if lines < 500 and memory_decls == 0:
            return 120
        return 300

    def _resolve_timeout(self, rtl_source: str) -> int:
        """Return the effective timeout for this run."""
        if self.timeout != AUTO_TIMEOUT:
            return self.timeout
        return self.estimate_timeout(rtl_source)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        rtl_source: str,
        top_module: str,
        optimize: bool = True,
        liberty_file: Optional[str] = None,
    ) -> dict:
        """Synthesize RTL and extract metrics."""
        rtl_path = self.work_dir / f"{top_module}.sv"
        rtl_path.write_text(rtl_source)

        script = self._generate_script(rtl_path, top_module, optimize, liberty_file)
        script_path = self.work_dir / "synth.ys"
        script_path.write_text(script)

        effective_timeout = self._resolve_timeout(rtl_source)
        logger.info("Yosys synthesis timeout: %ds (module=%s)", effective_timeout, top_module)

        try:
            proc = subprocess.run(
                ["yosys", "-s", str(script_path)],
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=self.work_dir,
            )
            if proc.returncode != 0:
                errors = [l for l in proc.stderr.splitlines() if "error" in l.lower()]
                if not errors:
                    errors = [proc.stderr[:500] if proc.stderr else "Unknown synthesis error"]
                return SynthesisResult(
                    success=False,
                    errors=errors,
                    log_excerpt=proc.stdout[-1000:] if proc.stdout else "",
                ).to_dict()

            result = self._parse_stats(proc.stdout)
            result.netlist_path = str(self.work_dir / "netlist.json")
            result.log_excerpt = proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout
            return result.to_dict()

        except FileNotFoundError:
            result = self._mock_synthesis(rtl_source, top_module, reason="yosys_not_found")
            return result.to_dict()
        except subprocess.TimeoutExpired:
            logger.warning(
                "Yosys timed out after %ds for %s — falling back to mock",
                effective_timeout,
                top_module,
            )
            result = self._mock_synthesis(
                rtl_source, top_module, reason=f"timeout_{effective_timeout}s"
            )
            return result.to_dict()

    def _generate_script(
        self,
        rtl_path: Path,
        top_module: str,
        optimize: bool,
        liberty_file: Optional[str],
    ) -> str:
        opt_cmds = "opt; opt_clean; opt -full;" if optimize else "opt;"
        script = f"""
read_verilog -sv {rtl_path}
hierarchy -check -top {top_module}
proc; {opt_cmds}
fsm; {opt_cmds}
memory; {opt_cmds}
techmap; {opt_cmds}
"""
        if liberty_file and Path(liberty_file).exists():
            script += f"abc -liberty {liberty_file}\n"
        else:
            script += f"abc -g AND,OR,XOR,NAND,NOR,XNOR,MUX,AOI3,OAI3,AOI4,OAI4\n{opt_cmds}\n"

        script += f"""
stat
write_json {self.work_dir / 'netlist.json'}
write_verilog {self.work_dir / 'synth.v'}
"""
        return script

    def _parse_stats(self, stdout: str) -> SynthesisResult:
        result = SynthesisResult(success=True)
        cell_counts: dict[str, int] = {}
        in_stats = False

        for line in stdout.splitlines():
            if "Printing statistics" in line or line.strip().startswith("=== "):
                in_stats = True
                continue
            if not in_stats:
                continue

            line = line.strip()
            if "Number of wires:" in line:
                match = re.search(r"(\d+)", line)
                if match:
                    result.num_wires = int(match.group(1))
            elif line.endswith("wires") and "wire bits" not in line and "public" not in line:
                match = re.search(r"(\d+)\s+wires", line)
                if match:
                    result.num_wires = int(match.group(1))
            elif "Number of cells:" in line:
                match = re.search(r"(\d+)", line)
                if match:
                    result.num_cells = int(match.group(1))
            elif line.endswith("cells"):
                match = re.search(r"(\d+)\s+cells", line)
                if match:
                    result.num_cells = int(match.group(1))
            elif "$_" in line:
                cell_match = re.search(r"(\d+)\s+\$_(\w+)_", line)
                if cell_match:
                    cell_counts[cell_match.group(2)] = int(cell_match.group(1))

        result.cell_counts = cell_counts
        result.area_cells = sum(cell_counts.values()) if cell_counts else result.num_cells
        # Scale area by process node
        from embodied_ai_architect.graphs.technology import get_technology

        tech = get_technology(self.process_nm)
        result.area_um2 = result.area_cells * tech.cell_area_um2
        return result

    def _mock_synthesis(
        self, rtl: str, top: str, reason: str = "yosys_not_found"
    ) -> SynthesisResult:
        num_always = rtl.count("always")
        num_assign = rtl.count("assign")
        num_ops = len(re.findall(r"[\+\-\*\/\&\|\^]", rtl))
        estimated_cells = max(10, (num_always * 10) + (num_assign * 3) + (num_ops * 2))

        from embodied_ai_architect.graphs.technology import get_technology

        tech = get_technology(self.process_nm)

        warnings = [f"Using mock synthesis (reason: {reason})"]
        if "timeout" in reason:
            warnings.append("Metrics are heuristic estimates — real synthesis timed out")

        return SynthesisResult(
            success=True,
            area_cells=estimated_cells,
            area_um2=estimated_cells * tech.cell_area_um2,
            num_wires=estimated_cells * 2,
            num_cells=estimated_cells,
            cell_counts={"estimated": estimated_cells},
            log_excerpt=f"Mock synthesis — {reason}",
            warnings=warnings,
            source="mock",
        )
