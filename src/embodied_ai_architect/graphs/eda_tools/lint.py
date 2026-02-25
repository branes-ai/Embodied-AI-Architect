"""RTL Linting Tool — fast syntax checking before synthesis.

3-tier fallback: Verilator → Yosys → regex-based parsing.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LintResult:
    """Results from RTL linting."""

    success: bool
    module_name: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ports: list[dict] = field(default_factory=list)
    parameters: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class RTLLintTool:
    """Lint and parse RTL to extract structure.

    Uses Verilator for fast linting, falls back to Yosys,
    then to basic regex-based parsing.
    """

    def __init__(self, work_dir: Optional[Path] = None):
        self.work_dir = (work_dir or Path(tempfile.mkdtemp())).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(self, rtl_source: str) -> dict:
        """Lint RTL and extract structure."""
        rtl_path = self.work_dir / "design.sv"
        rtl_path.write_text(rtl_source)

        result = self._try_verilator(rtl_path, rtl_source)
        if result is not None:
            return result.to_dict()

        result = self._try_yosys(rtl_path, rtl_source)
        if result is not None:
            return result.to_dict()

        return self._regex_parse(rtl_source).to_dict()

    def _try_verilator(self, rtl_path: Path, rtl_source: str) -> Optional[LintResult]:
        try:
            proc = subprocess.run(
                ["verilator", "--lint-only", "-Wall", "-Wno-fatal", str(rtl_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.work_dir,
            )
            errors = []
            warnings = []
            for line in proc.stderr.splitlines():
                if "%Error" in line:
                    errors.append(line.strip())
                elif "%Warning" in line:
                    warnings.append(line.strip())

            module_name, ports, params = self._extract_module_info(rtl_source)
            return LintResult(
                success=(proc.returncode == 0 and len(errors) == 0),
                module_name=module_name,
                errors=errors,
                warnings=warnings,
                ports=ports,
                parameters=params,
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return LintResult(success=False, errors=["Verilator lint timed out after 30s"])

    def _try_yosys(self, rtl_path: Path, rtl_source: str) -> Optional[LintResult]:
        script = f"read_verilog -sv {rtl_path}\nhierarchy -check\n"
        script_path = self.work_dir / "lint.ys"
        script_path.write_text(script)

        try:
            proc = subprocess.run(
                ["yosys", "-s", str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.work_dir,
            )
            errors = []
            warnings = []
            for line in proc.stderr.splitlines() + proc.stdout.splitlines():
                if "ERROR:" in line or "error:" in line.lower():
                    errors.append(line.strip())
                elif "Warning:" in line:
                    warnings.append(line.strip())

            module_name, ports, params = self._extract_module_info(rtl_source)
            return LintResult(
                success=(proc.returncode == 0 and len(errors) == 0),
                module_name=module_name,
                errors=errors,
                warnings=warnings,
                ports=ports,
                parameters=params,
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return LintResult(success=False, errors=["Yosys lint timed out after 30s"])

    def _regex_parse(self, rtl_source: str) -> LintResult:
        errors = []
        module_name, ports, params = self._extract_module_info(rtl_source)

        if not module_name:
            errors.append("No module definition found")
        n_module = len(re.findall(r"\bmodule\b", rtl_source))
        n_endmodule = len(re.findall(r"\bendmodule\b", rtl_source))
        if n_module != n_endmodule:
            errors.append("Mismatched module/endmodule")

        return LintResult(
            success=(len(errors) == 0),
            module_name=module_name,
            errors=errors,
            warnings=["Using regex-based parsing - EDA tools not available"],
            ports=ports,
            parameters=params,
        )

    def _extract_module_info(self, rtl: str) -> tuple:
        module_match = re.search(r"module\s+(\w+)", rtl)
        module_name = module_match.group(1) if module_match else ""

        port_pattern = r"(input|output|inout)\s+(?:wire|reg|logic)?\s*(?:\[[\d:]+\])?\s*(\w+)"
        ports = [{"direction": p[0], "name": p[1]} for p in re.findall(port_pattern, rtl)]

        param_pattern = r"parameter\s+(\w+)\s*=\s*([^,;]+)"
        params = [{"name": p[0], "value": p[1].strip()} for p in re.findall(param_pattern, rtl)]

        return module_name, ports, params
