"""Static file scanner for codebase analysis.

Walks a project directory tree, identifies languages, detects build systems,
finds ML model files, and classifies files by role. No LLM needed — this is
the fast first pass that feeds into the LLM analyzer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from embodied_ai_architect.codebase.models import ScanResult, SourceFile

logger = logging.getLogger(__name__)

# Language detection by file extension
LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".pyx": "python",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "header",
    ".hpp": "header",
    ".hxx": "header",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
    ".cu": "cuda",
    ".cuh": "cuda",
}

# ML model file extensions
ML_MODEL_EXTENSIONS: set[str] = {
    ".onnx",
    ".pt",
    ".pth",
    ".tflite",
    ".pb",
    ".safetensors",
    ".bin",
    ".h5",
    ".keras",
}

# Build system detection files
BUILD_SYSTEM_FILES: dict[str, str] = {
    "CMakeLists.txt": "cmake",
    "Cargo.toml": "cargo",
    "pyproject.toml": "pip/poetry",
    "setup.py": "pip/poetry",
    "Makefile": "make",
    "meson.build": "meson",
    "BUILD": "bazel",
    "BUILD.bazel": "bazel",
}

# Directories to skip during scanning
SKIP_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".tox",
    "build",
    "dist",
    ".eggs",
    "target",  # Rust build dir
}

# Entry point patterns per language
ENTRY_POINT_MARKERS: dict[str, list[str]] = {
    "python": ['if __name__ == "__main__"', "if __name__ == '__main__'"],
    "cpp": ["int main(", "int main ("],
    "c": ["int main(", "int main ("],
    "rust": ["fn main(", "fn main ("],
}


class CodebaseScanner:
    """Static file scanner for project directories.

    Walks the directory tree and produces a ScanResult with:
    - File inventory classified by language and role
    - Detected build system
    - Found ML model files
    - Extracted dependencies
    """

    def scan(self, project_path: Path) -> ScanResult:
        """Scan a project directory and return structured results.

        Args:
            project_path: Root directory of the project to scan.

        Returns:
            ScanResult with file inventory and project metadata.
        """
        project_path = Path(project_path).resolve()
        if not project_path.is_dir():
            raise ValueError(f"Not a directory: {project_path}")

        project_name = project_path.name

        source_files: list[SourceFile] = []
        ml_models: list[dict] = []
        languages_seen: set[str] = set()
        build_system = "unknown"
        dependencies: list[str] = []
        total_lines = 0

        for item in self._walk(project_path):
            rel_path = str(item.relative_to(project_path))

            # Check for build system files
            if item.name in BUILD_SYSTEM_FILES:
                build_system = BUILD_SYSTEM_FILES[item.name]
                deps = self._extract_dependencies(item, build_system)
                dependencies.extend(deps)

            # Check for ML model files
            if item.suffix.lower() in ML_MODEL_EXTENSIONS:
                ml_models.append(
                    {
                        "path": rel_path,
                        "format": item.suffix.lower().lstrip("."),
                        "size_bytes": item.stat().st_size,
                    }
                )
                continue

            # Identify source language
            lang = LANGUAGE_MAP.get(item.suffix.lower())
            if lang is None:
                continue

            languages_seen.add(lang)

            # Count lines
            line_count = self._count_lines(item)
            total_lines += line_count

            # Classify role
            role = self._classify_role(item, rel_path, lang)
            entry_type = None
            if role == "entry_point":
                entry_type = self._classify_entry_type(item, rel_path, lang)

            source_files.append(
                SourceFile(
                    path=rel_path,
                    language=lang,
                    lines=line_count,
                    role=role,
                    entry_type=entry_type,
                )
            )

        # Sort: entry points first, then by line count descending
        role_priority = {"entry_point": 0, "library": 1, "test": 2, "config": 3, "build": 4}
        source_files.sort(key=lambda f: (role_priority.get(f.role, 5), -f.lines))

        recommendation = self._recommend_application(source_files)

        return ScanResult(
            project_name=project_name,
            project_path=str(project_path),
            languages=sorted(languages_seen),
            source_files=source_files,
            ml_models=ml_models,
            build_system=build_system,
            dependencies=dependencies,
            total_lines=total_lines,
            recommended_entry_point=recommendation,
        )

    def _walk(self, root: Path):
        """Walk directory tree, skipping ignored directories."""
        for item in sorted(root.iterdir()):
            if item.is_dir():
                if item.name in SKIP_DIRS:
                    continue
                yield from self._walk(item)
            elif item.is_file():
                yield item

    def _count_lines(self, path: Path) -> int:
        """Count lines in a file, handling encoding errors gracefully."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return sum(1 for _ in f)
        except (OSError, UnicodeDecodeError):
            return 0

    def _classify_role(self, path: Path, rel_path: str, language: str) -> str:
        """Classify a source file's role in the project."""
        name_lower = path.name.lower()
        rel_lower = rel_path.lower()

        # Test files
        if "test" in name_lower or rel_lower.startswith("tests/") or rel_lower.startswith("test/"):
            return "test"

        # Config/build files
        if name_lower in ("setup.py", "setup.cfg", "conftest.py"):
            return "config"
        if name_lower.endswith(".toml") or name_lower.endswith(".cfg"):
            return "config"

        # Entry point detection
        markers = ENTRY_POINT_MARKERS.get(language, [])
        if markers and self._file_contains_any(path, markers):
            return "entry_point"

        # Rust lib.rs is library, main.rs is entry point
        if language == "rust":
            if name_lower == "main.rs":
                return "entry_point"
            if name_lower == "lib.rs":
                return "library"

        return "library"

    def _classify_entry_type(self, path: Path, rel_path: str, language: str) -> str:
        """Sub-classify an entry point as application, example, or test.

        Uses path-based heuristics first (strongest signal), then filename
        patterns. Default is "application" for entry points that don't match
        example or test patterns.
        """
        rel_lower = rel_path.lower()
        name_lower = path.stem.lower()

        # Path-based heuristics (strongest signal)
        example_dirs = ("examples/", "example/", "demos/", "demo/", "samples/")
        test_dirs = ("tests/", "test/", "benchmarks/", "bench/")
        utility_dirs = ("scripts/", "tools/", "utils/", "utilities/")

        for prefix in example_dirs:
            if rel_lower.startswith(prefix) or f"/{prefix}" in rel_lower:
                return "example"
        for prefix in test_dirs:
            if rel_lower.startswith(prefix) or f"/{prefix}" in rel_lower:
                return "test"
        for prefix in utility_dirs:
            if rel_lower.startswith(prefix) or f"/{prefix}" in rel_lower:
                return "example"

        # Filename-based heuristics
        example_prefixes = ("example_", "demo_", "sample_")
        example_suffixes = ("_example", "_demo", "_sample")
        if any(name_lower.startswith(p) for p in example_prefixes):
            return "example"
        if any(name_lower.endswith(s) for s in example_suffixes):
            return "example"

        test_prefixes = ("test_",)
        test_suffixes = ("_test",)
        if any(name_lower.startswith(p) for p in test_prefixes):
            return "test"
        if any(name_lower.endswith(s) for s in test_suffixes):
            return "test"

        app_names = ("run", "main", "app", "__main__")
        if name_lower in app_names:
            return "application"

        # Default: entry points not matching example/test are application candidates
        return "application"

    def _recommend_application(self, source_files: list[SourceFile]) -> dict | None:
        """Pick the best application entry point from classified files.

        Returns a dict with path, confidence, and alternatives, or None if
        no entry points were found.
        """
        app_entries = [
            sf for sf in source_files if sf.role == "entry_point" and sf.entry_type == "application"
        ]

        if not app_entries:
            # Fall back to any entry point as a starting suggestion
            all_entries = [sf for sf in source_files if sf.role == "entry_point"]
            if not all_entries:
                return None
            best = max(all_entries, key=lambda sf: sf.lines)
            return {
                "path": best.path,
                "confidence": "low",
                "alternatives": [sf.path for sf in all_entries if sf.path != best.path],
            }

        if len(app_entries) == 1:
            return {
                "path": app_entries[0].path,
                "confidence": "high",
                "alternatives": [],
            }

        # Multiple application candidates — rank by line count (larger = more likely main)
        app_entries.sort(key=lambda sf: sf.lines, reverse=True)
        return {
            "path": app_entries[0].path,
            "confidence": "medium",
            "alternatives": [sf.path for sf in app_entries[1:]],
        }

    def _file_contains_any(self, path: Path, markers: list[str]) -> bool:
        """Check if a file contains any of the given marker strings."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(64_000)  # Read first 64KB only
            return any(marker in content for marker in markers)
        except (OSError, UnicodeDecodeError):
            return False

    def _extract_dependencies(self, build_file: Path, build_system: str) -> list[str]:
        """Extract dependency names from build files."""
        deps: list[str] = []
        try:
            content = build_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return deps

        if build_system == "pip/poetry":
            deps.extend(self._parse_python_deps(content, build_file.name))
        elif build_system == "cargo":
            deps.extend(self._parse_cargo_deps(content))
        elif build_system == "cmake":
            deps.extend(self._parse_cmake_deps(content))

        return deps

    def _parse_python_deps(self, content: str, filename: str) -> list[str]:
        """Extract dependency names from pyproject.toml or setup.py."""
        deps: list[str] = []
        if filename == "pyproject.toml":
            in_deps = False
            for line in content.splitlines():
                stripped = line.strip()
                if stripped == "dependencies = [" or stripped.startswith("dependencies"):
                    in_deps = True
                    continue
                if in_deps:
                    if stripped == "]":
                        in_deps = False
                        continue
                    # Extract package name from "package>=version"
                    pkg = stripped.strip('"').strip("'").strip(",")
                    if pkg:
                        name = pkg.split(">=")[0].split("<=")[0].split("==")[0].split("[")[0]
                        name = name.strip()
                        if name:
                            deps.append(name)
        return deps

    def _parse_cargo_deps(self, content: str) -> list[str]:
        """Extract dependency names from Cargo.toml."""
        deps: list[str] = []
        in_deps = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped == "[dependencies]":
                in_deps = True
                continue
            if stripped.startswith("[") and in_deps:
                in_deps = False
                continue
            if in_deps and "=" in stripped:
                name = stripped.split("=")[0].strip()
                if name:
                    deps.append(name)
        return deps

    def _parse_cmake_deps(self, content: str) -> list[str]:
        """Extract dependency names from CMakeLists.txt."""
        deps: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("find_package("):
                # find_package(OpenCV REQUIRED)
                pkg = stripped[len("find_package(") :].split(")")[0].split()[0]
                if pkg:
                    deps.append(pkg)
            elif stripped.startswith("target_link_libraries("):
                # target_link_libraries(app opencv_core eigen3)
                parts = stripped[len("target_link_libraries(") :].split(")")[0].split()
                # Skip first arg (target name) and keywords
                keywords = {"PUBLIC", "PRIVATE", "INTERFACE"}
                for part in parts[1:]:
                    if part not in keywords:
                        deps.append(part)
        return deps
