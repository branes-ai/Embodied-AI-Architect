"""Tests for the codebase analysis pipeline.

Tests cover:
- CodebaseScanner: build system detection, language detection, ML model finding
- CodebaseConverter: kernel → workload mapping
- CodebaseAnalyzerAgent: end-to-end with mock LLM
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from embodied_ai_architect.codebase.models import (
    CodebaseAnalysisResult,
    ComputeKernel,
    DataflowLink,
    ScanResult,
    SourceFile,
)
from embodied_ai_architect.codebase.scanner import CodebaseScanner
from embodied_ai_architect.codebase.converter import CodebaseConverter

FIXTURES = Path(__file__).parent / "fixtures" / "sample_projects"


# ---------------------------------------------------------------------------
# Scanner tests
# ---------------------------------------------------------------------------


class TestCodebaseScanner:
    def test_scanner_detects_cmake_project(self):
        scanner = CodebaseScanner()
        result = scanner.scan(FIXTURES / "cpp_drone")

        assert result.build_system == "cmake"
        assert result.project_name == "cpp_drone"
        assert "cpp" in result.languages

    def test_scanner_detects_cargo_project(self):
        scanner = CodebaseScanner()
        result = scanner.scan(FIXTURES / "rust_embedded")

        assert result.build_system == "cargo"
        assert "rust" in result.languages
        assert result.project_name == "rust_embedded"

    def test_scanner_detects_python_project(self):
        scanner = CodebaseScanner()
        result = scanner.scan(FIXTURES / "python_ml")

        assert result.build_system == "pip/poetry"
        assert "python" in result.languages

    def test_scanner_finds_ml_models(self):
        scanner = CodebaseScanner()
        result = scanner.scan(FIXTURES / "python_ml")

        assert len(result.ml_models) >= 1
        formats = [m["format"] for m in result.ml_models]
        assert "onnx" in formats

    def test_scanner_extracts_cmake_dependencies(self):
        scanner = CodebaseScanner()
        result = scanner.scan(FIXTURES / "cpp_drone")

        assert "OpenCV" in result.dependencies
        assert "Eigen3" in result.dependencies

    def test_scanner_extracts_cargo_dependencies(self):
        scanner = CodebaseScanner()
        result = scanner.scan(FIXTURES / "rust_embedded")

        assert "nalgebra" in result.dependencies
        assert "embedded-hal" in result.dependencies

    def test_scanner_extracts_python_dependencies(self):
        scanner = CodebaseScanner()
        result = scanner.scan(FIXTURES / "python_ml")

        dep_names = [d.split(">=")[0].split("==")[0] for d in result.dependencies]
        assert "torch" in dep_names

    def test_scanner_classifies_entry_points(self):
        scanner = CodebaseScanner()
        result = scanner.scan(FIXTURES / "cpp_drone")

        entry_points = [f for f in result.source_files if f.role == "entry_point"]
        assert len(entry_points) >= 1

    def test_scanner_classifies_rust_main(self):
        scanner = CodebaseScanner()
        result = scanner.scan(FIXTURES / "rust_embedded")

        entry_points = [f for f in result.source_files if f.role == "entry_point"]
        assert len(entry_points) >= 1
        assert any("main.rs" in ep.path for ep in entry_points)

    def test_scanner_classifies_python_entry_point(self):
        scanner = CodebaseScanner()
        result = scanner.scan(FIXTURES / "python_ml")

        entry_points = [f for f in result.source_files if f.role == "entry_point"]
        assert len(entry_points) >= 1

    def test_scanner_counts_lines(self):
        scanner = CodebaseScanner()
        result = scanner.scan(FIXTURES / "cpp_drone")

        assert result.total_lines > 0
        for sf in result.source_files:
            assert sf.lines >= 0

    def test_scanner_rejects_non_directory(self, tmp_path):
        scanner = CodebaseScanner()
        fake_file = tmp_path / "not_a_dir.txt"
        fake_file.write_text("hello")

        with pytest.raises(ValueError, match="Not a directory"):
            scanner.scan(fake_file)

    def test_scanner_handles_empty_directory(self, tmp_path):
        scanner = CodebaseScanner()
        result = scanner.scan(tmp_path)

        assert result.project_name == tmp_path.name
        assert result.source_files == []
        assert result.total_lines == 0


# ---------------------------------------------------------------------------
# Converter tests
# ---------------------------------------------------------------------------


class TestCodebaseConverter:
    def test_converter_ml_kernel_to_workload(self):
        analysis = CodebaseAnalysisResult(
            project_name="test",
            kernels=[
                ComputeKernel(
                    name="yolo_inference",
                    source_file="inference.py",
                    kernel_type="ml_inference",
                    estimated_ops_per_invocation=8.7e9,
                    frameworks=["pytorch"],
                ),
            ],
        )

        converter = CodebaseConverter()
        profile = converter.to_workload_profile(analysis)

        assert profile["workload_count"] == 1
        assert profile["total_estimated_gflops"] > 0

        workload = profile["workloads"][0]
        assert workload["name"] == "yolo_inference"
        assert workload["model_class"] == "PyTorch"

        # Must have conv/matmul operators for ML inference
        op_types = {op["type"] for op in workload["operators"]}
        assert "convolution" in op_types or "matrix_multiply" in op_types

    def test_converter_signal_processing_kernel(self):
        analysis = CodebaseAnalysisResult(
            project_name="test",
            kernels=[
                ComputeKernel(
                    name="fft_filter",
                    source_file="dsp.cpp",
                    kernel_type="signal_processing",
                    estimated_ops_per_invocation=1e6,
                ),
            ],
        )

        converter = CodebaseConverter()
        profile = converter.to_workload_profile(analysis)

        workload = profile["workloads"][0]
        op_types = {op["type"] for op in workload["operators"]}
        assert "fft" in op_types or "filtering" in op_types

    def test_converter_control_loop_kernel(self):
        analysis = CodebaseAnalysisResult(
            project_name="test",
            kernels=[
                ComputeKernel(
                    name="pid_controller",
                    source_file="control.cpp",
                    kernel_type="control_loop",
                ),
            ],
        )

        converter = CodebaseConverter()
        profile = converter.to_workload_profile(analysis)

        workload = profile["workloads"][0]
        op_types = {op["type"] for op in workload["operators"]}
        assert "matrix_multiply" in op_types or "accumulate" in op_types

    def test_converter_empty_analysis_provides_default(self):
        analysis = CodebaseAnalysisResult(project_name="empty")
        converter = CodebaseConverter()
        profile = converter.to_workload_profile(analysis)

        assert profile["workload_count"] == 1
        assert profile["total_estimated_gflops"] > 0

    def test_converter_preserves_source_traceability(self):
        """Per-workload source_file, line_range, frameworks must survive
        the conversion (issue #42)."""
        analysis = CodebaseAnalysisResult(
            project_name="drone_app",
            kernels=[
                ComputeKernel(
                    name="yolo_inference",
                    source_file="src/perception/detect.py",
                    line_range=(45, 120),
                    kernel_type="ml_inference",
                    estimated_ops_per_invocation=8.7e9,
                    frameworks=["pytorch", "ultralytics"],
                ),
            ],
        )
        converter = CodebaseConverter()
        profile = converter.to_workload_profile(analysis)

        workload = profile["workloads"][0]
        assert workload["source_file"] == "src/perception/detect.py"
        assert workload["line_range"] == [45, 120]
        assert workload["frameworks"] == ["pytorch", "ultralytics"]

    def test_converter_marks_codebase_analysis_source(self):
        """Top-level workload_profile must have source='codebase_analysis'
        so /architect-assess can detect it (issue #42)."""
        analysis = CodebaseAnalysisResult(
            project_name="test",
            kernels=[
                ComputeKernel(
                    name="k",
                    source_file="x.py",
                    kernel_type="ml_inference",
                ),
            ],
        )
        profile = CodebaseConverter().to_workload_profile(analysis)
        assert profile["source"] == "codebase_analysis"
        assert profile["project_name"] == "test"

    def test_converter_multiple_kernels(self):
        analysis = CodebaseAnalysisResult(
            project_name="drone",
            kernels=[
                ComputeKernel(
                    name="perception",
                    source_file="perception.cpp",
                    kernel_type="ml_inference",
                    estimated_ops_per_invocation=8e9,
                ),
                ComputeKernel(
                    name="control",
                    source_file="control.cpp",
                    kernel_type="control_loop",
                    invocation_frequency_hz=100,
                ),
                ComputeKernel(
                    name="imu_fusion",
                    source_file="sensor.cpp",
                    kernel_type="sensor_fusion",
                    invocation_frequency_hz=200,
                ),
            ],
        )

        converter = CodebaseConverter()
        profile = converter.to_workload_profile(analysis)

        assert profile["workload_count"] == 3
        assert profile["total_estimated_gflops"] > 0
        assert profile["total_estimated_memory_mb"] > 0
        assert profile["source"] == "codebase_analysis"

    def test_converter_output_has_required_fields(self):
        """Verify the output has all fields expected by the specialists pipeline."""
        analysis = CodebaseAnalysisResult(
            project_name="test",
            kernels=[
                ComputeKernel(
                    name="test_kernel",
                    source_file="test.py",
                    kernel_type="ml_inference",
                ),
            ],
        )

        converter = CodebaseConverter()
        profile = converter.to_workload_profile(analysis)

        # These fields are required by workload_analyzer/hw_explorer
        assert "workloads" in profile
        assert "total_estimated_gflops" in profile
        assert "total_estimated_memory_mb" in profile
        assert "dominant_op" in profile
        assert "workload_count" in profile
        assert "source" in profile

        # Each workload must have required fields
        for w in profile["workloads"]:
            assert "name" in w
            assert "operators" in w
            assert "estimated_gflops" in w
            assert "estimated_memory_mb" in w
            assert "scheduling" in w

    def test_converter_preserves_framework_mapping(self):
        """Test that different frameworks map to different model classes."""
        for framework, expected_class in [
            ("pytorch", "PyTorch"),
            ("tensorflow", "TensorFlow"),
            ("opencv", "OpenCV"),
            ("eigen", "Eigen"),
        ]:
            analysis = CodebaseAnalysisResult(
                project_name="test",
                kernels=[
                    ComputeKernel(
                        name="kernel",
                        source_file="test.py",
                        kernel_type="general_compute",
                        frameworks=[framework],
                    ),
                ],
            )
            converter = CodebaseConverter()
            profile = converter.to_workload_profile(analysis)
            assert profile["workloads"][0]["model_class"] == expected_class


# ---------------------------------------------------------------------------
# Agent tests (mock LLM)
# ---------------------------------------------------------------------------


class TestCodebaseAnalyzerAgent:
    def test_agent_scan_only(self):
        """Test agent with skip_llm=True (no API key needed)."""
        from embodied_ai_architect.agents.codebase_analyzer import CodebaseAnalyzerAgent

        agent = CodebaseAnalyzerAgent()
        result = agent.execute(
            {
                "project_path": str(FIXTURES / "cpp_drone"),
                "skip_llm": True,
            }
        )

        assert result.success
        assert "scan_result" in result.data
        assert "workload_profile" in result.data

        scan = result.data["scan_result"]
        assert scan["build_system"] == "cmake"
        assert "cpp" in scan["languages"]

    def test_agent_missing_project_path(self):
        from embodied_ai_architect.agents.codebase_analyzer import CodebaseAnalyzerAgent

        agent = CodebaseAnalyzerAgent()
        result = agent.execute({})

        assert not result.success
        assert "project_path" in result.error

    def test_agent_nonexistent_directory(self):
        from embodied_ai_architect.agents.codebase_analyzer import CodebaseAnalyzerAgent

        agent = CodebaseAnalyzerAgent()
        result = agent.execute({"project_path": "/nonexistent/path"})

        assert not result.success

    def test_agent_with_mock_llm(self):
        """Test agent with mocked LLM client."""
        from embodied_ai_architect.agents.codebase_analyzer import CodebaseAnalyzerAgent

        # Create a mock LLM response
        mock_response = MagicMock()
        mock_response.text = '{"build_system": "cmake", "dependencies": [], "project_type": "embedded_app", "key_frameworks": ["opencv"]}'

        mock_llm = MagicMock()
        mock_llm.chat.return_value = mock_response

        with patch("embodied_ai_architect.llm.client.LLMClient", return_value=mock_llm):
            agent = CodebaseAnalyzerAgent()
            result = agent.execute(
                {
                    "project_path": str(FIXTURES / "cpp_drone"),
                }
            )

        assert result.success
        assert "workload_profile" in result.data


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_source_file_defaults(self):
        sf = SourceFile(path="test.py", language="python")
        assert sf.lines == 0
        assert sf.role == "library"

    def test_compute_kernel_defaults(self):
        k = ComputeKernel(name="test", source_file="test.py")
        assert k.kernel_type == "general_compute"
        assert k.parallelism == "sequential"
        assert k.data_types == ["float32"]

    def test_scan_result_defaults(self):
        sr = ScanResult(project_name="test", project_path="/test")
        assert sr.languages == []
        assert sr.build_system == "unknown"

    def test_analysis_result_defaults(self):
        ar = CodebaseAnalysisResult(project_name="test")
        assert ar.kernels == []
        assert ar.dataflow == []
        assert ar.summary == ""

    def test_dataflow_link_defaults(self):
        dl = DataflowLink(source_kernel="a", sink_kernel="b")
        assert dl.data_size_bytes == 0
        assert dl.transfer_type == "memory"


# ---------------------------------------------------------------------------
# LLM tools tests
# ---------------------------------------------------------------------------


class TestCodebaseTools:
    def test_tool_definitions_have_required_fields(self):
        from embodied_ai_architect.llm.codebase_tools import get_codebase_tool_definitions

        tools = get_codebase_tool_definitions()
        assert len(tools) == 3

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_tool_names(self):
        from embodied_ai_architect.llm.codebase_tools import get_codebase_tool_definitions

        tools = get_codebase_tool_definitions()
        names = {t["name"] for t in tools}
        assert names == {"scan_project", "analyze_codebase", "assess_codebase_on_hardware"}

    def test_scan_project_executor(self):
        from embodied_ai_architect.llm.codebase_tools import create_codebase_tool_executors
        import json

        executors = create_codebase_tool_executors()
        result = executors["scan_project"](str(FIXTURES / "cpp_drone"))

        parsed = json.loads(result)
        assert parsed["build_system"] == "cmake"
        assert "cpp" in parsed["languages"]

    def test_scan_project_nonexistent(self):
        from embodied_ai_architect.llm.codebase_tools import create_codebase_tool_executors
        import json

        executors = create_codebase_tool_executors()
        result = executors["scan_project"]("/nonexistent/path")

        parsed = json.loads(result)
        assert "error" in parsed

    def test_tools_registered_in_main_tools(self):
        """Verify codebase tools are registered in the main tools module."""
        from embodied_ai_architect.llm.tools import get_tool_definitions, create_tool_executors

        definitions = get_tool_definitions()
        names = {t["name"] for t in definitions}
        assert "scan_project" in names

        executors = create_tool_executors()
        assert "scan_project" in executors


# ---------------------------------------------------------------------------
# Session creation from codebase analysis (issue #42)
# ---------------------------------------------------------------------------


class TestCodebaseSessionCreation:
    """Tests for _save_codebase_session and the codebase_metadata field."""

    def test_save_codebase_session_populates_metadata(self, tmp_path, monkeypatch):
        """The CLI helper must persist project_path, languages, and kernel
        sources into the session's codebase_metadata field for /architect-*."""
        from embodied_ai_architect.cli.commands.codebase import _save_codebase_session
        from embodied_ai_architect.graphs.session_store import SessionStore

        # Point SessionStore at a temp directory so we don't pollute the real one
        monkeypatch.setattr(
            "embodied_ai_architect.graphs.session_store.DEFAULT_SESSION_DIR",
            tmp_path,
            raising=False,
        )

        data = {
            "scan_result": {
                "project_name": "drone_app",
                "languages": ["python", "cpp"],
                "build_system": "cmake",
                "total_lines": 12500,
                "source_files": [{"path": "main.py", "language": "python", "lines": 200}],
                "dependencies": ["torch", "ultralytics"],
                "ml_models": [{"path": "yolov8.pt", "format": "pt", "size_bytes": 6_000_000}],
            },
            "analysis": {
                "kernels": [
                    {
                        "name": "yolo_inference",
                        "source_file": "src/perception/detect.py",
                        "kernel_type": "ml_inference",
                    }
                ],
            },
            "workload_profile": {
                "source": "codebase_analysis",
                "workloads": [
                    {
                        "name": "yolo_inference",
                        "source_file": "src/perception/detect.py",
                        "line_range": [45, 120],
                        "estimated_gflops": 8.4,
                        "estimated_memory_mb": 50.0,
                        "kernel_type": "ml_inference",
                        "frameworks": ["pytorch"],
                    },
                ],
            },
        }

        store = SessionStore(session_dir=tmp_path)
        session_id = _save_codebase_session("/path/to/drone_app", data, 15.0, 33.0)

        # Reload via the same temp store
        loaded = store.load(session_id)
        assert loaded is not None
        assert loaded["use_case"] == "codebase_analysis"
        assert loaded["workload_profile"]["source"] == "codebase_analysis"

        meta = loaded.get("codebase_metadata", {})
        assert meta["project_path"] == "/path/to/drone_app"
        assert meta["project_name"] == "drone_app"
        assert "python" in meta["languages"]
        assert meta["build_system"] == "cmake"
        assert meta["kernel_count"] == 1
        assert meta["scan_summary"]["total_lines"] == 12500

        # Per-workload source data must be preserved on the workload itself
        workload = loaded["workload_profile"]["workloads"][0]
        assert workload["source_file"] == "src/perception/detect.py"
        assert workload["line_range"] == [45, 120]
