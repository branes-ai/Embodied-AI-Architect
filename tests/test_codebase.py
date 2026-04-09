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
        # 4 tools after issue #37 added design_from_codebase
        assert len(tools) == 4

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_tool_names(self):
        from embodied_ai_architect.llm.codebase_tools import get_codebase_tool_definitions

        tools = get_codebase_tool_definitions()
        names = {t["name"] for t in tools}
        assert names == {
            "scan_project",
            "analyze_codebase",
            "assess_codebase_on_hardware",
            "design_from_codebase",  # issue #37
        }

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

        # Use a real absolute path so .resolve() doesn't error
        project_dir = tmp_path / "drone_app_src"
        project_dir.mkdir()

        store = SessionStore(session_dir=tmp_path)
        session_id = _save_codebase_session(str(project_dir), data, 15.0, 33.0)

        # Reload via the same temp store
        loaded = store.load(session_id)
        assert loaded is not None
        assert loaded["use_case"] == "codebase_analysis"
        assert loaded["workload_profile"]["source"] == "codebase_analysis"

        meta = loaded.get("codebase_metadata", {})
        # project_path must be absolute (resolved) so drill-down works
        # regardless of cwd at session load time
        assert meta["project_path"] == str(project_dir.resolve())
        assert Path(meta["project_path"]).is_absolute()
        assert meta["project_name"] == "drone_app"
        assert "python" in meta["languages"]
        assert meta["build_system"] == "cmake"
        assert meta["kernel_count"] == 1
        assert meta["scan_summary"]["total_lines"] == 12500

        # Per-workload source data must be preserved on the workload itself
        workload = loaded["workload_profile"]["workloads"][0]
        assert workload["source_file"] == "src/perception/detect.py"
        assert workload["line_range"] == [45, 120]


# ---------------------------------------------------------------------------
# Issue #37: Codebase → SoCDesignState bridge
# ---------------------------------------------------------------------------


class TestCodebaseToSoCState:
    """The new codebase_to_soc_state() / codebase_data_to_soc_state() bridge."""

    def _make_analysis(self, kernel_type: str = "ml_inference") -> "object":
        from embodied_ai_architect.codebase.models import (
            CodebaseAnalysisResult,
            ComputeKernel,
        )

        return CodebaseAnalysisResult(
            project_name="test_drone_app",
            languages=["python"],
            build_system="pip",
            dependencies=["torch", "ultralytics", "numpy"],
            ml_models=[{"name": "yolov8.pt", "framework": "pytorch"}],
            kernels=[
                ComputeKernel(
                    name="yolo_detection",
                    source_file="perception/detect.py",
                    line_range=(10, 80),
                    kernel_type=kernel_type,
                    estimated_ops_per_invocation=8.4e9,
                    frameworks=["pytorch"],
                ),
                ComputeKernel(
                    name="object_tracker",
                    source_file="perception/track.py",
                    line_range=(5, 60),
                    kernel_type=kernel_type,
                    frameworks=["numpy"],
                ),
            ],
            summary="Drone perception pipeline with YOLO and tracker",
        )

    def test_returns_populated_soc_state(self, tmp_path):
        from embodied_ai_architect.codebase.converter import codebase_to_soc_state

        analysis = self._make_analysis()
        state = codebase_to_soc_state(analysis, project_path=str(tmp_path))

        # Workload profile populated by the converter
        wp = state["workload_profile"]
        assert wp["source"] == "codebase_analysis"
        assert wp["workload_count"] == 2
        assert wp["total_estimated_gflops"] > 0
        # use_case stays canonical (CodeRabbit PR #88) — the dominant kernel
        # type is preserved separately on metadata for downstream consumers.
        assert state["use_case"] == "codebase_analysis"

    def test_goal_includes_project_name_and_summary(self, tmp_path):
        from embodied_ai_architect.codebase.converter import codebase_to_soc_state

        analysis = self._make_analysis()
        state = codebase_to_soc_state(analysis, project_path=str(tmp_path))
        goal = state["goal"]
        assert "test_drone_app" in goal
        assert "ml_inference" in goal
        assert "Drone perception" in goal

    def test_dominant_kernel_type_recorded_on_metadata(self, tmp_path):
        from embodied_ai_architect.codebase.converter import codebase_to_soc_state

        analysis = self._make_analysis(kernel_type="signal_processing")
        state = codebase_to_soc_state(analysis, project_path=str(tmp_path))
        # use_case is canonical, but the dominant kernel type is preserved
        # so a future consumer can branch on it without us minting
        # unrecognized labels
        assert state["use_case"] == "codebase_analysis"
        assert state["codebase_metadata"]["dominant_kernel_type"] == "signal_processing"

    def test_dominant_kernel_type_default_when_no_kernels(self, tmp_path):
        from embodied_ai_architect.codebase.converter import codebase_to_soc_state
        from embodied_ai_architect.codebase.models import CodebaseAnalysisResult

        analysis = CodebaseAnalysisResult(project_name="empty")
        state = codebase_to_soc_state(analysis, project_path=str(tmp_path))
        assert state["use_case"] == "codebase_analysis"
        assert state["codebase_metadata"]["dominant_kernel_type"] == "general_compute"

    def test_codebase_metadata_carries_project_path(self, tmp_path):
        from embodied_ai_architect.codebase.converter import codebase_to_soc_state

        analysis = self._make_analysis()
        state = codebase_to_soc_state(analysis, project_path=str(tmp_path))
        meta = state["codebase_metadata"]
        # Path is absolute (resolved) so /architect-drill source: works
        assert Path(meta["project_path"]).is_absolute()
        assert meta["project_name"] == "test_drone_app"
        assert meta["build_system"] == "pip"
        assert meta["kernel_count"] == 2
        assert "torch" in meta["scan_summary"]["dependencies"]

    def test_constraints_propagated(self, tmp_path):
        from embodied_ai_architect.codebase.converter import codebase_to_soc_state
        from embodied_ai_architect.graphs.soc_state import DesignConstraints

        analysis = self._make_analysis()
        constraints = DesignConstraints(
            max_power_watts=5.0,
            max_latency_ms=33.3,
            max_area_mm2=100.0,
        )
        state = codebase_to_soc_state(analysis, constraints=constraints, project_path=str(tmp_path))
        assert state["constraints"]["max_power_watts"] == 5.0
        assert state["constraints"]["max_latency_ms"] == 33.3
        assert state["constraints"]["max_area_mm2"] == 100.0

    def test_dict_helper_accepts_payload(self, tmp_path):
        """codebase_data_to_soc_state accepts plain dict payloads from the agent."""
        from embodied_ai_architect.codebase.converter import codebase_data_to_soc_state

        analysis_data = {
            "project_name": "data_test",
            "kernels": [
                {
                    "name": "k1",
                    "source_file": "k1.py",
                    "line_range": [1, 10],
                    "kernel_type": "ml_inference",
                    "estimated_ops_per_invocation": 5e9,
                    "frameworks": ["pytorch"],
                }
            ],
        }
        scan_data = {
            "languages": ["python"],
            "build_system": "pip",
            "dependencies": ["torch"],
        }
        state = codebase_data_to_soc_state(
            analysis_data=analysis_data,
            scan_data=scan_data,
            project_path=str(tmp_path),
        )
        assert state["use_case"] == "codebase_analysis"
        assert state["workload_profile"]["workload_count"] == 1
        assert state["codebase_metadata"]["build_system"] == "pip"
        assert state["codebase_metadata"]["dominant_kernel_type"] == "ml_inference"

    def test_dict_helper_backfills_from_scan_when_analysis_dump_is_default(self, tmp_path):
        """CodeRabbit PR #88: model_dump() of an empty CodebaseAnalysisResult
        produces `[]` / `"unknown"` defaults. The merge must still backfill
        from scan_data instead of treating the defaults as "set"."""
        from embodied_ai_architect.codebase.converter import codebase_data_to_soc_state
        from embodied_ai_architect.codebase.models import CodebaseAnalysisResult

        # Simulate the agent path: an analysis dump that's mostly defaults
        empty_analysis = CodebaseAnalysisResult(project_name="").model_dump()
        # languages=[], build_system="unknown", source_files=[], etc. are
        # already set on the dump — naive setdefault would skip the fallback
        scan_data = {
            "project_name": "real_project",
            "languages": ["python", "cpp"],
            "build_system": "cmake",
            "dependencies": ["torch", "opencv-python"],
            "ml_models": [{"name": "yolo.pt"}],
            "source_files": [{"path": "main.py", "language": "python"}],
        }
        state = codebase_data_to_soc_state(
            analysis_data=empty_analysis,
            scan_data=scan_data,
            project_path=str(tmp_path),
        )
        meta = state["codebase_metadata"]
        # All scan fields successfully backfilled into the metadata
        assert meta["project_name"] == "real_project"
        assert "python" in meta["languages"]
        assert "cpp" in meta["languages"]
        assert meta["build_system"] == "cmake"
        assert "torch" in meta["scan_summary"]["dependencies"]
        assert meta["ml_models"]

    def test_session_round_trips_state(self, tmp_path):
        from embodied_ai_architect.codebase.converter import codebase_to_soc_state
        from embodied_ai_architect.graphs.session_store import SessionStore

        analysis = self._make_analysis()
        state = codebase_to_soc_state(
            analysis, project_path=str(tmp_path), session_id="soc_codebase37"
        )
        store = SessionStore(session_dir=tmp_path)
        store.save(state)

        loaded = store.load("soc_codebase37")
        assert loaded is not None
        assert loaded["use_case"] == "codebase_analysis"
        assert loaded["workload_profile"]["workload_count"] == 2
        assert loaded["codebase_metadata"]["project_name"] == "test_drone_app"
        assert loaded["codebase_metadata"]["dominant_kernel_type"] == "ml_inference"


# ---------------------------------------------------------------------------
# Issue #38: Inferred design constraints from codebase characteristics
# ---------------------------------------------------------------------------


class TestInferConstraints:
    """The infer_constraints() heuristic layer."""

    def _kernel(self, **kwargs):
        from embodied_ai_architect.codebase.models import ComputeKernel

        defaults = {
            "name": "k",
            "source_file": "k.py",
            "line_range": (1, 10),
            "kernel_type": "general_compute",
            "estimated_ops_per_invocation": 0.0,
            "invocation_frequency_hz": 0.0,
        }
        defaults.update(kwargs)
        return ComputeKernel(**defaults)

    def _analysis(self, *kernels):
        from embodied_ai_architect.codebase.models import CodebaseAnalysisResult

        return CodebaseAnalysisResult(project_name="test", kernels=list(kernels))

    def test_no_kernels_returns_empty(self):
        from embodied_ai_architect.codebase.converter import infer_constraints

        suggestions = infer_constraints(self._analysis())
        assert suggestions.constraints == []
        assert "0 kernels" in suggestions.summary

    def test_control_loop_at_100hz_implies_10ms_latency(self):
        """The headline heuristic from the issue body."""
        from embodied_ai_architect.codebase.converter import infer_constraints

        analysis = self._analysis(
            self._kernel(
                name="control",
                kernel_type="control_loop",
                invocation_frequency_hz=100.0,
            )
        )
        suggestions = infer_constraints(analysis)
        latency_entries = [c for c in suggestions.constraints if c.name == "max_latency_ms"]
        assert len(latency_entries) == 1
        latency = latency_entries[0]
        assert latency.value == 10.0
        assert latency.confidence == "high"
        assert "100" in latency.rationale

    def test_control_loop_at_1khz_implies_1ms_latency(self):
        from embodied_ai_architect.codebase.converter import infer_constraints

        analysis = self._analysis(
            self._kernel(kernel_type="control_loop", invocation_frequency_hz=1000.0)
        )
        latency = next(
            c for c in infer_constraints(analysis).constraints if c.name == "max_latency_ms"
        )
        assert latency.value == 1.0

    def test_total_gflops_implies_power_envelope(self):
        from embodied_ai_architect.codebase.converter import infer_constraints

        analysis = self._analysis(
            self._kernel(
                kernel_type="ml_inference",
                estimated_ops_per_invocation=8.4e9,
            )
        )
        power = next(
            c for c in infer_constraints(analysis).constraints if c.name == "max_power_watts"
        )
        # 8.4 GFLOPS / 2000 = 0.0042 → floored to 1.0 W
        assert power.value >= 1.0
        assert power.confidence == "medium"

    def test_high_gflops_drops_power_confidence(self):
        from embodied_ai_architect.codebase.converter import infer_constraints

        analysis = self._analysis(
            self._kernel(
                kernel_type="ml_inference",
                estimated_ops_per_invocation=200e9,
            )
        )
        power = next(
            c for c in infer_constraints(analysis).constraints if c.name == "max_power_watts"
        )
        assert power.confidence == "low"

    def test_ml_dominant_with_npu_threshold(self):
        from embodied_ai_architect.codebase.converter import infer_constraints

        kernels = [
            self._kernel(
                name=f"ml_{i}",
                kernel_type="ml_inference",
                estimated_ops_per_invocation=2e9,
            )
            for i in range(4)
        ]
        kernels.append(self._kernel(name="other", kernel_type="general_compute"))
        analysis = self._analysis(*kernels)
        suggestions = infer_constraints(analysis)
        hw = next(c for c in suggestions.constraints if c.name == "hardware_class")
        assert hw.value == "npu"
        assert hw.confidence == "high"

    def test_ml_dominant_with_gpu_threshold(self):
        from embodied_ai_architect.codebase.converter import infer_constraints

        analysis = self._analysis(
            self._kernel(
                kernel_type="ml_inference",
                estimated_ops_per_invocation=60e9,
            ),
        )
        suggestions = infer_constraints(analysis)
        hw = next(c for c in suggestions.constraints if c.name == "hardware_class")
        assert hw.value == "gpu"

    def test_signal_processing_at_high_freq_implies_dsp(self):
        from embodied_ai_architect.codebase.converter import infer_constraints

        analysis = self._analysis(
            self._kernel(kernel_type="signal_processing", invocation_frequency_hz=5000.0),
            self._kernel(kernel_type="signal_processing", invocation_frequency_hz=2000.0),
            self._kernel(kernel_type="general_compute"),
        )
        hw_entries = [
            c
            for c in infer_constraints(analysis).constraints
            if c.name == "hardware_class" and c.value == "dsp"
        ]
        assert len(hw_entries) == 1

    def test_io_bound_dominant_flags_memory_bw(self):
        from embodied_ai_architect.codebase.converter import infer_constraints

        analysis = self._analysis(
            self._kernel(kernel_type="io_bound"),
            self._kernel(kernel_type="io_bound"),
            self._kernel(kernel_type="general_compute"),
        )
        flag = next(
            c for c in infer_constraints(analysis).constraints if c.name == "memory_bw_critical"
        )
        assert flag.value is True
        assert flag.confidence == "medium"

    def test_to_design_constraints_kwargs_returns_only_canonical(self):
        from embodied_ai_architect.codebase.converter import infer_constraints

        analysis = self._analysis(
            self._kernel(
                kernel_type="control_loop",
                invocation_frequency_hz=100.0,
            ),
            self._kernel(
                kernel_type="ml_inference",
                estimated_ops_per_invocation=5e9,
                invocation_frequency_hz=10.0,  # 5*10 = 50 GFLOPS → medium confidence
            ),
        )
        suggestions = infer_constraints(analysis)
        # Default min_confidence="high" → only the high-confidence latency
        kwargs_high = suggestions.to_design_constraints_kwargs()
        assert "max_latency_ms" in kwargs_high
        assert "max_power_watts" not in kwargs_high  # power is medium-confidence
        # Lowering the bar pulls in the medium-confidence power suggestion
        kwargs_med = suggestions.to_design_constraints_kwargs(min_confidence="medium")
        assert "max_latency_ms" in kwargs_med
        assert "max_power_watts" in kwargs_med
        # Advisory keys excluded at any confidence level
        assert "hardware_class" not in kwargs_med
        assert "memory_bw_critical" not in kwargs_med

    def test_throughput_uses_invocation_frequency(self):
        """CodeRabbit PR #89: total_gflops must scale by invocation frequency.

        A 10 GFLOP/invocation kernel at 100Hz is 1000 GFLOPS, which crosses
        the GPU threshold. With the bug, it stayed at 10 GFLOPS and only
        triggered the NPU rule.
        """
        from embodied_ai_architect.codebase.converter import infer_constraints

        analysis = self._analysis(
            self._kernel(
                kernel_type="ml_inference",
                estimated_ops_per_invocation=10e9,
                invocation_frequency_hz=100.0,
            )
        )
        suggestions = infer_constraints(analysis)
        hw = next(c for c in suggestions.constraints if c.name == "hardware_class")
        # 10 × 100 = 1000 GFLOPS → above GPU threshold (50)
        assert hw.value == "gpu"

    def test_single_hardware_class_winner(self):
        """CodeRabbit PR #89: only one hardware_class entry is emitted even
        when both ML and DSP heuristics fire on the same analysis."""
        from embodied_ai_architect.codebase.converter import infer_constraints

        # Construct a workload that triggers BOTH the ML rule (NPU) and
        # the DSP rule (signal_processing at high frequency). Without the
        # single-winner fix, two hardware_class entries would land in
        # the list and one would silently win in to_dict().
        analysis = self._analysis(
            *[
                self._kernel(
                    name=f"ml_{i}",
                    kernel_type="ml_inference",
                    estimated_ops_per_invocation=2e9,
                )
                for i in range(4)
            ],
            self._kernel(
                name="sp",
                kernel_type="signal_processing",
                invocation_frequency_hz=5000.0,
            ),
        )
        suggestions = infer_constraints(analysis)
        hw_entries = [c for c in suggestions.constraints if c.name == "hardware_class"]
        assert len(hw_entries) == 1
        # ML rule is high confidence — it must beat the medium-confidence DSP rule
        assert hw_entries[0].confidence == "high"
        assert hw_entries[0].value == "npu"
        # to_dict() round-trips the single winner
        assert suggestions.to_dict()["hardware_class"]["value"] == "npu"

    def test_to_design_constraints_kwargs_feeds_constructor(self):
        """The kwargs subset can be spread directly into a DesignConstraints."""
        from embodied_ai_architect.codebase.converter import infer_constraints
        from embodied_ai_architect.graphs.soc_state import DesignConstraints

        analysis = self._analysis(
            self._kernel(kernel_type="control_loop", invocation_frequency_hz=100.0)
        )
        suggestions = infer_constraints(analysis)
        constraints = DesignConstraints(**suggestions.to_design_constraints_kwargs())
        assert constraints.max_latency_ms == 10.0

    def test_to_dict_renders_all_fields(self):
        from embodied_ai_architect.codebase.converter import infer_constraints

        analysis = self._analysis(
            self._kernel(kernel_type="control_loop", invocation_frequency_hz=100.0)
        )
        d = infer_constraints(analysis).to_dict()
        assert "max_latency_ms" in d
        entry = d["max_latency_ms"]
        assert entry["value"] == 10.0
        assert entry["confidence"] == "high"
        assert "rationale" in entry
