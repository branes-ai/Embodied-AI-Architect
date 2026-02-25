"""Tool definitions for the Embodied AI Architect agent.

Converts existing agents into LLM-callable tools with JSON schemas.
Includes integration with branes-ai/graphs for detailed analysis.
"""

import json
import traceback
from pathlib import Path
from typing import Any, Callable

from embodied_ai_architect.agents.model_analyzer import ModelAnalyzerAgent
from embodied_ai_architect.agents.hardware_profile import HardwareProfileAgent
from embodied_ai_architect.agents.benchmark import BenchmarkAgent

# Import graphs tools (optional dependency)
try:
    from .graphs_tools import (
        get_graphs_tool_definitions,
        create_graphs_tool_executors,
        HAS_GRAPHS,
    )
except ImportError:
    HAS_GRAPHS = False

    def get_graphs_tool_definitions():
        return []

    def create_graphs_tool_executors():
        return {}


# Import architecture analysis tools
try:
    from .architecture_tools import (
        get_architecture_tool_definitions,
        create_architecture_tool_executors,
    )

    HAS_ARCHITECTURE_TOOLS = True
except ImportError:
    HAS_ARCHITECTURE_TOOLS = False

    def get_architecture_tool_definitions():
        return []

    def create_architecture_tool_executors():
        return {}


# Import codebase analysis tools
try:
    from .codebase_tools import (
        get_codebase_tool_definitions,
        create_codebase_tool_executors,
    )

    HAS_CODEBASE_TOOLS = True
except ImportError:
    HAS_CODEBASE_TOOLS = False

    def get_codebase_tool_definitions():
        return []

    def create_codebase_tool_executors():
        return {}


# Import optimization tools (optional, requires numpy)
try:
    from .optimization_tools import (
        get_optimization_tool_definitions,
        create_optimization_tool_executors,
    )

    HAS_MOO = True
except ImportError:
    HAS_MOO = False

    def get_optimization_tool_definitions():
        return []

    def create_optimization_tool_executors():
        return {}


def get_tool_definitions() -> list[dict[str, Any]]:
    """Get tool definitions in Anthropic's tool format.

    Returns:
        List of tool definitions with name, description, and input_schema
    """
    # Base tools from embodied-ai-architect agents
    base_tools = [
        {
            "name": "analyze_model",
            "description": (
                "Analyze a PyTorch model's structure, including layer types, "
                "parameter counts, memory requirements, and computational characteristics. "
                "Use this to understand what kind of model you're working with before "
                "making hardware recommendations."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to the PyTorch model file (.pt or .pth)",
                    }
                },
                "required": ["model_path"],
            },
        },
        {
            "name": "recommend_hardware",
            "description": (
                "Get hardware recommendations for deploying a model based on its "
                "characteristics and user constraints. Returns ranked list of suitable "
                "hardware targets (Jetson, Coral, FPGA, cloud GPU, etc.) with scores "
                "and reasoning. Requires model analysis results from analyze_model."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_analysis": {
                        "type": "object",
                        "description": "Model analysis results from analyze_model tool",
                    },
                    "power_budget_watts": {
                        "type": "number",
                        "description": "Maximum power consumption in watts (optional)",
                    },
                    "latency_target_ms": {
                        "type": "number",
                        "description": "Target inference latency in milliseconds (optional)",
                    },
                    "memory_limit_mb": {
                        "type": "number",
                        "description": "Maximum memory usage in MB (optional)",
                    },
                    "cost_limit_usd": {
                        "type": "number",
                        "description": "Maximum hardware cost in USD (optional)",
                    },
                    "use_case": {
                        "type": "string",
                        "enum": ["edge", "cloud", "mobile", "drone", "robot", "automotive"],
                        "description": "Target deployment use case",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of recommendations to return (default: 5)",
                    },
                },
                "required": ["model_analysis"],
            },
        },
        {
            "name": "run_benchmark",
            "description": (
                "Benchmark a model's inference performance on a specific backend. "
                "Measures latency, throughput, and resource usage. Available backends: "
                "'local' (CPU), 'ssh' (remote machine), 'kubernetes' (K8s cluster)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to the PyTorch model file",
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["local", "ssh", "kubernetes"],
                        "description": "Backend to run benchmark on (default: local)",
                    },
                    "input_shape": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Input tensor shape, e.g. [1, 3, 224, 224] for image",
                    },
                    "iterations": {
                        "type": "integer",
                        "description": "Number of benchmark iterations (default: 100)",
                    },
                    "warmup_iterations": {
                        "type": "integer",
                        "description": "Warmup iterations before measuring (default: 10)",
                    },
                },
                "required": ["model_path"],
            },
        },
        {
            "name": "list_files",
            "description": (
                "List files in a directory. Use this to explore the user's project "
                "and find model files."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to list (default: current directory)",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '*.pt' for PyTorch models)",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "read_file",
            "description": (
                "Read the contents of a text file. Use this to examine configuration "
                "files, logs, or code."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default: 100)",
                    },
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "deploy_model",
            "description": (
                "Deploy a model to an edge device target (Jetson, Coral, etc.) with optional "
                "quantization. Supports INT8 quantization with calibration data for optimal "
                "edge performance. Returns deployment artifacts and validation results. "
                "Requires TensorRT for Jetson targets."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to the model file (.pt, .pth, or .onnx)",
                    },
                    "target": {
                        "type": "string",
                        "enum": ["jetson", "openvino", "coral"],
                        "description": "Deployment target: jetson (TensorRT/NVIDIA), openvino (Intel/AMD x86), coral (Edge TPU, INT8 only). Default: jetson",
                    },
                    "precision": {
                        "type": "string",
                        "enum": ["fp32", "fp16", "int8"],
                        "description": "Target precision (default: int8)",
                    },
                    "input_shape": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Model input shape, e.g. [1, 3, 224, 224]",
                    },
                    "calibration_data": {
                        "type": "string",
                        "description": "Path to calibration dataset directory (required for INT8)",
                    },
                    "calibration_samples": {
                        "type": "integer",
                        "description": "Number of calibration samples (default: 100)",
                    },
                    "calibration_preprocessing": {
                        "type": "string",
                        "enum": ["imagenet", "yolo", "coco", "none"],
                        "description": "Preprocessing for calibration images (default: imagenet)",
                    },
                    "test_data": {
                        "type": "string",
                        "description": "Path to test dataset for validation",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for deployment artifacts (default: ./deployments)",
                    },
                },
                "required": ["model_path", "input_shape"],
            },
        },
    ]

    # Add graphs tools if available (more detailed analysis)
    if HAS_GRAPHS:
        base_tools.extend(get_graphs_tool_definitions())

    # Add architecture analysis tools if available
    if HAS_ARCHITECTURE_TOOLS:
        base_tools.extend(get_architecture_tool_definitions())

    # Add codebase analysis tools if available
    if HAS_CODEBASE_TOOLS:
        base_tools.extend(get_codebase_tool_definitions())

    # Add optimization tools if available
    if HAS_MOO:
        base_tools.extend(get_optimization_tool_definitions())

    return base_tools


def create_tool_executors() -> dict[str, Callable]:
    """Create tool executor functions that wrap the agents.

    Returns:
        Dictionary mapping tool names to executor functions
    """
    # Initialize agents
    model_analyzer = ModelAnalyzerAgent()
    hardware_profiler = HardwareProfileAgent()
    benchmark_agent = BenchmarkAgent()

    def analyze_model(model_path: str) -> str:
        """Execute model analysis."""
        try:
            result = model_analyzer.execute({"model": model_path})
            if result.success:
                return json.dumps(result.data, indent=2, default=str)
            else:
                return f"Error analyzing model: {result.error}"
        except Exception as e:
            return f"Error: {str(e)}\n{traceback.format_exc()}"

    def recommend_hardware(
        model_analysis: dict,
        power_budget_watts: float | None = None,
        latency_target_ms: float | None = None,
        memory_limit_mb: float | None = None,
        cost_limit_usd: float | None = None,
        use_case: str | None = None,
        top_n: int = 5,
    ) -> str:
        """Execute hardware recommendation."""
        try:
            constraints = {}
            if power_budget_watts is not None:
                constraints["power_watts"] = power_budget_watts
            if latency_target_ms is not None:
                constraints["latency_ms"] = latency_target_ms
            if memory_limit_mb is not None:
                constraints["memory_mb"] = memory_limit_mb
            if cost_limit_usd is not None:
                constraints["cost_usd"] = cost_limit_usd

            result = hardware_profiler.execute(
                {
                    "model_analysis": model_analysis,
                    "constraints": constraints,
                    "target_use_case": use_case,
                    "top_n": top_n,
                }
            )

            if result.success:
                return json.dumps(result.data, indent=2, default=str)
            else:
                return f"Error getting hardware recommendations: {result.error}"
        except Exception as e:
            return f"Error: {str(e)}\n{traceback.format_exc()}"

    def run_benchmark(
        model_path: str,
        backend: str = "local",
        input_shape: list[int] | None = None,
        iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> str:
        """Execute benchmark."""
        try:
            result = benchmark_agent.execute(
                {
                    "model": model_path,
                    "backends": [backend],
                    "input_shape": input_shape,
                    "iterations": iterations,
                    "warmup_iterations": warmup_iterations,
                }
            )

            if result.success:
                return json.dumps(result.data, indent=2, default=str)
            else:
                return f"Error running benchmark: {result.error}"
        except Exception as e:
            return f"Error: {str(e)}\n{traceback.format_exc()}"

    def list_files(
        directory: str = ".",
        pattern: str | None = None,
    ) -> str:
        """List files in a directory."""
        try:
            path = Path(directory).expanduser().resolve()
            if not path.exists():
                return f"Directory not found: {directory}"

            if pattern:
                files = list(path.glob(pattern))
            else:
                files = list(path.iterdir())

            # Sort and format
            files.sort()
            result = []
            for f in files[:50]:  # Limit to 50 files
                prefix = "📁 " if f.is_dir() else "📄 "
                size = f.stat().st_size if f.is_file() else 0
                size_str = f" ({size:,} bytes)" if size > 0 else ""
                result.append(f"{prefix}{f.name}{size_str}")

            if len(files) > 50:
                result.append(f"... and {len(files) - 50} more files")

            return "\n".join(result) if result else "Directory is empty"
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def read_file(
        file_path: str,
        max_lines: int = 100,
    ) -> str:
        """Read a text file."""
        try:
            path = Path(file_path).expanduser().resolve()
            if not path.exists():
                return f"File not found: {file_path}"

            if not path.is_file():
                return f"Not a file: {file_path}"

            # Check file size
            size = path.stat().st_size
            if size > 1_000_000:  # 1MB limit
                return f"File too large ({size:,} bytes). Maximum size is 1MB."

            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[:max_lines]

            content = "".join(lines)
            if len(lines) == max_lines:
                content += f"\n... (truncated at {max_lines} lines)"

            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def deploy_model(
        model_path: str,
        input_shape: list[int],
        target: str = "jetson",
        precision: str = "int8",
        calibration_data: str | None = None,
        calibration_samples: int = 100,
        calibration_preprocessing: str = "imagenet",
        test_data: str | None = None,
        output_dir: str = "./deployments",
    ) -> str:
        """Execute model deployment."""
        try:
            from embodied_ai_architect.agents.deployment import DeploymentAgent

            agent = DeploymentAgent()

            input_data = {
                "model": model_path,
                "target": target,
                "precision": precision,
                "input_shape": tuple(input_shape),
                "output_dir": output_dir,
            }

            if calibration_data:
                input_data["calibration_data"] = calibration_data
                input_data["calibration_samples"] = calibration_samples
                input_data["calibration_preprocessing"] = calibration_preprocessing

            if test_data:
                input_data["test_data"] = test_data

            result = agent.execute(input_data)

            if result.success:
                return json.dumps(result.data, indent=2, default=str)
            else:
                return f"Deployment failed: {result.error}"

        except ImportError:
            return (
                "Deployment dependencies not installed. "
                "Install with: pip install embodied-ai-architect[jetson]"
            )
        except Exception as e:
            return f"Error: {str(e)}\n{traceback.format_exc()}"

    executors = {
        "analyze_model": analyze_model,
        "recommend_hardware": recommend_hardware,
        "run_benchmark": run_benchmark,
        "list_files": list_files,
        "read_file": read_file,
        "deploy_model": deploy_model,
    }

    # Add graphs executors if available
    if HAS_GRAPHS:
        executors.update(create_graphs_tool_executors())

    # Add architecture analysis executors if available
    if HAS_ARCHITECTURE_TOOLS:
        executors.update(create_architecture_tool_executors())

    # Add codebase analysis executors if available
    if HAS_CODEBASE_TOOLS:
        executors.update(create_codebase_tool_executors())

    # Add optimization executors if available
    if HAS_MOO:
        executors.update(create_optimization_tool_executors())

    return executors
