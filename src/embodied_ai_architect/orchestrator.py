"""Orchestrator - Coordinates agent execution in the Embodied AI Architect system."""

from typing import Dict, Any, List
from enum import Enum
from pydantic import BaseModel

from .agents.base import BaseAgent, AgentResult


class WorkflowStatus(str, Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowResult(BaseModel):
    """Result of a complete workflow execution."""

    status: WorkflowStatus
    agent_results: Dict[str, AgentResult]
    summary: Dict[str, Any] = {}
    error: str | None = None


class Orchestrator:
    """Orchestrates agent execution and manages workflow state.

    The Orchestrator is the main entry point for processing Embodied AI models.
    It coordinates multiple agents and manages the overall workflow.
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflow_history: List[WorkflowResult] = []

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent: Agent instance to register
        """
        self.agents[agent.name] = agent
        print(f"Registered agent: {agent.name}")

    def process(self, request: Dict[str, Any]) -> WorkflowResult:
        """Process a user request through the agent pipeline.

        Args:
            request: Dictionary containing:
                - 'model': Model to analyze (PyTorch model or path)
                - 'targets': List of hardware targets (optional)
                - 'constraints': Performance constraints (optional)

        Returns:
            WorkflowResult containing results from all agents
        """
        print(f"\n{'='*60}")
        print("🚀 Starting Embodied AI Architect Workflow")
        print(f"{'='*60}\n")

        agent_results = {}
        workflow_status = WorkflowStatus.RUNNING

        try:
            # Step 1: Model Analysis
            if "ModelAnalyzer" in self.agents:
                print("📊 Running Model Analysis...")
                model_analyzer = self.agents["ModelAnalyzer"]
                result = model_analyzer.execute({"model": request.get("model")})
                agent_results["ModelAnalyzer"] = result

                if not result.success:
                    workflow_status = WorkflowStatus.FAILED
                    return WorkflowResult(
                        status=workflow_status,
                        agent_results=agent_results,
                        error=f"Model analysis failed: {result.error}",
                    )

                print("✓ Model Analysis completed")
                self._print_model_summary(result.data)

            # Step 2: Hardware Profiling
            if "HardwareProfile" in self.agents:
                print("\n🖥️  Running Hardware Profiling...")
                hw_agent = self.agents["HardwareProfile"]
                result = hw_agent.execute(
                    {
                        "model_analysis": agent_results["ModelAnalyzer"].data,
                        "constraints": request.get("constraints", {}),
                        "target_use_case": request.get("target_use_case"),
                        "top_n": request.get("top_n_hardware", 5),
                    }
                )
                agent_results["HardwareProfile"] = result

                if not result.success:
                    workflow_status = WorkflowStatus.FAILED
                    return WorkflowResult(
                        status=workflow_status,
                        agent_results=agent_results,
                        error=f"Hardware profiling failed: {result.error}",
                    )

                print("✓ Hardware Profiling completed")
                self._print_hardware_summary(result.data)

            # Step 3: Benchmarking
            if "Benchmark" in self.agents:
                print("\n⚡ Running Benchmarks...")
                benchmark_agent = self.agents["Benchmark"]
                result = benchmark_agent.execute(
                    {
                        "model": request.get("model"),
                        "input_shape": request.get("input_shape"),
                        "iterations": request.get("iterations", 100),
                        "warmup_iterations": request.get("warmup_iterations", 10),
                        "backends": request.get("backends"),
                    }
                )
                agent_results["Benchmark"] = result

                if not result.success:
                    workflow_status = WorkflowStatus.FAILED
                    return WorkflowResult(
                        status=workflow_status,
                        agent_results=agent_results,
                        error=f"Benchmark failed: {result.error}",
                    )

                print("✓ Benchmarks completed")
                self._print_benchmark_summary(result.data)

            # Step 4: Report Synthesis
            if "ReportSynthesis" in self.agents:
                print("\n📄 Generating Report...")
                report_agent = self.agents["ReportSynthesis"]
                result = report_agent.execute(
                    {
                        "workflow_id": request.get("workflow_id"),
                        "agent_results": agent_results,
                        "request": request,
                        "timestamp": request.get("timestamp", ""),
                    }
                )
                agent_results["ReportSynthesis"] = result

                if not result.success:
                    print(f"  Warning: Report generation failed: {result.error}")
                else:
                    print("✓ Report generated")
                    print(f"  View report: {result.data.get('report_html')}")

            # Future: Add more agent steps here
            # - Code transformation
            # - Deployment

            workflow_status = WorkflowStatus.COMPLETED
            summary = self._generate_summary(agent_results)

            result = WorkflowResult(
                status=workflow_status, agent_results=agent_results, summary=summary
            )

            self.workflow_history.append(result)
            print(f"\n{'='*60}")
            print("✅ Workflow completed successfully")
            print(f"{'='*60}\n")

            return result

        except Exception as e:
            workflow_status = WorkflowStatus.FAILED
            error_msg = f"Workflow error: {str(e)}"
            print(f"\n❌ {error_msg}\n")

            return WorkflowResult(
                status=workflow_status, agent_results=agent_results, error=error_msg
            )

    def _generate_summary(self, agent_results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """Generate a summary of the workflow execution.

        Args:
            agent_results: Results from all executed agents

        Returns:
            Dictionary containing workflow summary
        """
        summary = {
            "total_agents_executed": len(agent_results),
            "successful_agents": sum(1 for r in agent_results.values() if r.success),
            "failed_agents": sum(1 for r in agent_results.values() if not r.success),
        }

        # Add model-specific summary if available
        if "ModelAnalyzer" in agent_results:
            model_data = agent_results["ModelAnalyzer"].data
            summary["model_summary"] = {
                "type": model_data.get("model_type"),
                "total_parameters": model_data.get("total_parameters"),
                "total_layers": model_data.get("total_layers"),
            }

        return summary

    def _print_model_summary(self, analysis: Dict[str, Any]) -> None:
        """Pretty print model analysis summary.

        Args:
            analysis: Model analysis data
        """
        print(f"\n  Model Type: {analysis.get('model_type')}")
        print(f"  Total Parameters: {analysis.get('total_parameters'):,}")
        print(f"  Trainable Parameters: {analysis.get('trainable_parameters'):,}")
        print(f"  Total Layers: {analysis.get('total_layers')}")

        layer_types = analysis.get("layer_type_counts", {})
        if layer_types:
            print("\n  Layer Types:")
            for layer_type, count in sorted(layer_types.items(), key=lambda x: -x[1])[:5]:
                print(f"    - {layer_type}: {count}")

    def _print_benchmark_summary(self, benchmark_data: Dict[str, Any]) -> None:
        """Pretty print benchmark results summary.

        Args:
            benchmark_data: Benchmark data with results from all backends
        """
        benchmarks = benchmark_data.get("benchmarks", {})
        summary = benchmark_data.get("summary", {})

        if not benchmarks:
            print("\n  No benchmark results available")
            return

        print("\n  Benchmark Results:")
        for backend_name, result in benchmarks.items():
            print(f"\n    {backend_name.upper()}:")
            print(f"      Mean Latency: {result.get('mean_latency_ms', 0):.3f} ms")
            print(f"      Std Dev: {result.get('std_latency_ms', 0):.3f} ms")
            print(
                f"      Min/Max: {result.get('min_latency_ms', 0):.3f} / {result.get('max_latency_ms', 0):.3f} ms"
            )

            throughput = result.get("throughput_samples_per_sec")
            if throughput is not None:
                print(f"      Throughput: {throughput:.2f} samples/sec")

        if summary.get("fastest_backend"):
            print(
                f"\n  ⭐ Fastest: {summary['fastest_backend']} ({summary['fastest_latency_ms']:.3f} ms)"
            )

    def _print_hardware_summary(self, hardware_data: Dict[str, Any]) -> None:
        """Pretty print hardware profiling summary.

        Args:
            hardware_data: Hardware profiling data with recommendations
        """
        recommendations = hardware_data.get("recommendations", [])
        model_chars = hardware_data.get("model_characteristics", {})

        if not recommendations:
            print("\n  No hardware recommendations available")
            return

        print("\n  Model Characteristics:")
        print(f"    Parameters: {model_chars.get('parameters', 0):,}")
        print(f"    Estimated Memory: {model_chars.get('estimated_memory_mb', 0):.1f} MB")
        print(f"    Operations: {', '.join(model_chars.get('operation_types', []))}")

        print("\n  Top Hardware Recommendations:")
        for rec in recommendations[:3]:  # Show top 3
            print(f"\n    #{rec['rank']}: {rec['name']} ({rec['vendor']})")
            print(f"      Score: {rec['score']:.1f}/100")
            print(f"      Type: {rec['type'].upper()}")
            if rec.get("cost_usd"):
                print(f"      Cost: ${rec['cost_usd']:,}")
            print(f"      Power: {rec['power_watts']}W")

            # Show top reasons
            if rec.get("reasons"):
                print("      Reasons:")
                for reason in rec["reasons"][:3]:
                    print(f"        • {reason}")

            # Show warnings if any
            if rec.get("warnings"):
                print("      ⚠️  Warnings:")
                for warning in rec["warnings"]:
                    print(f"        • {warning}")

            # Show estimated performance
            if rec.get("estimated_performance"):
                est_perf = rec["estimated_performance"]
                if "estimated_latency_ms" in est_perf:
                    print(f"      Est. Latency: {est_perf['estimated_latency_ms']:.2f} ms")

    def get_agent(self, name: str) -> BaseAgent | None:
        """Get a registered agent by name.

        Args:
            name: Name of the agent

        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agent names.

        Returns:
            List of agent names
        """
        return list(self.agents.keys())
