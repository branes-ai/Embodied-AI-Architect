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
                        error=f"Model analysis failed: {result.error}"
                    )

                print(f"✓ Model Analysis completed")
                self._print_model_summary(result.data)

            # Future: Add more agent steps here
            # - Hardware profiling
            # - Code transformation
            # - Benchmarking
            # - Report generation

            workflow_status = WorkflowStatus.COMPLETED
            summary = self._generate_summary(agent_results)

            result = WorkflowResult(
                status=workflow_status,
                agent_results=agent_results,
                summary=summary
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
                status=workflow_status,
                agent_results=agent_results,
                error=error_msg
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
            "failed_agents": sum(1 for r in agent_results.values() if not r.success)
        }

        # Add model-specific summary if available
        if "ModelAnalyzer" in agent_results:
            model_data = agent_results["ModelAnalyzer"].data
            summary["model_summary"] = {
                "type": model_data.get("model_type"),
                "total_parameters": model_data.get("total_parameters"),
                "total_layers": model_data.get("total_layers")
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

        layer_types = analysis.get('layer_type_counts', {})
        if layer_types:
            print(f"\n  Layer Types:")
            for layer_type, count in sorted(layer_types.items(), key=lambda x: -x[1])[:5]:
                print(f"    - {layer_type}: {count}")

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
