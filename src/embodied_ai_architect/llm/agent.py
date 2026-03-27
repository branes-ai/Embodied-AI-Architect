"""Agentic loop for the Embodied AI Architect.

This module implements a Claude Code-style interactive agent that can
reason about user requests and call tools to accomplish tasks.
"""

from typing import Any, Callable

from .client import LLMClient, LLMResponse
from .tools import get_tool_definitions, create_tool_executors

SYSTEM_PROMPT = """\
You are the Embodied AI Architect, an expert assistant for designing and deploying
AI systems on embedded and edge hardware. You help users:

- Analyze neural network models to understand their structure and requirements
- Recommend optimal hardware targets (Jetson, Coral Edge TPU, FPGA, cloud GPU, etc.)
- Benchmark model performance on different backends
- Optimize deployments for constraints like power, latency, and memory

## Your Approach

When a user asks about deploying a model:
1. If they have specific constraints (latency, power, memory), use the verdict-first
   tools (check_latency, check_power, check_memory) for clear PASS/FAIL answers
2. For comprehensive analysis, use full_analysis with optional constraint checking
3. For hardware comparison, use compare_hardware_targets
4. Offer to run_benchmark if they want real-world measurements on actual hardware

## Verdict-First Tools (Preferred for Constraint Checking)

When users ask "can model X meet Y constraint on hardware Z?", use these tools:

- **check_latency**: Returns PASS/FAIL for latency targets (e.g., "under 10ms")
- **check_power**: Returns PASS/FAIL for power budgets (e.g., "under 15W")
- **check_memory**: Returns PASS/FAIL for memory limits (e.g., "fits in 512MB")
- **full_analysis**: Comprehensive analysis with optional constraint, returns detailed
  metrics including roofline analysis, energy breakdown, and memory footprint

These tools return a verdict (PASS/FAIL), confidence level (high/medium/low),
and actionable suggestions when constraints are not met.

## Model and Hardware Names

Common models: resnet18, resnet50, mobilenet_v2, mobilenet_v3_small, efficientnet_b0,
vit_b_16, yolov8n, yolov8s, yolov8m

Common hardware: H100-SXM5-80GB, A100-SXM4-80GB, Jetson-Orin-AGX, Jetson-Orin-Nano,
TPU-v4, Coral-Edge-TPU, Hailo-8, TDA4VM

Use list_available_hardware to see all options.

## Communication Style

- Be direct and technical - users are engineers
- Lead with the verdict (PASS/FAIL) when checking constraints
- Cite specific numbers from tool results (latency in ms, power in W, memory in MB)
- When a constraint fails, explain by how much and suggest alternatives
- If you don't have enough information, ask clarifying questions
- When tools return errors, explain what went wrong and suggest fixes

## Available Tools

You have access to tools for:
- Verdict-first analysis (check_latency, check_power, check_memory, full_analysis)
- Hardware comparison (compare_hardware_targets, identify_bottleneck)
- Model analysis (analyze_model, analyze_model_detailed)
- Hardware discovery (list_available_hardware, estimate_power_consumption)
- Benchmarking (run_benchmark)
- File exploration (list_files, read_file)
- Architecture analysis (list_architectures, show_architecture, analyze_architecture, check_scheduling)

## Architecture Analysis Tools

For complete embodied AI systems (not just individual models):
- **list_architectures**: List available software architectures (drone perception, ADAS, etc.)
- **show_architecture**: View architecture details including operators, dataflow, and variants
- **analyze_architecture**: End-to-end analysis of an architecture on hardware (latency, power, memory)
- **check_scheduling**: Verify multi-rate scheduling feasibility (can all operators meet their target rates?)

## Optimization Tools (Phase 4)

When an architecture doesn't meet requirements, use these tools to find solutions:
- **identify_bottleneck**: Find the operator limiting throughput and classify why (compute/memory/IO bound)
- **suggest_optimizations**: Get actionable recommendations (operator swaps, quantization, hardware upgrades)
- **compare_variants**: Compare architecture variants to find the best fit for a hardware target

Example workflow:
1. analyze_architecture → FAIL (52ms exceeds 33ms target)
2. identify_bottleneck → "yolo_detector_s takes 38ms (73% of total), compute-bound"
3. suggest_optimizations → "Replace with yolo_detector_n to save 20ms"
4. compare_variants → "Edge variant meets target at 28ms"

## Benchmarking Tools (Phase 5)

For measuring actual performance (simulated or real):
- **benchmark_architecture**: Run benchmark with per-operator timing breakdown
  - Returns measured latency, throughput, and statistical metrics (p50, p95, p99)
  - Compares measured vs estimated performance
  - Identifies measured bottleneck with contribution percentage

Example benchmark output:
- Measured: 28.3ms ± 2.1ms (estimated: 26.5ms, error: +6.8%)
- Throughput: 35.3 FPS
- Bottleneck: detector (19.2ms, 68% of total)
- Per-operator breakdown with p95/p99 latencies

Use these tools to gather data before making recommendations. Don't guess -
verify with actual analysis when possible.

## Interactive SoC Design (Phase 6)

For full SoC design with human-in-the-loop review:
- **design_soc**: Start an interactive design session. The planner decomposes the goal
  into a task graph (DAG of specialist tasks), then pauses for human review.
- **review_plan**: Approve, modify, or reject the plan. You can add/remove tasks,
  reassign agents, reorder dependencies, or override constraints.
- **steer_optimization**: At each optimization iteration, see constraint slackness
  (how tight/slack each constraint is), optimization trajectory, Pareto frontier,
  and strategy analysis. Direct the optimizer by focusing on a specific objective,
  forcing a strategy, or relaxing/tightening constraints.
- **show_optimization_status**: View the current optimization state without steering.
- **show_plan**: View the current task graph plan.

The key value of this flow is transparency: the human architect sees every decision,
every trade-off, and every Pareto point — building intuition about where to take the
design. Never just present a final answer without showing the journey.

Example interactive design workflow:
1. design_soc → shows task graph for review
2. review_plan(decision="approve") → starts execution
3. (optimization loop runs, pauses at each iteration)
4. steer_optimization(decision="redirect", focus_objective="power") → focus on power
5. steer_optimization(decision="accept") → take current design
"""


class ArchitectAgent:
    """Interactive agent for Embodied AI architecture design.

    Implements a tool-use loop similar to Claude Code, where the LLM
    can call tools and reason about results.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: dict[str, Callable] | None = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """Initialize the agent.

        Args:
            llm: LLM client (creates default if not provided)
            tools: Tool executors (creates defaults if not provided)
            max_iterations: Maximum tool-use iterations per turn
            verbose: Print detailed execution info
        """
        self.llm = llm or LLMClient()
        self.tools = tools or create_tool_executors()
        self.tool_definitions = get_tool_definitions()
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.messages: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Clear conversation history."""
        self.messages = []

    def run(
        self,
        user_input: str,
        on_tool_start: Callable[[str, dict], None] | None = None,
        on_tool_end: Callable[[str, str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> str:
        """Process user input and return response.

        Implements the agentic loop:
        1. Send user message to LLM
        2. If LLM requests tool calls, execute them
        3. Send tool results back to LLM
        4. Repeat until LLM produces final response

        Args:
            user_input: User's message
            on_tool_start: Callback when tool execution starts (name, args)
            on_tool_end: Callback when tool execution ends (name, result)
            on_thinking: Callback for LLM's intermediate thoughts

        Returns:
            Final response text from the LLM
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_input})

        iterations = 0
        while iterations < self.max_iterations:
            iterations += 1

            # Get LLM response
            response = self.llm.chat(
                messages=self.messages,
                tools=self.tool_definitions,
                system=SYSTEM_PROMPT,
            )

            # If there's text and we're thinking, emit it
            if response.text and on_thinking and response.has_tool_calls:
                on_thinking(response.text)

            if response.has_tool_calls:
                # Add assistant message with tool use
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": self._format_tool_use_content(response),
                    }
                )

                # Execute each tool and collect results
                tool_results = []
                for tool_call in response.tool_calls:
                    if on_tool_start:
                        on_tool_start(tool_call.name, tool_call.args)

                    result = self._execute_tool(tool_call.name, tool_call.args)

                    if on_tool_end:
                        on_tool_end(tool_call.name, result)

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": result,
                        }
                    )

                # Add tool results
                self.messages.append(
                    {
                        "role": "user",
                        "content": tool_results,
                    }
                )
            else:
                # No tool calls - we have the final response
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": response.text,
                    }
                )
                return response.text

        # Hit max iterations
        return (
            "I've reached the maximum number of tool calls for this turn. "
            "Please let me know if you'd like me to continue."
        )

    def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool and return its result.

        Args:
            name: Tool name
            args: Tool arguments

        Returns:
            Tool result as string
        """
        if name not in self.tools:
            return f"Error: Unknown tool '{name}'"

        try:
            executor = self.tools[name]
            result = executor(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

    def _format_tool_use_content(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Format response content including tool use blocks.

        Args:
            response: LLM response

        Returns:
            Content list for message
        """
        content = []

        if response.text:
            content.append({"type": "text", "text": response.text})

        for tool_call in response.tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.args,
                }
            )

        return content

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for debugging.

        Returns:
            Formatted conversation summary
        """
        lines = []
        for msg in self.messages:
            role = msg["role"].upper()
            content = msg["content"]

            if isinstance(content, str):
                preview = content[:100] + "..." if len(content) > 100 else content
                lines.append(f"{role}: {preview}")
            elif isinstance(content, list):
                # Tool use or tool results
                types = [c.get("type", "unknown") for c in content]
                lines.append(f"{role}: [{', '.join(types)}]")

        return "\n".join(lines)
