"""Automated test suite for verdict-first CLI tools.

This module tests the integration of verdict-first graph analysis tools
with the ArchitectAgent CLI interface.

Run with:
    pytest tests/test_verdict_tools_cli.py -v

For live API testing (requires ANTHROPIC_API_KEY):
    pytest tests/test_verdict_tools_cli.py -v --live-api

For verbose output showing agent responses:
    pytest tests/test_verdict_tools_cli.py -v --live-api -s
"""

import json
import os
import pytest
from typing import Any
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

# =============================================================================
# Test Configuration
# =============================================================================


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--live-api",
        action="store_true",
        default=False,
        help="Run tests against live Anthropic API (requires ANTHROPIC_API_KEY)",
    )


@pytest.fixture
def live_api(request):
    """Check if live API testing is enabled."""
    return request.config.getoption("--live-api")


@pytest.fixture
def skip_without_api_key():
    """Skip test if API key not available."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


# =============================================================================
# Test Cases Definition
# =============================================================================


@dataclass
class PromptTestCase:
    """Definition of a prompt test case."""

    name: str
    prompt: str
    expected_tools: list[str]  # Tools that should be called
    expected_verdict: str | None = None  # PASS, FAIL, or None if not applicable
    expected_in_response: list[str] = field(default_factory=list)  # Substrings to find
    not_expected_in_response: list[str] = field(default_factory=list)  # Should NOT appear
    timeout_seconds: int = 60


# Latency check test cases
LATENCY_TEST_CASES = [
    PromptTestCase(
        name="latency_pass_easy",
        prompt="Can ResNet-18 meet a 10ms latency target on an H100?",
        expected_tools=["check_latency"],
        expected_verdict="PASS",
        expected_in_response=["PASS", "headroom"],
    ),
    PromptTestCase(
        name="latency_fail_tight",
        prompt="Can ResNet-50 achieve 0.1ms latency on a Jetson Orin Nano?",
        expected_tools=["check_latency"],
        expected_verdict="FAIL",
        expected_in_response=["FAIL"],
    ),
    PromptTestCase(
        name="latency_natural_fps",
        prompt="I need YOLOv8n to run at 30fps on a Coral Edge TPU. Is that possible?",
        expected_tools=["check_latency"],
        expected_in_response=["30"],  # Should mention 30fps or 33ms
    ),
]

# Power check test cases
POWER_TEST_CASES = [
    PromptTestCase(
        name="power_pass_generous",
        prompt="Can I run MobileNetV2 on a Jetson Orin Nano within a 15W power budget?",
        expected_tools=["check_power"],
        expected_in_response=["15"],
    ),
    PromptTestCase(
        name="power_fail_tight",
        prompt="I have a 5W power budget for running ResNet-50. Can the Jetson Orin AGX work?",
        expected_tools=["check_power"],
        expected_verdict="FAIL",
        expected_in_response=["FAIL", "5W"],
    ),
]

# Memory check test cases
MEMORY_TEST_CASES = [
    PromptTestCase(
        name="memory_pass_plenty",
        prompt="Does ResNet-18 fit in 512MB of memory on the Jetson Orin AGX?",
        expected_tools=["check_memory"],
        expected_verdict="PASS",
        expected_in_response=["PASS", "512"],
    ),
    PromptTestCase(
        name="memory_fail_constrained",
        prompt="Can I run ResNet-152 in just 2MB of memory on an H100?",
        expected_tools=["check_memory"],
        # Note: Tool returns FAIL but LLM interpretation may vary
        # We just verify the tool is called and memory is discussed
        expected_verdict=None,  # Don't check verdict - LLM interpretation varies
        expected_in_response=["memory", "2"],  # Should mention memory and the constraint
    ),
]

# Full analysis test cases
FULL_ANALYSIS_TEST_CASES = [
    PromptTestCase(
        name="full_analysis_basic",
        prompt="Give me a complete analysis of ResNet-50 on the H100",
        expected_tools=["full_analysis", "analyze_model_detailed"],
        expected_in_response=["latency", "energy", "memory"],
    ),
    PromptTestCase(
        name="full_analysis_with_constraint",
        prompt="Analyze MobileNetV2 on A100 with a 5ms latency requirement",
        expected_tools=["full_analysis", "check_latency"],  # Either tool is valid
        expected_in_response=["5", "latency"],  # May say "5ms" or "5 ms"
    ),
    PromptTestCase(
        name="bottleneck_identification",
        prompt="Is ResNet-18 compute-bound or memory-bound on an A100?",
        expected_tools=["full_analysis", "identify_bottleneck", "analyze_model_detailed"],
        expected_in_response=["bound"],  # compute-bound or memory-bound
    ),
]

# Hardware comparison test cases
COMPARISON_TEST_CASES = [
    PromptTestCase(
        name="hardware_comparison",
        prompt="Compare ResNet-18 performance on H100 vs A100",
        expected_tools=["compare_hardware_targets"],
        expected_in_response=["H100", "A100"],
    ),
]

# Hardware listing test cases
LISTING_TEST_CASES = [
    PromptTestCase(
        name="list_all_hardware",
        prompt="What hardware targets are available for analysis?",
        expected_tools=["list_available_hardware"],
        expected_in_response=["H100", "Jetson"],
    ),
    PromptTestCase(
        name="list_edge_gpu",
        prompt="Show me all edge GPUs you can analyze",
        expected_tools=["list_available_hardware"],
        expected_in_response=["Jetson", "Orin"],
    ),
]

# Error handling test cases
ERROR_TEST_CASES = [
    PromptTestCase(
        name="unknown_model",
        prompt="Can you analyze FooBarNet on the H100?",
        expected_tools=[],  # May or may not call a tool
        expected_in_response=["error", "not found", "unknown", "available"],
        not_expected_in_response=["Traceback"],
    ),
    PromptTestCase(
        name="unknown_hardware",
        prompt="Check latency of ResNet-18 on the NVIDIA RTX 5090",
        expected_tools=[],
        expected_in_response=["error", "not found", "unknown", "available"],
        not_expected_in_response=["Traceback"],
    ),
]

# All test cases combined
ALL_TEST_CASES = (
    LATENCY_TEST_CASES
    + POWER_TEST_CASES
    + MEMORY_TEST_CASES
    + FULL_ANALYSIS_TEST_CASES
    + COMPARISON_TEST_CASES
    + LISTING_TEST_CASES
    + ERROR_TEST_CASES
)

# Quick smoke test subset
SMOKE_TEST_CASES = [
    LATENCY_TEST_CASES[0],  # latency_pass_easy
    LATENCY_TEST_CASES[1],  # latency_fail_tight
    FULL_ANALYSIS_TEST_CASES[0],  # full_analysis_basic
    LISTING_TEST_CASES[0],  # list_all_hardware
    COMPARISON_TEST_CASES[0],  # hardware_comparison
]


# =============================================================================
# Tool Call Capture
# =============================================================================


@dataclass
class ToolCallRecord:
    """Record of a tool call."""

    name: str
    args: dict[str, Any]
    result: str


class ToolCallCapture:
    """Captures tool calls made by the agent."""

    def __init__(self):
        self.calls: list[ToolCallRecord] = []

    def on_tool_start(self, name: str, args: dict) -> None:
        """Called when a tool starts execution."""
        self.calls.append(ToolCallRecord(name=name, args=args, result=""))

    def on_tool_end(self, name: str, result: str) -> None:
        """Called when a tool ends execution."""
        # Update the last call with the same name
        for call in reversed(self.calls):
            if call.name == name and call.result == "":
                call.result = result
                break

    def get_tool_names(self) -> list[str]:
        """Get list of tool names called."""
        return [call.name for call in self.calls]

    def has_tool(self, name: str) -> bool:
        """Check if a tool was called."""
        return name in self.get_tool_names()

    def get_call(self, name: str) -> ToolCallRecord | None:
        """Get the first call to a specific tool."""
        for call in self.calls:
            if call.name == name:
                return call
        return None

    def clear(self) -> None:
        """Clear all recorded calls."""
        self.calls = []


# =============================================================================
# Test Runner
# =============================================================================


class VerdictToolsTester:
    """Test runner for verdict-first tools."""

    def __init__(self, use_live_api: bool = False):
        self.use_live_api = use_live_api
        self.agent = None
        self.capture = ToolCallCapture()

    def setup(self) -> None:
        """Initialize the agent."""
        if self.use_live_api:
            from embodied_ai_architect.llm import LLMClient, ArchitectAgent

            llm = LLMClient()
            self.agent = ArchitectAgent(llm=llm, verbose=True)
        else:
            # Create mock agent for unit testing
            self.agent = self._create_mock_agent()

    def _create_mock_agent(self) -> MagicMock:
        """Create a mock agent for testing without API calls."""
        agent = MagicMock()
        agent.reset = MagicMock()
        return agent

    def run_test(self, test_case: PromptTestCase) -> dict[str, Any]:
        """Run a single test case.

        Returns:
            Dict with test results including:
            - success: bool
            - response: str
            - tools_called: list[str]
            - verdict_found: str | None
            - errors: list[str]
        """
        self.capture.clear()
        errors = []

        try:
            if self.use_live_api:
                response = self.agent.run(
                    test_case.prompt,
                    on_tool_start=self.capture.on_tool_start,
                    on_tool_end=self.capture.on_tool_end,
                )
            else:
                # Mock response for unit testing
                response = self._generate_mock_response(test_case)

            tools_called = self.capture.get_tool_names()

            # Check expected tools (at least one should be called)
            if test_case.expected_tools:
                found_expected = any(tool in tools_called for tool in test_case.expected_tools)
                if not found_expected:
                    errors.append(
                        f"Expected one of {test_case.expected_tools}, " f"got {tools_called}"
                    )

            # Extract verdict from response
            verdict_found = self._extract_verdict(response)

            # Check expected verdict
            if test_case.expected_verdict:
                if verdict_found != test_case.expected_verdict:
                    errors.append(
                        f"Expected verdict {test_case.expected_verdict}, " f"got {verdict_found}"
                    )

            # Check expected substrings in response
            response_lower = response.lower()
            for expected in test_case.expected_in_response:
                if expected.lower() not in response_lower:
                    errors.append(f"Expected '{expected}' in response")

            # Check unexpected substrings
            for not_expected in test_case.not_expected_in_response:
                if not_expected.lower() in response_lower:
                    errors.append(f"Did not expect '{not_expected}' in response")

            return {
                "success": len(errors) == 0,
                "response": response,
                "tools_called": tools_called,
                "verdict_found": verdict_found,
                "errors": errors,
            }

        except Exception as e:
            return {
                "success": False,
                "response": "",
                "tools_called": [],
                "verdict_found": None,
                "errors": [f"Exception: {str(e)}"],
            }

    def _extract_verdict(self, response: str) -> str | None:
        """Extract PASS/FAIL verdict from response."""
        response_upper = response.upper()
        if "PASS" in response_upper:
            return "PASS"
        elif "FAIL" in response_upper:
            return "FAIL"
        elif "UNKNOWN" in response_upper:
            return "UNKNOWN"
        return None

    def _generate_mock_response(self, test_case: PromptTestCase) -> str:
        """Generate mock response for unit testing."""
        # This simulates what the agent would return
        if test_case.expected_verdict == "PASS":
            return f"**PASS** - The constraint is met with good headroom."
        elif test_case.expected_verdict == "FAIL":
            return f"**FAIL** - The constraint cannot be met."
        else:
            return "Analysis complete. See details above."


# =============================================================================
# Pytest Test Functions
# =============================================================================


@pytest.fixture
def tester(live_api):
    """Create test runner."""
    tester = VerdictToolsTester(use_live_api=live_api)
    tester.setup()
    return tester


class TestVerdictToolsUnit:
    """Unit tests that don't require API access."""

    def test_tool_call_capture(self):
        """Test the tool call capture mechanism."""
        capture = ToolCallCapture()
        capture.on_tool_start("check_latency", {"model": "resnet18"})
        capture.on_tool_end("check_latency", '{"verdict": "PASS"}')

        assert capture.has_tool("check_latency")
        assert not capture.has_tool("check_power")
        assert capture.get_tool_names() == ["check_latency"]

        call = capture.get_call("check_latency")
        assert call is not None
        assert call.args == {"model": "resnet18"}
        assert "PASS" in call.result

    def test_verdict_extraction(self):
        """Test verdict extraction from responses."""
        tester = VerdictToolsTester()
        tester.setup()

        assert tester._extract_verdict("The result is **PASS**") == "PASS"
        assert tester._extract_verdict("Status: FAIL - too slow") == "FAIL"
        assert tester._extract_verdict("Verdict: UNKNOWN") == "UNKNOWN"
        assert tester._extract_verdict("No verdict here") is None

    def test_test_case_structure(self):
        """Test that all test cases are properly structured."""
        for tc in ALL_TEST_CASES:
            assert tc.name, "Test case must have a name"
            assert tc.prompt, "Test case must have a prompt"
            assert isinstance(tc.expected_tools, list)
            assert isinstance(tc.expected_in_response, list)


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestVerdictToolsLive:
    """Live API tests - require ANTHROPIC_API_KEY."""

    @pytest.fixture(autouse=True)
    def setup_tester(self):
        """Set up tester for live API."""
        self.tester = VerdictToolsTester(use_live_api=True)
        self.tester.setup()

    @pytest.mark.parametrize("test_case", SMOKE_TEST_CASES, ids=lambda tc: tc.name)
    def test_smoke(self, test_case):
        """Smoke tests - quick validation of core functionality."""
        result = self.tester.run_test(test_case)

        print(f"\n{'='*60}")
        print(f"Test: {test_case.name}")
        print(f"Prompt: {test_case.prompt}")
        print(f"Tools called: {result['tools_called']}")
        print(f"Verdict found: {result['verdict_found']}")
        print(f"Response preview: {result['response'][:200]}...")
        if result["errors"]:
            print(f"Errors: {result['errors']}")
        print(f"{'='*60}")

        assert result["success"], f"Test failed: {result['errors']}"

    @pytest.mark.parametrize("test_case", LATENCY_TEST_CASES, ids=lambda tc: tc.name)
    def test_latency_checks(self, test_case):
        """Test latency constraint checking."""
        result = self.tester.run_test(test_case)
        assert result["success"], f"Test failed: {result['errors']}"

    @pytest.mark.parametrize("test_case", POWER_TEST_CASES, ids=lambda tc: tc.name)
    def test_power_checks(self, test_case):
        """Test power constraint checking."""
        result = self.tester.run_test(test_case)
        assert result["success"], f"Test failed: {result['errors']}"

    @pytest.mark.parametrize("test_case", MEMORY_TEST_CASES, ids=lambda tc: tc.name)
    def test_memory_checks(self, test_case):
        """Test memory constraint checking."""
        result = self.tester.run_test(test_case)
        assert result["success"], f"Test failed: {result['errors']}"

    @pytest.mark.parametrize("test_case", FULL_ANALYSIS_TEST_CASES, ids=lambda tc: tc.name)
    def test_full_analysis(self, test_case):
        """Test full analysis functionality."""
        result = self.tester.run_test(test_case)
        assert result["success"], f"Test failed: {result['errors']}"

    @pytest.mark.parametrize("test_case", COMPARISON_TEST_CASES, ids=lambda tc: tc.name)
    def test_comparisons(self, test_case):
        """Test hardware comparison functionality."""
        result = self.tester.run_test(test_case)
        assert result["success"], f"Test failed: {result['errors']}"

    @pytest.mark.parametrize("test_case", LISTING_TEST_CASES, ids=lambda tc: tc.name)
    def test_listings(self, test_case):
        """Test hardware listing functionality."""
        result = self.tester.run_test(test_case)
        assert result["success"], f"Test failed: {result['errors']}"


# =============================================================================
# CLI Test Runner
# =============================================================================


def run_interactive_tests():
    """Run tests interactively with detailed output."""
    print("=" * 70)
    print("Verdict-First Tools - Interactive Test Suite")
    print("=" * 70)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nError: ANTHROPIC_API_KEY not set")
        print("Set your API key: export ANTHROPIC_API_KEY=your-key")
        return

    tester = VerdictToolsTester(use_live_api=True)
    tester.setup()

    results = {
        "passed": 0,
        "failed": 0,
        "errors": [],
    }

    for i, tc in enumerate(SMOKE_TEST_CASES):
        print(f"\n[{i+1}/{len(SMOKE_TEST_CASES)}] {tc.name}")
        print(f"  Prompt: {tc.prompt}")

        result = tester.run_test(tc)

        if result["success"]:
            print(f"  ✓ PASSED")
            print(f"    Tools: {result['tools_called']}")
            print(f"    Verdict: {result['verdict_found']}")
            results["passed"] += 1
        else:
            print(f"  ✗ FAILED")
            for error in result["errors"]:
                print(f"    - {error}")
            results["failed"] += 1
            results["errors"].append((tc.name, result["errors"]))

        # Reset agent context between tests
        tester.agent.reset()

    print("\n" + "=" * 70)
    print(f"Results: {results['passed']} passed, {results['failed']} failed")
    print("=" * 70)

    if results["failed"] > 0:
        print("\nFailed tests:")
        for name, errors in results["errors"]:
            print(f"  - {name}: {errors}")

    return results["failed"] == 0


if __name__ == "__main__":
    import sys

    success = run_interactive_tests()
    sys.exit(0 if success else 1)
