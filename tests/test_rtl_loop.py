"""Tests for RTL inner loop (lint -> synthesize -> validate)."""

from embodied_ai_architect.graphs.rtl_loop import RTLLoopResult, run_rtl_loop

SIMPLE_RTL = """
module test_mod (
    input  wire clk,
    input  wire rst,
    input  wire [7:0] a,
    input  wire [7:0] b,
    output reg  [15:0] result
);
    always @(posedge clk) begin
        if (rst)
            result <= 16'd0;
        else
            result <= a * b;
    end
endmodule
"""


class TestRTLLoop:
    def test_rtl_loop_success(self):
        """Run with simple RTL, check success and module_name."""
        result = run_rtl_loop("test_mod", SIMPLE_RTL)
        assert isinstance(result, RTLLoopResult)
        assert result.success is True
        assert result.module_name == "test_mod"

    def test_rtl_loop_has_lint_result(self):
        """Check lint_result dict is present with 'success' key."""
        result = run_rtl_loop("test_mod", SIMPLE_RTL)
        assert isinstance(result.lint_result, dict)
        assert "success" in result.lint_result

    def test_rtl_loop_has_synthesis_result(self):
        """Check synthesis_result has 'success' key."""
        result = run_rtl_loop("test_mod", SIMPLE_RTL)
        assert isinstance(result.synthesis_result, dict)
        assert "success" in result.synthesis_result

    def test_rtl_loop_metrics(self):
        """Check metrics dict has area_cells key."""
        result = run_rtl_loop("test_mod", SIMPLE_RTL)
        assert isinstance(result.metrics, dict)
        assert "area_cells" in result.metrics
        assert result.metrics["area_cells"] > 0

    def test_rtl_loop_records_history(self):
        """History list is non-empty, entries have iteration and stage keys."""
        result = run_rtl_loop("test_mod", SIMPLE_RTL)
        assert isinstance(result.history, list)
        assert len(result.history) > 0
        for entry in result.history:
            assert "iteration" in entry
            assert "stage" in entry
