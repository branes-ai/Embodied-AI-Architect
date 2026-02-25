"""Tests for EDA tool wrappers (all use mock fallbacks)."""

from embodied_ai_architect.graphs.eda_tools import (
    RTLLintTool,
    RTLSynthesisTool,
    SimulationTool,
    EDAToolchain,
)

SIMPLE_RTL = """\
module counter #(
    parameter WIDTH = 8
)(
    input  logic clk,
    input  logic rst_n,
    output logic [WIDTH-1:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= '0;
        else
            count <= count + 1;
    end
endmodule
"""

SIMPLE_TB = """\
module counter_tb;
    logic clk = 0;
    logic rst_n;
    logic [7:0] count;

    counter dut(.clk(clk), .rst_n(rst_n), .count(count));

    always #5 clk = ~clk;

    initial begin
        rst_n = 0;
        #20 rst_n = 1;
        #100;
        if (count > 0) $display("PASS");
        else $display("FAIL");
        $finish;
    end
endmodule
"""


class TestRTLLintTool:
    def test_lint_valid_rtl(self, tmp_path):
        tool = RTLLintTool(tmp_path)
        result = tool.run(SIMPLE_RTL)
        assert result["success"] is True
        assert result["module_name"] == "counter"

    def test_lint_detects_module(self, tmp_path):
        tool = RTLLintTool(tmp_path)
        result = tool.run(SIMPLE_RTL)
        assert len(result["ports"]) >= 3


class TestRTLSynthesisTool:
    def test_mock_synthesis(self, tmp_path):
        tool = RTLSynthesisTool(tmp_path, process_nm=28)
        result = tool.run(SIMPLE_RTL, "counter")
        assert result["success"] is True
        assert result["area_cells"] > 0


class TestSimulationTool:
    def test_mock_simulation(self, tmp_path):
        tool = SimulationTool(tmp_path)
        result = tool.run(SIMPLE_RTL, SIMPLE_TB, "counter")
        assert result["success"] is True


class TestEDAToolchain:
    def test_available_tools(self, tmp_path):
        toolchain = EDAToolchain(tmp_path, process_nm=28)
        tools = toolchain.available_tools
        assert isinstance(tools, dict)
        assert "verilator" in tools
        assert "yosys" in tools
        assert "iverilog" in tools

    def test_lint_through_toolchain(self, tmp_path):
        toolchain = EDAToolchain(tmp_path, process_nm=28)
        result = toolchain.lint(SIMPLE_RTL)
        assert result["success"] is True

    def test_synthesize_through_toolchain(self, tmp_path):
        toolchain = EDAToolchain(tmp_path, process_nm=28)
        result = toolchain.synthesize(SIMPLE_RTL, "counter")
        assert result["success"] is True
