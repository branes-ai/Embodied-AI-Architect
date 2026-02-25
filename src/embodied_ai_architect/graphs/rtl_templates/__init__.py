"""RTL template engine for KPU sub-components.

Renders Jinja2 templates for Verilog modules based on KPU configuration.
Each component type has a .sv.j2 template and a companion _tb.sv.j2 testbench.

Usage:
    from embodied_ai_architect.graphs.rtl_templates import RTLTemplateEngine

    engine = RTLTemplateEngine()
    rtl = engine.render("mac_unit", {"data_width": 8, "accum_width": 32})
    tb = engine.render_testbench("mac_unit", "mac_unit_8bit", {"data_width": 8})
    components = engine.get_kpu_components(kpu_config)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    from jinja2 import Environment, FileSystemLoader, TemplateNotFound

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

TEMPLATE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Fallback string-template rendering when Jinja2 is not installed
# ---------------------------------------------------------------------------


def _simple_render(template_str: str, params: dict[str, Any]) -> str:
    """Simple string.format-style rendering for when Jinja2 is unavailable."""
    result = template_str
    for key, value in params.items():
        result = result.replace(f"{{{{ {key} }}}}", str(value))
        result = result.replace(f"{{{{{key}}}}}", str(value))
    return result


# ---------------------------------------------------------------------------
# Template metadata
# ---------------------------------------------------------------------------

COMPONENT_TEMPLATES: dict[str, dict[str, Any]] = {
    "mac_unit": {
        "template": "mac_unit.sv.j2",
        "testbench": "mac_unit_tb.sv.j2",
        "params": ["data_width", "accum_width", "pipeline_stages"],
        "defaults": {"data_width": 8, "accum_width": 32, "pipeline_stages": 1},
    },
    "compute_tile": {
        "template": "compute_tile.sv.j2",
        "testbench": "compute_tile_tb.sv.j2",
        "params": ["array_rows", "array_cols", "data_width", "accum_width"],
        "defaults": {"array_rows": 16, "array_cols": 16, "data_width": 8, "accum_width": 32},
    },
    "l1_skew_buffer": {
        "template": "l1_skew_buffer.sv.j2",
        "testbench": "l1_skew_buffer_tb.sv.j2",
        "params": ["size_bytes", "num_banks", "data_width"],
        "defaults": {"size_bytes": 32768, "num_banks": 4, "data_width": 256},
    },
    "l2_cache_bank": {
        "template": "l2_cache_bank.sv.j2",
        "testbench": "l2_cache_bank_tb.sv.j2",
        "params": ["size_bytes", "num_banks", "read_ports", "write_ports"],
        "defaults": {"size_bytes": 262144, "num_banks": 8, "read_ports": 2, "write_ports": 1},
    },
    "l3_tile": {
        "template": "l3_tile.sv.j2",
        "testbench": "l3_tile_tb.sv.j2",
        "params": ["tile_size_bytes", "num_banks"],
        "defaults": {"tile_size_bytes": 524288, "num_banks": 4},
    },
    "noc_router": {
        "template": "noc_router.sv.j2",
        "testbench": "noc_router_tb.sv.j2",
        "params": ["num_ports", "link_width_bits", "buffer_depth"],
        "defaults": {"num_ports": 5, "link_width_bits": 256, "buffer_depth": 4},
    },
    "dma_engine": {
        "template": "dma_engine.sv.j2",
        "testbench": "dma_engine_tb.sv.j2",
        "params": ["max_transfer_bytes", "queue_depth"],
        "defaults": {"max_transfer_bytes": 1048576, "queue_depth": 8},
    },
    "block_mover": {
        "template": "block_mover.sv.j2",
        "testbench": "block_mover_tb.sv.j2",
        "params": ["transfer_width_bits"],
        "defaults": {"transfer_width_bits": 256},
    },
    "streamer": {
        "template": "streamer.sv.j2",
        "testbench": "streamer_tb.sv.j2",
        "params": ["prefetch_depth", "buffer_size_bytes"],
        "defaults": {"prefetch_depth": 4, "buffer_size_bytes": 16384},
    },
    "memory_controller": {
        "template": "memory_controller.sv.j2",
        "testbench": "memory_controller_tb.sv.j2",
        "params": ["addr_width", "data_width", "num_channels"],
        "defaults": {"addr_width": 32, "data_width": 64, "num_channels": 2},
    },
    "register_file": {
        "template": "register_file.sv.j2",
        "testbench": "register_file_tb.sv.j2",
        "params": ["num_regs", "data_width", "read_ports", "write_ports"],
        "defaults": {"num_regs": 32, "data_width": 32, "read_ports": 2, "write_ports": 1},
    },
    "alu": {
        "template": "alu.sv.j2",
        "testbench": "alu_tb.sv.j2",
        "params": ["data_width", "operations"],
        "defaults": {"data_width": 32, "operations": "add,sub,and,or,xor"},
    },
}


class RTLTemplateEngine:
    """Render RTL templates for KPU sub-components."""

    def __init__(self, template_dir: Optional[Path] = None):
        self._template_dir = template_dir or TEMPLATE_DIR
        if HAS_JINJA2:
            self._env = Environment(
                loader=FileSystemLoader(str(self._template_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self._env = None

    def available_templates(self) -> list[str]:
        """List available component types."""
        return list(COMPONENT_TEMPLATES.keys())

    def can_generate(self, component_type: str) -> bool:
        """Check if a component type template exists."""
        return component_type in COMPONENT_TEMPLATES

    def render(self, component_type: str, params: Optional[dict[str, Any]] = None) -> str:
        """Render a Verilog module from template.

        Args:
            component_type: Type of component (e.g. "mac_unit").
            params: Template parameters. Defaults filled from COMPONENT_TEMPLATES.

        Returns:
            Rendered SystemVerilog source code.
        """
        if component_type not in COMPONENT_TEMPLATES:
            raise ValueError(f"Unknown component type: {component_type}")

        meta = COMPONENT_TEMPLATES[component_type]
        merged = dict(meta["defaults"])
        if params:
            merged.update(params)

        template_name = meta["template"]
        return self._render_template(template_name, merged)

    def render_testbench(
        self,
        component_type: str,
        module_name: str,
        params: Optional[dict[str, Any]] = None,
    ) -> str:
        """Render a testbench for a component.

        Args:
            component_type: Type of component.
            module_name: Name of the DUT module.
            params: Template parameters.

        Returns:
            Rendered SystemVerilog testbench source code.
        """
        if component_type not in COMPONENT_TEMPLATES:
            raise ValueError(f"Unknown component type: {component_type}")

        meta = COMPONENT_TEMPLATES[component_type]
        merged = dict(meta["defaults"])
        if params:
            merged.update(params)
        merged["module_name"] = module_name

        tb_template = meta.get("testbench")
        if not tb_template:
            raise ValueError(f"No testbench template for {component_type}")

        return self._render_template(tb_template, merged)

    def get_kpu_components(self, kpu_config) -> list[dict[str, Any]]:
        """Map a KPU config to a list of component render specifications.

        Returns a list of dicts with keys: component_type, module_name, params
        """
        ct = kpu_config.compute_tile
        mt = kpu_config.memory_tile

        components = [
            {
                "component_type": "mac_unit",
                "module_name": "mac_unit",
                "params": {
                    "data_width": 8,
                    "accum_width": 32,
                    "pipeline_stages": 1,
                },
            },
            {
                "component_type": "compute_tile",
                "module_name": "compute_tile",
                "params": {
                    "array_rows": ct.array_rows,
                    "array_cols": ct.array_cols,
                    "data_width": 8,
                    "accum_width": 32,
                },
            },
            {
                "component_type": "l1_skew_buffer",
                "module_name": "l1_skew_buffer",
                "params": {
                    "size_bytes": ct.l1_size_bytes,
                    "num_banks": ct.l1_num_banks,
                    "data_width": 256,
                },
            },
            {
                "component_type": "l2_cache_bank",
                "module_name": "l2_cache_bank",
                "params": {
                    "size_bytes": ct.l2_size_bytes,
                    "num_banks": ct.l2_num_banks,
                    "read_ports": ct.l2_read_ports,
                    "write_ports": ct.l2_write_ports,
                },
            },
            {
                "component_type": "l3_tile",
                "module_name": "l3_tile",
                "params": {
                    "tile_size_bytes": mt.l3_tile_size_bytes,
                    "num_banks": mt.l3_num_banks,
                },
            },
            {
                "component_type": "noc_router",
                "module_name": "noc_router",
                "params": {
                    "num_ports": 5,
                    "link_width_bits": kpu_config.noc.link_width_bits,
                    "buffer_depth": 4,
                },
            },
            {
                "component_type": "dma_engine",
                "module_name": "dma_engine",
                "params": {
                    "max_transfer_bytes": mt.dma_max_transfer_bytes,
                    "queue_depth": mt.dma_queue_depth,
                },
            },
            {
                "component_type": "block_mover",
                "module_name": "block_mover",
                "params": {
                    "transfer_width_bits": 256,
                },
            },
            {
                "component_type": "streamer",
                "module_name": "streamer",
                "params": {
                    "prefetch_depth": ct.streamer_prefetch_depth,
                    "buffer_size_bytes": ct.streamer_buffer_bytes,
                },
            },
            {
                "component_type": "memory_controller",
                "module_name": "memory_controller",
                "params": {
                    "addr_width": 32,
                    "data_width": 64,
                    "num_channels": kpu_config.dram.channels_per_controller,
                },
            },
        ]
        return components

    def _render_template(self, template_name: str, params: dict[str, Any]) -> str:
        """Render a template file with parameters."""
        template_path = self._template_dir / template_name
        if template_path.exists() and self._env is not None:
            try:
                template = self._env.get_template(template_name)
                return template.render(**params)
            except Exception:
                pass

        # Fallback: use built-in minimal templates
        return _get_builtin_template(template_name, params)


# ---------------------------------------------------------------------------
# Built-in minimal templates (no Jinja2 dependency)
# ---------------------------------------------------------------------------


def _get_builtin_template(template_name: str, params: dict[str, Any]) -> str:
    """Generate minimal but valid SystemVerilog from parameters."""
    # Extract component type from template name
    base = template_name.replace(".sv.j2", "").replace("_tb", "")
    is_tb = "_tb" in template_name

    if is_tb:
        return _builtin_testbench(base, params)
    return _builtin_module(base, params)


def _builtin_module(component: str, params: dict[str, Any]) -> str:
    """Generate a minimal valid SystemVerilog module."""
    dw = params.get("data_width", 8)
    module_name = component

    if component == "mac_unit":
        aw = params.get("accum_width", 32)
        return f"""\
module {module_name} #(
    parameter DATA_WIDTH = {dw},
    parameter ACCUM_WIDTH = {aw}
)(
    input  logic clk,
    input  logic rst_n,
    input  logic signed [DATA_WIDTH-1:0] a,
    input  logic signed [DATA_WIDTH-1:0] b,
    input  logic signed [ACCUM_WIDTH-1:0] acc_in,
    output logic signed [ACCUM_WIDTH-1:0] acc_out,
    input  logic valid_in,
    output logic valid_out
);
    logic signed [ACCUM_WIDTH-1:0] product;
    assign product = a * b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= '0;
            valid_out <= 1'b0;
        end else begin
            acc_out <= acc_in + product;
            valid_out <= valid_in;
        end
    end
endmodule
"""

    elif component == "compute_tile":
        rows = params.get("array_rows", 16)
        cols = params.get("array_cols", 16)
        aw = params.get("accum_width", 32)
        return f"""\
module {module_name} #(
    parameter ARRAY_ROWS = {rows},
    parameter ARRAY_COLS = {cols},
    parameter DATA_WIDTH = {dw},
    parameter ACCUM_WIDTH = {aw}
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [ARRAY_ROWS*DATA_WIDTH-1:0] data_in,
    input  logic [ARRAY_COLS*DATA_WIDTH-1:0] weight_in,
    output logic [ARRAY_COLS*ACCUM_WIDTH-1:0] result_out,
    input  logic start,
    output logic done
);
    logic [ARRAY_COLS*ACCUM_WIDTH-1:0] accum;
    logic running;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0;
            done <= 1'b0;
            accum <= '0;
            result_out <= '0;
        end else if (start) begin
            running <= 1'b1;
            done <= 1'b0;
        end else if (running) begin
            for (int c = 0; c < ARRAY_COLS; c++) begin
                accum[c*ACCUM_WIDTH +: ACCUM_WIDTH] <=
                    accum[c*ACCUM_WIDTH +: ACCUM_WIDTH]
                    + data_in[0 +: DATA_WIDTH]
                    * weight_in[c*DATA_WIDTH +: DATA_WIDTH];
            end
            done <= 1'b1;
            running <= 1'b0;
            result_out <= accum;
        end
    end
endmodule
"""

    elif component == "l1_skew_buffer":
        size = params.get("size_bytes", 32768)
        banks = params.get("num_banks", 4)
        addr_w = max(1, (size // banks - 1).bit_length())
        return f"""\
module {module_name} #(
    parameter SIZE_BYTES = {size},
    parameter NUM_BANKS = {banks},
    parameter DATA_WIDTH = {dw},
    parameter ADDR_WIDTH = {addr_w}
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [ADDR_WIDTH-1:0] addr,
    input  logic [DATA_WIDTH-1:0] wdata,
    output logic [DATA_WIDTH-1:0] rdata,
    input  logic we,
    input  logic re
);
    logic [DATA_WIDTH-1:0] mem [0:(SIZE_BYTES/({dw}/8))-1];

    always @(posedge clk) begin
        if (we)
            mem[addr] <= wdata;
        if (re)
            rdata <= mem[addr];
    end
endmodule
"""

    elif component == "l2_cache_bank":
        size = params.get("size_bytes", 262144)
        banks = params.get("num_banks", 8)
        rp = params.get("read_ports", 2)
        wp = params.get("write_ports", 1)
        addr_w = max(1, (size - 1).bit_length())
        return f"""\
module {module_name} #(
    parameter SIZE_BYTES = {size},
    parameter NUM_BANKS = {banks},
    parameter READ_PORTS = {rp},
    parameter WRITE_PORTS = {wp},
    parameter ADDR_WIDTH = {addr_w}
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [ADDR_WIDTH-1:0] raddr,
    input  logic [ADDR_WIDTH-1:0] waddr,
    input  logic [63:0] wdata,
    output logic [63:0] rdata,
    input  logic we,
    input  logic re
);
    logic [63:0] mem [0:SIZE_BYTES/8-1];

    always @(posedge clk) begin
        if (we)
            mem[waddr[ADDR_WIDTH-1:3]] <= wdata;
        if (re)
            rdata <= mem[raddr[ADDR_WIDTH-1:3]];
    end
endmodule
"""

    elif component == "l3_tile":
        size = params.get("tile_size_bytes", 524288)
        banks = params.get("num_banks", 4)
        addr_w = max(1, (size - 1).bit_length())
        return f"""\
module {module_name} #(
    parameter TILE_SIZE_BYTES = {size},
    parameter NUM_BANKS = {banks},
    parameter ADDR_WIDTH = {addr_w}
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [ADDR_WIDTH-1:0] addr,
    input  logic [255:0] wdata,
    output logic [255:0] rdata,
    input  logic we,
    input  logic re
);
    logic [255:0] mem [0:TILE_SIZE_BYTES/32-1];

    always @(posedge clk) begin
        if (we)
            mem[addr[ADDR_WIDTH-1:5]] <= wdata;
        if (re)
            rdata <= mem[addr[ADDR_WIDTH-1:5]];
    end
endmodule
"""

    elif component == "noc_router":
        np = params.get("num_ports", 5)
        lw = params.get("link_width_bits", 256)
        bd = params.get("buffer_depth", 4)
        return f"""\
module {module_name} #(
    parameter NUM_PORTS = {np},
    parameter LINK_WIDTH = {lw},
    parameter BUFFER_DEPTH = {bd}
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [NUM_PORTS*LINK_WIDTH-1:0] data_in,
    input  logic [NUM_PORTS-1:0] valid_in,
    output logic [NUM_PORTS-1:0] ready_out,
    output logic [NUM_PORTS*LINK_WIDTH-1:0] data_out,
    output logic [NUM_PORTS-1:0] valid_out,
    input  logic [NUM_PORTS-1:0] ready_in
);
    // Simple round-robin routing
    logic [$clog2(NUM_PORTS)-1:0] grant;
    integer next;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            grant <= '0;
            data_out <= '0;
            valid_out <= '0;
            ready_out <= {{NUM_PORTS{{1'b1}}}};
        end else begin
            grant <= (grant + 1) % NUM_PORTS;
            for (int i = 0; i < NUM_PORTS; i++) begin
                next = (i + 1) % NUM_PORTS;
                if (valid_in[i] && ready_in[next]) begin
                    data_out[next*LINK_WIDTH +: LINK_WIDTH] <=
                        data_in[i*LINK_WIDTH +: LINK_WIDTH];
                    valid_out[next] <= 1'b1;
                end else begin
                    valid_out[i] <= 1'b0;
                end
                ready_out[i] <= 1'b1;
            end
        end
    end
endmodule
"""

    elif component == "dma_engine":
        mt = params.get("max_transfer_bytes", 1048576)
        qd = params.get("queue_depth", 8)
        addr_w = max(1, (mt - 1).bit_length())
        return f"""\
module {module_name} #(
    parameter ADDR_WIDTH = {addr_w},
    parameter QUEUE_DEPTH = {qd}
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [ADDR_WIDTH-1:0] src_addr,
    input  logic [ADDR_WIDTH-1:0] dst_addr,
    input  logic [15:0] length,
    input  logic start,
    output logic busy,
    output logic done
);
    logic [15:0] counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            counter <= '0;
        end else if (start && !busy) begin
            busy <= 1'b1;
            done <= 1'b0;
            counter <= length;
        end else if (busy) begin
            counter <= counter - 1;
            if (counter == 1) begin
                busy <= 1'b0;
                done <= 1'b1;
            end
        end else begin
            done <= 1'b0;
        end
    end
endmodule
"""

    elif component == "block_mover":
        tw = params.get("transfer_width_bits", 256)
        return f"""\
module {module_name} #(
    parameter TRANSFER_WIDTH = {tw}
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [TRANSFER_WIDTH-1:0] data_in,
    output logic [TRANSFER_WIDTH-1:0] data_out,
    input  logic valid_in,
    output logic valid_out,
    input  logic start,
    output logic done
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= '0;
            valid_out <= 1'b0;
            done <= 1'b0;
        end else if (valid_in) begin
            data_out <= data_in;
            valid_out <= 1'b1;
            done <= 1'b1;
        end else begin
            valid_out <= 1'b0;
            done <= 1'b0;
        end
    end
endmodule
"""

    elif component == "streamer":
        pd = params.get("prefetch_depth", 4)
        bs = params.get("buffer_size_bytes", 16384)
        return f"""\
module {module_name} #(
    parameter PREFETCH_DEPTH = {pd},
    parameter BUFFER_SIZE = {bs}
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [255:0] mem_data,
    input  logic mem_valid,
    output logic mem_req,
    output logic [255:0] compute_data,
    output logic compute_valid,
    input  logic compute_ready
);
    logic [255:0] buffer [0:PREFETCH_DEPTH-1];
    logic [$clog2(PREFETCH_DEPTH):0] count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= '0;
            mem_req <= 1'b1;
            compute_valid <= 1'b0;
        end else begin
            if (mem_valid && count < PREFETCH_DEPTH) begin
                buffer[count[$clog2(PREFETCH_DEPTH)-1:0]] <= mem_data;
                count <= count + 1;
            end
            mem_req <= (count < PREFETCH_DEPTH);
            if (compute_ready && count > 0) begin
                compute_data <= buffer[0];
                compute_valid <= 1'b1;
                count <= count - 1;
            end else begin
                compute_valid <= 1'b0;
            end
        end
    end
endmodule
"""

    elif component == "memory_controller":
        aw = params.get("addr_width", 32)
        ddw = params.get("data_width", 64)
        nc = params.get("num_channels", 2)
        return f"""\
module {module_name} #(
    parameter ADDR_WIDTH = {aw},
    parameter DATA_WIDTH = {ddw},
    parameter NUM_CHANNELS = {nc}
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [ADDR_WIDTH-1:0] addr,
    input  logic [DATA_WIDTH-1:0] wdata,
    output logic [DATA_WIDTH-1:0] rdata,
    input  logic we,
    input  logic re,
    output logic ready,
    output logic valid
);
    logic [2:0] latency_counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ready <= 1'b1;
            valid <= 1'b0;
            rdata <= '0;
            latency_counter <= '0;
        end else if ((we || re) && ready) begin
            ready <= 1'b0;
            latency_counter <= 3'd4;
        end else if (latency_counter > 0) begin
            latency_counter <= latency_counter - 1;
            if (latency_counter == 1) begin
                ready <= 1'b1;
                valid <= 1'b1;
                rdata <= '0;
            end
        end else begin
            valid <= 1'b0;
        end
    end
endmodule
"""

    elif component == "register_file":
        nr = params.get("num_regs", 32)
        rp = params.get("read_ports", 2)
        wp = params.get("write_ports", 1)
        reg_aw = max(1, (nr - 1).bit_length())
        return f"""\
module {module_name} #(
    parameter NUM_REGS = {nr},
    parameter DATA_WIDTH = {dw},
    parameter ADDR_WIDTH = {reg_aw}
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [ADDR_WIDTH-1:0] raddr1,
    input  logic [ADDR_WIDTH-1:0] raddr2,
    output logic [DATA_WIDTH-1:0] rdata1,
    output logic [DATA_WIDTH-1:0] rdata2,
    input  logic [ADDR_WIDTH-1:0] waddr,
    input  logic [DATA_WIDTH-1:0] wdata,
    input  logic we
);
    logic [DATA_WIDTH-1:0] regs [0:NUM_REGS-1];

    assign rdata1 = regs[raddr1];
    assign rdata2 = regs[raddr2];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_REGS; i++)
                regs[i] <= '0;
        end else if (we) begin
            regs[waddr] <= wdata;
        end
    end
endmodule
"""

    elif component == "alu":
        ops = params.get("operations", "add,sub,and,or,xor")
        return f"""\
module {module_name} #(
    parameter DATA_WIDTH = {dw}
)(
    input  logic clk,
    input  logic [DATA_WIDTH-1:0] a,
    input  logic [DATA_WIDTH-1:0] b,
    input  logic [2:0] op,
    output logic [DATA_WIDTH-1:0] result,
    output logic zero
);
    always @(*) begin
        case (op)
            3'd0: result = a + b;      // ADD
            3'd1: result = a - b;      // SUB
            3'd2: result = a & b;      // AND
            3'd3: result = a | b;      // OR
            3'd4: result = a ^ b;      // XOR
            default: result = '0;
        endcase
    end

    assign zero = (result == '0);
endmodule
"""

    else:
        # Generic fallback
        return f"""\
module {module_name} (
    input  logic clk,
    input  logic rst_n
);
    // Placeholder for {component}
endmodule
"""


def _builtin_testbench(component: str, params: dict[str, Any]) -> str:
    """Generate a minimal testbench."""
    module_name = params.get("module_name", component)
    dw = params.get("data_width", 8)

    return f"""\
module {component}_tb;
    logic clk = 0;
    logic rst_n;

    always #5 clk = ~clk;

    {module_name} dut (
        .clk(clk),
        .rst_n(rst_n)
    );

    initial begin
        rst_n = 0;
        #20 rst_n = 1;
        #100;
        $display("ALL TESTS PASSED");
        $finish;
    end
endmodule
"""
