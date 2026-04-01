Deep-dive analysis of a specific bottleneck target: $ARGUMENTS

The target can be a subsystem (perception, control), an operator (yolo_detector, tracker, vio), a kernel (conv2d, fft, matmul), or a physical component (kpu, sram, interconnect).

## Steps

1. **Identify the target**: Parse $ARGUMENTS to determine what to analyze. If ambiguous, list the available targets at each level and ask the architect to clarify.

2. **Gather metrics for the target**:
   - Run `.venv/bin/branes mcp analyze` if it's a model/operator
   - Check PPA metrics from the current design state
   - Check bandwidth_validator, floorplan_validator results if physical
   - Check the optimization_review_snapshot if available

3. **Show the detailed breakdown**:

   For an **operator** (e.g., YOLO detector):
   ```
   YOLO Detector — Detailed Analysis
   Total: 2.1W | 15ms | compute-bound | mapped to: KPU

   Kernel          Power   Latency  Bound     Utilization
   ─────────────── ──────  ───────  ────────  ───────────
   backbone_conv   1.2W    8ms      compute   78% ALU
   neck_fpn        0.4W    3ms      memory    65% BW
   head_detect     0.3W    2ms      compute   52% ALU
   nms             0.2W    2ms      CPU-bound 12% (single-thread)
   ```

   For a **physical component** (e.g., SRAM):
   ```
   On-Chip SRAM — Detailed Analysis
   Total: 512KB | 2.1mm² | 0.3W static

   Bank    Size    Read BW    Write BW   Utilization  Purpose
   ──────  ──────  ─────────  ─────────  ───────────  ────────
   L1      64KB    19.2 GB/s  9.6 GB/s   91%          Activation buffer
   L2      256KB   12.8 GB/s  6.4 GB/s   72%          Weight cache
   Scratch 192KB   6.4 GB/s   6.4 GB/s   45%          Change detection
   ```

   For a **constraint** (e.g., power, cost):
   ```
   Power Budget — Detailed Analysis
   Budget: 5.0W | Actual: 6.2W | Margin: -24% (FAIL)

   Component        Power    % of Total   Reducible?
   ───────────────  ───────  ──────────   ──────────
   KPU compute      3.2W     52%          Yes (quantize, duty-cycle)
   SRAM leakage     0.8W     13%          Partially (power gating)
   CPU subsystem    0.6W     10%          Minimal
   IO/PHY           0.5W     8%           No (fixed)
   DRAM interface   0.4W     6%           Yes (reduce BW)
   Clock tree       0.3W     5%           Yes (clock gating)
   Other            0.4W     6%           -
   ```

4. **Identify root cause**: Why is this target a bottleneck?
   - Is it inherent to the workload (fundamental compute requirement)?
   - Is it a mapping issue (wrong operator→hardware assignment)?
   - Is it a configuration issue (clock speed, memory hierarchy sizing)?
   - Is it a physical limitation (process node, die area)?

5. **Propose targeted actions**: 3-5 specific things that could improve this target, with estimated impact and side effects.

Present the analysis clearly. The architect is drilling down because they identified this as a priority — give them the depth they need to make a decision.
