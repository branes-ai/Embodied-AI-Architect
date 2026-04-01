Generate the multi-level metrics dashboard for the current design state.

This is the "where am I" command — shows metrics at every level of abstraction so the architect can quickly identify what needs attention.

## Steps

1. **Find the current design state**: Look for the most recent session — either an active `SoCDesignRunner` state, recent `demo_interactive_review.py` output, or the result of a `branes design plan` run.

2. **System-level overview**: Show all SWaP-C metrics with budget utilization bars:
   ```
   Power:   [value]W / [budget]W  [████░░░░░░] [%]  [PASS/FAIL]
   Latency: [value]ms / [budget]ms ...
   Area:    [value]mm² / [budget]mm² ...
   Cost:    $[value] / $[budget] ...
   Weight:  [value]g / [budget]g ...
   Thermal: [value]°C / [max]°C ...
   ```

3. **Subsystem breakdown**: For each subsystem (perception, control, comms, storage):
   - Power contribution (W and % of total)
   - Latency contribution (ms and % of pipeline)
   - Which operators belong to this subsystem

4. **Operator breakdown**: For each operator in the pipeline:
   - GFLOPS, memory footprint, latency, power
   - Which hardware block it's mapped to
   - Utilization of that hardware block
   - Bound classification (compute/memory/IO)

5. **Efficiency metrics**:
   - Capability/Watt (mission success rate per watt)
   - GOPS/Watt (raw compute efficiency)
   - KPU/GPU/CPU utilization (% of peak)
   - Memory BW utilization (% of available)
   - Power headroom (watts remaining before limit)
   - Latency headroom (ms remaining before deadline)
   - Thermal headroom (°C below junction limit)

6. **Highlight the top 3 concerns**: The three metrics closest to their limits or already exceeding them. Use clear indicators:
   - GREEN: >20% headroom
   - YELLOW: 5-20% headroom (watch closely)
   - RED: <5% headroom or exceeded

Present as a single cohesive dashboard, not separate tool outputs. The architect should be able to scan this in 30 seconds and know exactly where to focus.
