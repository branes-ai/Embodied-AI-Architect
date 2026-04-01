# Die and Package costs

## Die Cost Estimates

Dies per wafer are computed using the standard formula for a 300mm wafer, then applied yield via a Poisson model with typical defect densities per node.

### Assumptions

- 300mm wafer (standard for all these nodes)
- 200mm² die at TSMC (2025/2026 pricing)
- Dies per wafer ≈ π·(150²) / 200 − π·150 / √200 ≈ **340** gross dies (edge loss ~15 dies)
- Yield model: Poisson, Y = e^(−D₀·A), where A = 2 cm²
- Defect densities and wafer costs from industry consensus (TSMC, Morgan Stanley, TrendForce)

| Node | Wafer Cost | Defect Density (D₀/cm²) | Yield | Dies/Wafer (gross) | Good Dies | Die Cost |
|------|-----------|------------------------|-------|-------------------|-----------|----------|
| 22nm | ~$3,500 | ~0.06 | ~89% | ~340 | ~302 | **~$12** |
| 12nm | ~$5,500 | ~0.05 | ~90% | ~340 | ~306 | **~$18** |
| 7nm | ~$9,500 | ~0.07 | ~87% | ~340 | ~296 | **~$32** |
| 5nm | ~$16,500 | ~0.09 | ~84% | ~340 | ~285 | **~$58** |
| 4nm | ~$19,000 | ~0.10 | ~82% | ~340 | ~279 | **~$68** |
| 3nm | ~$20,500 | ~0.12 | ~79% | ~340 | ~269 | **~$76** |

**Notes:**

- Wafer prices are **contract-dependent** and volume-sensitive — the numbers above are reasonable industry estimates, not published rack rates. Large customers (Apple, Nvidia) negotiate significantly different terms.
- Negotiations with AI/HPC customers suggest 4nm-class wafers recently moved from ~$18K to ~$20K per wafer, so 4nm costs are actively shifting.
- Samsung 3nm runs approximately 20–25% cheaper than TSMC at ~$15K/wafer, so foundry choice matters substantially at leading edge.
- 22nm and 12nm defect densities are well-characterized (mature processes); 3nm/4nm D₀ figures are still improving as the nodes mature — your actual yield could be better or worse depending on tape-out timing.
- This is **bare die cost only** — packaging (CoWoS, etc.), NRE amortization, and testing are all additive.

## Packaging Costs

Here are the two packaging cost tables appended to the original die cost data. All figures are for 2025/2026 pricing.

---

### Package Option 1: Flip-Chip BGA (FCBGA) — Traditional PCB Assembly

This is the standard path for CPUs, FPGAs, network SoCs, and anything going onto a PCB with discrete components. The package cost is essentially **node-independent** — it's driven by substrate layer count, ball count, and package size, not what process the die was made on. For a 200mm² die you'd need a reasonably capable multi-layer organic substrate.

| Node | Die Cost | FCBGA Substrate | Bumping/Assembly | Test & Final | **Total Packaged Cost** |
|------|----------|----------------|-----------------|--------------|------------------------|
| 22nm | ~$12 | ~$8–12 | ~$3–5 | ~$2–3 | **~$25–32** |
| 12nm | ~$18 | ~$8–12 | ~$3–5 | ~$2–3 | **~$31–38** |
| 7nm | ~$32 | ~$10–15 | ~$4–6 | ~$3–4 | **~$49–57** |
| 5nm | ~$58 | ~$10–15 | ~$4–6 | ~$3–4 | **~$75–83** |
| 4nm | ~$68 | ~$10–15 | ~$4–6 | ~$3–4 | **~$85–93** |
| 3nm | ~$76 | ~$10–15 | ~$4–6 | ~$3–4 | **~$93–101** |

**Notes:**

- Substrate cost scales with layer count and routing density, not fab node. A 200mm² die at high I/O (~2000+ bumps) pushes you to 8–12 layer build-up, ~$10–15.
- At lower I/O (<500 bumps), you can get away with 4–6 layer substrates at ~$6–8, saving ~$4–6.
- Bumping is done at wafer level before dicing; the per-die cost above amortizes that.
- Test cost here is basic post-package functional + burn-in. Does not include wafer-level KGD screening, which you'd want to add at 7nm and below (~$1–3/die extra).

---

### Package Option 2: CoWoS-S 2.5D MCM — Logic Die + 1–2× HBM Stacks

This is the TSMC CoWoS-S (silicon interposer) path. Your 200mm² logic die sits alongside 1 or 2 HBM3/HBM3E stacks on a silicon interposer, all landing on an ABF organic substrate. The HBM die and interposer costs are **additive on top of the logic die cost**.

| Node | Logic Die | Silicon Interposer¹ | Substrate (ABF) | Assembly/Microbumps | HBM3E (×1 stack, 24GB) | HBM3E (×2 stacks, 48GB) | Test (KGD + post-pkg) | **Total (1× HBM)** | **Total (2× HBM)** |
|------|-----------|--------------------|-----------------|--------------------|------------------------|--------------------------|----------------------|--------------------|--------------------|
| 22nm | ~$12 | ~$40–60 | ~$20–30 | ~$15–20 | ~$300 | ~$600 | ~$8–12 | **~$395–404** | **~$695–704** |
| 12nm | ~$18 | ~$40–60 | ~$20–30 | ~$15–20 | ~$300 | ~$600 | ~$8–12 | **~$401–410** | **~$701–710** |
| 7nm | ~$32 | ~$40–60 | ~$20–30 | ~$15–20 | ~$300 | ~$600 | ~$8–12 | **~$415–424** | **~$715–724** |
| 5nm | ~$58 | ~$40–60 | ~$20–30 | ~$15–20 | ~$300 | ~$600 | ~$8–12 | **~$441–450** | **~$741–750** |
| 4nm | ~$68 | ~$40–60 | ~$20–30 | ~$15–20 | ~$300 | ~$600 | ~$8–12 | **~$451–460** | **~$751–760** |
| 3nm | ~$76 | ~$40–60 | ~$20–30 | ~$15–20 | ~$300 | ~$600 | ~$8–12 | **~$459–468** | **~$759–768** |

¹ Interposer cost modeled for a ~600–800mm² silicon interposer (accommodating 200mm² logic + 1–2× HBM footprint). The interposer typically accounts for 50–70% of CoWoS packaging cost.

### Notes and critical caveats

- **HBM dominates the BOM.** HBM3E costs approximately $300 per 36GB stack. With two stacks, HBM alone is ~$600 — dwarfing the logic die cost at every node. This is why logic node choice barely moves the needle on total CoWoS system cost.
- **Interposer cost is area-driven**, not logic-node-driven. A larger interposer (needed for 2× HBM) pushes cost up. At 2× reticle size interposers (~1700mm²), costs can reach $80–120.
- Full CoW + substrate packaging margins run 40–45%, with B200-class packaging reaching ~$300–400/unit — those are much larger dies with 6× HBM stacks, which puts the above 1–2× HBM numbers in reasonable perspective.
- **CoWoS supply is constrained.** CoWoS capacity was constrained through 2025 with prices up 10–20%, and allocation pressure from hyperscalers (Nvidia holds ~60% of 2026 capacity) means smaller customers pay premiums or face queue times.
- Test cost at CoWoS level is higher than FCBGA because you must test each die as KGD before placement, then test the assembled package. A defective HBM or logic die post-assembly is a complete write-off.
- The 22nm/12nm rows in CoWoS are unusual in practice — you wouldn't typically put a mature-node die on CoWoS unless it was a specialized chiplet (e.g., I/O die, SerDes tile). The numbers are valid but the use case is niche.