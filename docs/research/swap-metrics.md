# SWaP (Size, Weight, and Power) Metrics: Industry Landscape & Product Design Flow

> Research summary: Which industries and product development flows are governed by SWaP metrics,
> and how SWaP specifications are defined, applied, and refined throughout product design.

---

## Table of Contents

1. [Industries Governed by SWaP Constraints](#1-industries-governed-by-swap-constraints)
2. [How SWaP Specifications Are Defined](#2-how-swap-specifications-are-defined)
3. [Key Standards and Frameworks](#3-key-standards-and-frameworks)
4. [SWaP Throughout the Product Design Lifecycle](#4-swap-throughout-the-product-design-lifecycle)
5. [SWaP Refinement and Trade-Off Methodologies](#5-swap-refinement-and-trade-off-methodologies)
6. [Technology Strategies for SWaP Optimization](#6-technology-strategies-for-swap-optimization)
7. [Edge AI and Robotics: The Emerging SWaP Frontier](#7-edge-ai-and-robotics-the-emerging-swap-frontier)
8. [Summary Comparison](#8-summary-comparison)
9. [References](#9-references)

---

## 1. Industries Governed by SWaP Constraints

### 1.1 Defense and Aerospace (Origin Domain)

SWaP originated in the defense and aerospace sector and remains its most formalized application.
The Department of Defense uses SWaP (and its extensions SWaP-C and SWaP-C2) as fundamental design
criteria for virtually every system it procures. Every kilogram saved directly contributes to
increased payload capacity or extended operational range. Power efficiency influences mission
duration and battlefield survivability, while size determines platform integrability.

**Key product categories:**

- **Unmanned Aerial Vehicles (UAVs)**: The most SWaP-constrained class. Payload weight directly
  reduces flight endurance and range. Processing for video, thermal imaging, and mapping must fit
  within strict envelopes.
- **Soldier-Worn Equipment**: Soldiers carry up to 120 pounds of equipment. SWaP optimization of
  radios, displays, batteries, sensors, and computing is critical. L3Harris's AN/PRC-171 Compact
  Team Radio is explicitly marketed as "low-SWaP."
- **Tactical Vehicles**: Vetronics (vehicle electronics), display systems, and C4ISR suites must
  integrate into space-constrained platforms.
- **Satellites and Spacecraft**: Mass is the fundamental resource -- launch cost scales directly
  with mass, and exceeding launch vehicle capacity is a hard boundary.
- **Missiles and Munitions**: Guidance electronics, seekers, and fuzing systems operate under
  extreme size and weight budgets.
- **Avionics**: Line Replaceable Units (LRUs) must meet strict volume and power allocations within
  aircraft bays.
- **Radar and Electronic Warfare**: Signal processing demands high compute at constrained power;
  FPGA power modeling is critical for SWaP-optimized EW solutions.

### 1.2 Space Systems

Space programs apply the most rigorous quantitative SWaP management of any industry. Mass and
power budgets are tracked as formal Technical Performance Measures (TPMs) throughout the entire
lifecycle. A satellite's mass budget is allocated across subsystems with specific margins, and
exceeding the launch vehicle mass limit is a hard program failure.

Typical LEO spacecraft mass allocation (percent of dry mass):

| Subsystem              | % of Dry Mass |
|------------------------|---------------|
| Payload                | 31%           |
| Structure/Mechanisms   | 27%           |
| Power systems          | 21%           |
| Attitude control       | 6%            |
| Processing             | 5%            |
| Propulsion             | 3%            |
| Thermal                | 2%            |
| Communications         | 2%            |

Power allocation follows a similar discipline, with payload consuming roughly 48% of available
power.

### 1.3 Automotive (Electrification-Driven)

SWaP concerns in automotive have intensified with electrification. Battery Electric Vehicles (BEVs)
are 15--33% heavier than equivalent ICE vehicles, making weight a first-order design constraint.

Key concerns:

- **Battery packs**: Energy density vs. weight tradeoff drives range.
- **Power delivery networks**: 48V zonal architecture can reduce harness weight by up to 85%.
- **Thermal management**: Up to 33% weight reduction possible with optimized cooling.
- **ADAS/autonomous driving compute**: Edge AI processors must deliver high TOPS within automotive
  thermal and power envelopes.

### 1.4 Medical Devices

Portable and wearable medical devices face acute SWaP constraints:

- **Wearable monitors** (continuous glucose monitors, ECG patches, pulse oximeters): Must be light
  enough for continuous wear while lasting days on coin-cell batteries.
- **Portable diagnostics** (ultrasound, point-of-care analyzers): Weight and battery life determine
  field usability.
- **Implantable devices** (pacemakers, neurostimulators): Extreme miniaturization with decade-long
  battery life requirements.
- **Home care devices** (CPAP, nebulizers): Must be consumer-friendly in size and weight.

Power consumption varies enormously by sensor type: accelerometers use microwatts, ECG sensors use
~1 mW, cameras use ~300 mW. Design strategies involve aggressive duty cycling -- leaving only the
accelerometer running and waking higher-power subsystems only when movement is detected.

### 1.5 Robotics

Autonomous mobile robots face a fundamental tension: battery capacity increases range but adds
weight, which consumes more energy. Locomotion alone accounts for over 50% of total energy
consumption.

Key categories:

- **Autonomous Mobile Robots (AMRs)**: Warehouse and logistics robots must balance payload capacity
  against battery life.
- **Inspection drones**: SWaP directly determines flight time and sensor capability.
- **Surgical robots**: Size constraints for minimally invasive procedures.
- **Field robots**: Agricultural, mining, and construction robots operating untethered.

### 1.6 Consumer Electronics

While not typically using the "SWaP" acronym, consumer electronics follow identical principles:

- **Smartphones**: Relentless miniaturization with increasing compute demands.
- **Smartwatches and fitness trackers**: Must maintain expected form factors while adding sensor
  complexity.
- **Wireless earbuds**: Extreme size constraints with meaningful compute (ANC, spatial audio).
- **Laptops/tablets**: Power envelope determines battery life and thermal design.

### 1.7 IoT and Industrial Sensors

IoT sensor nodes operate under perhaps the most extreme power constraints of any domain. Many must
run for years on a single battery or energy harvesting alone:

- **Industrial wireless sensors**: Vibration, temperature, pressure monitoring in remote locations.
- **Smart agriculture**: Soil sensors, weather stations operating on solar harvesting.
- **Smart buildings**: Occupancy sensors, air quality monitors.
- **Asset tracking**: GPS + cellular trackers with multi-year battery life.

---

## 2. How SWaP Specifications Are Defined

### 2.1 The SWaP / SWaP-C / SWaP-C2 Framework Evolution

The framework has evolved through three generations:

| Acronym     | Components                          | Context                                              |
|-------------|-------------------------------------|------------------------------------------------------|
| **SWaP**    | Size, Weight, Power                 | Original formulation                                 |
| **SWaP-C**  | Size, Weight, Power, Cost           | Added when cost was recognized as equally critical    |
| **SWaP-C2** | Size, Weight, Power, Cost, Cooling  | DoD expansion reflecting thermal management as a first-class constraint |

The addition of Cooling reflects that "more components in smaller spaces generates more heat, with
less flow space," making thermal management a distinct engineering challenge rather than a
subordinate concern [1].

### 2.2 Budget-Based Specification

SWaP specifications are established through a hierarchical budget allocation process:

1. **System-Level Constraint**: The top-level constraint is set by the platform (e.g., launch
   vehicle mass capacity, aircraft power bus capacity, soldier load limit, battery capacity).
2. **Subsystem Allocation**: The system-level budget is decomposed and allocated to subsystems,
   with each receiving a mass allocation, power allocation, and volume allocation.
3. **Margin Reservation**: A portion of the total budget is held as unallocated margin to absorb
   growth during development.

### 2.3 NASA/Aerospace Margin Standards

NASA's Systems Engineering Handbook defines margin as "allowances carried in budget, projected
schedules, and technical performance parameters (e.g., weight, power, or memory) to account for
uncertainties and risks." The recommended margins decrease as design maturity increases:

| Development Phase                  | Recommended Margin |
|------------------------------------|--------------------|
| Phase A (Concept)                  | ~100%              |
| Phase B (Preliminary Design)       | ~30--50%           |
| Phase C (Detailed Design)          | ~25%               |

Historical data shows that mean inert mass growth across space programs is 28%, and 30% of programs
exceed the recommended 32.5% allowable growth-plus-margin [14, 15].

---

## 3. Key Standards and Frameworks

### 3.1 Defense Standards

| Standard           | Scope                                                                                     |
|--------------------|-------------------------------------------------------------------------------------------|
| **MIL-STD-810H**   | Environmental testing (temperature, vibration, shock, humidity, altitude, sand/dust). Constrains material choices, enclosure designs, and cooling approaches that determine size and weight [22]. |
| **MIL-STD-461G**   | EMC/EMI requirements. Drives shielding weight and power filtering, directly impacting SWaP budgets. |
| **MIL-STD-704**    | Aircraft electric power characteristics. Defines the electrical power characteristics aircraft equipment must accommodate. |
| **MIL-PRF-38534**  | Qualification for hybrid microcircuits in high-reliability applications. Relevant to SWaP-optimized RF components and modules. |

### 3.2 Aerospace Standards

| Standard                     | Scope                                                                                   |
|------------------------------|-----------------------------------------------------------------------------------------|
| **ANSI/AIAA S-120A-2015**    | Mass properties control for space systems -- the primary industry standard. Defines terminology and processes for management, control, monitoring, determination, verification, and documentation of mass properties [16]. |
| **SMC-T-002**                | Tailoring guidance for AIAA S-120A, providing compliance tables based on mission risk class. |
| **DO-178C**                  | Software considerations in airborne systems. Airworthiness certification using Design Assurance Levels (DAL) A through E [19]. |
| **DO-254**                   | Design assurance guidance for airborne electronic hardware (FPGAs, ASICs, PLDs). Recognized by the FAA through AC 20-152 [19]. |
| **RTCA DO-160**              | Environmental conditions and test procedures for airborne equipment (civil aviation equivalent of MIL-STD-810). |
| **ECSS-E-ST-10**             | European Cooperation for Space Standardization framework for system engineering, including budget and margin management [21]. |

### 3.3 Automotive Standards

| Standard             | Scope                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------|
| **ISO 26262**        | Functional safety for road vehicles. Deeply impacts SWaP through constraints on power management approaches. Power gating -- a common power reduction technique -- may not be viable in collision avoidance systems because wake-up latency could be catastrophic [20]. |
| **FMVSS No. 305a**  | Federal safety standard for electric powertrain integrity.                                 |

### 3.4 Medical Device Standards

| Standard              | Scope                                                                                    |
|-----------------------|------------------------------------------------------------------------------------------|
| **IEC 60601-1**       | Medical electrical equipment safety. Sets thermal safety limits, leakage current requirements (<10 µA), and isolation requirements (4000 VAC for 2xMOPP). Critically constrains power supply design for portable medical devices [24]. |
| **IEC 60601-1-11**    | Specific requirements for home-use medical devices (CPAP, nebulizers, breast pumps).      |
| **IEC 62368-1**       | Safety standard for IT/AV equipment (relevant for consumer wearables without medical claims). |

### 3.5 Defense Acquisition Frameworks

- **Modular Open Systems Approach (MOSA)**: DoD policy encouraging modular, standards-based
  architectures that enable SWaP-C optimization through component interchangeability.
- **Other Transaction Authorities (OTAs)**: Acquisition pathway used by DoD to accelerate SWaP-C
  innovation with non-traditional defense contractors.

---

## 4. SWaP Throughout the Product Design Lifecycle

### 4.1 Phase A -- Concept (CoDR / SRR)

- **Mission/system requirements** establish top-level SWaP constraints (e.g., "payload must weigh
  less than X kg," "total power consumption less than Y watts").
- **Mass Estimating Relationships (MERs)**: Historically-based parametric models predict subsystem
  masses from top-level parameters.
- **Initial budget allocation**: First-pass decomposition of mass, power, and volume to subsystems.
- **Large margins held**: ~100% margin at this stage acknowledges high uncertainty.
- **Trade space exploration**: Broad evaluation of architectures and technologies.

### 4.2 Phase B -- Preliminary Design (PDR)

- **PDR presents** basic system designs across software, mechanical, power distribution, thermal
  management, and electronic domains with preliminary estimates of weight, power consumption, and
  volume.
- **Margin reduced to ~30%**: Reflecting increased design certainty.
- **Subsystem trade studies**: Formal evaluation of component alternatives against SWaP-C criteria.
- **Thermal analysis begins**: Initial thermal models identify hot spots and cooling requirements.
- **Power mode analysis**: Detailed power consumption estimates across operational modes (standby,
  active, peak, safe mode).

### 4.3 Phase C -- Detailed Design (CDR)

- **CDR presents** final designs through completed analyses, simulations, schematics, and test
  results.
- **Margin reduced to ~25%**: Design should be largely frozen.
- **Detailed mass properties**: Actual component weights replace estimates.
- **Power measurements**: Breadboard and prototype measurements validate power models.
- **Thermal verification**: Detailed thermal analysis and early test correlation.
- **Design must be complete and comprehensive** before significant production begins.

### 4.4 Phase D -- Fabrication, Integration, and Test

- **As-built mass properties**: Measured weights of actual hardware.
- **Power testing**: Measured consumption across all operational modes.
- **Thermal vacuum testing**: Validates thermal models under flight-like conditions.
- **Margin consumption tracking**: Remaining margin tracked against actuals.
- **Environmental qualification**: MIL-STD-810, DO-160, or equivalent testing.

### 4.5 Phase E -- Operations

- **Performance monitoring**: Actual power consumption and thermal behavior in operational
  environment.
- **Lessons learned**: Feeds back into MERs and margin guidelines for future programs.

---

## 5. SWaP Refinement and Trade-Off Methodologies

### 5.1 Iterative Trade-Off Process

SWaP-C optimization is fundamentally about balance. Meeting all four criteria simultaneously is
"difficult, if not impossible," requiring strategic decisions about which objectives to
prioritize [6].

The standard trade-off process:

1. **Define the trade space**: Identify design alternatives (components, materials, architectures).
2. **Establish evaluation criteria**: Weight the SWaP-C parameters based on mission priorities.
3. **Quantify alternatives**: Score each option against criteria using analysis, simulation, or test
   data.
4. **Visualize trade-offs**: Decision matrices, Pareto charts, spider diagrams, or multi-criteria
   decision analysis (MCDA).
5. **Select and document**: Choose the preferred alternative with rationale.
6. **Iterate**: Revisit as design matures and estimates become actuals.

### 5.2 Multi-Criteria Decision Analysis (MCDA)

Formal MCDA techniques are applied to SWaP trade studies:

- **Weighted scoring matrices**: Each SWaP-C parameter receives a weight reflecting mission
  priorities; alternatives are scored and ranked.
- **Pareto front analysis**: Multi-objective optimization identifies the set of solutions where no
  parameter can be improved without degrading another. The Pareto front provides a "comprehensive
  view of all possible trade-offs, allowing decision makers to select a solution a posteriori based
  on their preferences."
- **Utility functions**: Map physical parameters (grams, watts, cubic centimeters, dollars) to a
  common utility scale for comparison.

### 5.3 Model-Based Systems Engineering (MBSE)

MBSE is increasingly applied to SWaP management:

- **Digital twins**: Virtual models track mass, power, and thermal properties throughout design.
- **Automated trade studies**: MBSE tools systematize criteria, utility functions, and weighting
  factors for selection of optimal alternatives.
- **Virtual Satellite** (German Aerospace Center / DLR): An open-source MBSE tool that stores
  system models and automates steady-state and transient thermal analyses [27].

### 5.4 Parametric Cost Estimation Tools

**Galorath SEER** is the leading parametric estimation platform for SWaP-C trade studies in defense
programs [25, 26]:

- Provides comparative analysis between commercial and bespoke solutions.
- Historical performance data validates vendor claims.
- Risk assessment for commercial technology adaptation.
- Cost modeling for ruggedization and MIL-SPEC compliance.
- Programs can evaluate module configurations against baseline performance metrics, model ripple
  effects of changes across the system, project lifecycle cost implications, and evaluate
  maintenance/sustainment impacts.
- Supports digital engineering integration with virtual prototyping and "cost twin" models.

### 5.5 Advanced Analytics and Machine Learning

Modern SWaP-C optimization increasingly uses:

- **ML algorithms** to identify optimal configurations across large design spaces.
- **Pattern recognition** across historical project data.
- **Predictive modeling** for performance and cost outcomes.

---

## 6. Technology Strategies for SWaP Optimization

### 6.1 Semiconductor and Packaging

- **System-on-Chip (SoC)**: Replacing multiple chips with one SoC dramatically reduces total power
  consumption. Moving IP cores (Ethernet, PCIe, DDR controllers) into hard silicon provides
  lower-power, more efficient designs.
- **Adaptive SoCs / Hybrid FPGAs**: Heterogeneous architectures combining FPGA logic, CPUs, DSP
  accelerators, and high-speed network fabrics on a single device. Matrix multiplication and
  convolution operations offloaded to FPGA fabric consume less power than general-purpose
  processors [23].
- **Advanced Packaging (2.5D/3D-IC)**: Chiplet-based architectures stack dies vertically, reducing
  footprint and interconnect power. 2.5D architectures "can significantly bolster CSWaP
  initiatives" [30].
- **GaN, SiGe, CMOS advances**: Enable higher integration and improved efficiency for RF and power
  electronics.

### 6.2 Thermal Management Approaches

With the expansion to SWaP-C2, thermal management has become a first-class design dimension:

- **Passive cooling**: Heat sinks, heat pipes, conductive enclosure materials (preferred for
  reliability -- no moving parts).
- **Direct Air Flow-Through (AFT)**: Circulates air directly over heat-generating components.
- **Liquid Flow-Through (LFT)**: Circulates coolant for superior heat removal.
- **Pumped two-phase liquid cooling**: Advanced approach for highest-density applications [28, 29].

### 6.3 Architecture-Level Strategies

- **48V zonal architecture** (automotive): Reduces wiring harness weight by up to 85%, eliminates
  low-voltage auxiliary batteries, reduces thermal management weight by 33%.
- **Modular open systems**: Connector families sharing parts, tooling, and qualified components
  across platforms reduce cost while enabling SWaP-optimized module swaps.
- **Duty cycling** (IoT/wearables): Aggressive power management where low-power sensors
  (accelerometers at microwatts) remain active while high-power subsystems (cameras at 300 mW, GPS)
  sleep until triggered [31].
- **Energy harvesting** (IoT): Vibration, solar, thermal, and RF harvesting to supplement or
  replace batteries entirely.

### 6.4 Materials and Manufacturing

- **Milled aluminum enclosures**: Provide rigidity, thermal management, and customizable designs --
  reported achieving products at "nearly 50% of the weight limit outlined in client program
  specifications" [10].
- **High-Density Interconnect (HDI) PCBs**: Microvias, blind vias, and buried vias enable
  multilayered designs in smaller footprints.
- **Flexible and Rigid-Flex PCBs**: Critical for wearable and IoT form factors.

---

## 7. Edge AI and Robotics: The Emerging SWaP Frontier

The intersection of AI inference and SWaP constraints is a rapidly evolving domain, and directly
relevant to the Embodied AI Architect project:

- **Power per inference** is now a primary design metric for embedded AI, not secondary.
- Embedded AI targets **1--100 TOPS/W**, with Neural Processing Units delivering 40--2000 TOPS.
- **Memory constraints**: Techniques like memory tiling, double buffering, and activation reuse
  prevent stalls in memory-constrained devices.
- **Deterministic latency**: Safety systems demand bounded compute paths and predictable memory
  access -- AI must not introduce jitter.
- **Heterogeneous compute** (CPU + GPU + NPU) enables more efficient workload distribution.
- Battery-powered robots and smart cameras require "balanced solutions" that optimize
  performance-per-watt [32, 33].

---

## 8. Summary Comparison

| Dimension            | Defense/Aerospace            | Automotive                | Medical                   | Robotics/IoT              | Consumer              |
|----------------------|------------------------------|---------------------------|---------------------------|---------------------------|-----------------------|
| **Primary driver**   | Mission capability, survivability | Range, safety          | Wearability, battery life | Autonomy duration         | User experience       |
| **Formalization**    | Very high (AIAA S-120A, MIL-STDs) | High (ISO 26262, FMVSS) | High (IEC 60601)        | Medium                    | Low (market-driven)   |
| **Margin tracking**  | Quantitative budgets at subsystem level | Growing with EV programs | Per-device power budgets | Battery life modeling   | Informal              |
| **Trade-off tools**  | SEER, MBSE, Pareto analysis  | Automotive MBSE, simulation | FDA risk frameworks     | Simulation, prototyping   | Iterative prototyping |
| **Cooling constraint** | SWaP-C2 formalized         | Major (battery thermal)   | Skin safety limits        | Sealed enclosures         | Thin form factors     |

The SWaP framework, born in defense, has become a universal engineering concern as systems across
all industries push toward greater capability in smaller, lighter, more power-efficient packages.
The defense and aerospace sectors remain the most methodologically rigorous, with formal standards,
quantitative budget tracking, and margin management processes that other industries are increasingly
adopting as their own SWaP pressures intensify.

---

## 9. References

1. Sealevel Systems. "SWaP-C & SWaP-C2: What's Next for Embedded Computing."
   <https://www.sealevel.com/swap-swapc2>

2. Sealevel Systems. "SWaP-C2 Applications: The Battlefield & Beyond."
   <https://www.sealevel.com/swap-c2-applications>

3. BAE Systems. "What is SWaP-C?"
   <https://www.baesystems.com/en-us/definition/what-is-swap-c>

4. Portescap. "The Importance of SWaP in Aerospace and Defense Applications." July 2023.
   <https://www.portescap.com/en/newsroom/blog/2023/07/the-importance-of-swap-in-aerospace-and-defense-applications>

5. Curtiss-Wright. "Size, Weight, Power and Cost."
   <https://defense-solutions.curtisswright.com/capabilities/technologies/swap-c>

6. NSTXL. "What is SWaP-C?"
   <https://nstxl.org/what-is-swap-c/>

7. Amphenol Aerospace. "SWaP-C Fundamentals Are the Key to Future Defense Architectures."
   <https://www.amphenol-aerospace.com/blog/swap-c-fundamentals-are-the-key-to-future-defense-architectures>

8. Molex. "Mastering SWaP-C: Connector Strategies for Defense and Beyond."
   <https://www.molex.com/en-us/blog/defining-swap-c>

9. Criteria Labs. "SWaP-C for High-Reliability and Harsh Environment Applications."
   <https://www.criterialabs.com/resource/blog-swap-c-for-high-reliability-and-harsh-environment-applications/>

10. Digital Systems Engineering. "What is SWaP-C?"
    <https://www.digitalsys.com/what-is-swap-c/>

11. PNI Sensor. "SWaP-C: Size Weight Power and Cost."
    <https://www.pnisensor.com/swap-c-size-weight-power-and-cost/>

12. Vicor. "Reducing EV Weight with High-Density Power Modules."
    <https://www.vicorpower.com/resource-library/articles/automotive/reducing-ev-weight>

13. Vicor. "BEVs Weight Problems Can't Be Solved with Traditional Approaches."
    <https://www.vicorpower.com/resource-library/articles/automotive/bevs-have-a-weight-problem>

14. NASA. "NASA Systems Engineering Handbook."
    <https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf>

15. NASA. "Resource Management and Contingencies." NTRS.
    <https://ntrs.nasa.gov/api/citations/20120013284/downloads/20120013284.pdf>

16. AIAA. "ANSI/AIAA S-120A-2015: Mass Properties Control for Space Systems."
    <https://arc.aiaa.org/doi/10.2514/4.103858.001>

17. Aerospace Corporation. "Weight Watchers: Keeping Track of Vehicle Mass Properties."
    <https://aerospace.org/story/weight-watchers-keeping-track-vehicle-mass-properties>

18. NanoSTAR. "Systems Engineering Budgets."
    <https://nanostar-project.gitlab.io/main/source/preliminary-design/systems.html>

19. PTC. "DO-178C and DO-254 Explained."
    <https://www.ptc.com/en/blogs/alm/do178c-and-do254-explained>

20. Synopsys. "What is ISO 26262?"
    <https://www.synopsys.com/glossary/what-is-iso-26262.html>

21. ECSS. "ECSS-E-ST-10C Rev.1: System Engineering General Requirements." February 2017.
    <https://ecss.nl/standard/ecss-e-st-10c-rev-1-system-engineering-general-requirements-15-february-2017/>

22. Digital Systems Engineering. "MIL-STD-810 Environmental Testing Standards for Rugged Electronics."
    <https://www.digitalsys.com/mil-std-810-environmental-testing-standards-for-rugged-electronics/>

23. Military Embedded Systems. "Enabling SWaP-Optimized EW Solutions Through Accurate FPGA Power Modeling."
    <https://militaryembedded.com/radar-ew/signal-processing/enabling-swap-optimized-ew-solutions-through-accurate-fpga-power-modeling>

24. Quankang. "IEC 60601-1 Safety Standards Explained for Power Supply."
    <https://quankang-cn.com/iec-60601-1-safety-standards-explained-for-power-supply/>

25. Galorath. "Optimizing SWaP-C in Defense and Aerospace: Strategies for 2025."
    <https://galorath.com/blog/optimizing-swap-c-defense-aerospace-2025/>

26. Galorath. "SEER Platform."
    <https://galorath.com/seer/>

27. MDPI Sensors. "MBSE for Trade-Off Analysis."
    <https://www.mdpi.com/1424-8220/21/9/3201>

28. Electronic Design. "Advanced Cooling Technologies Overcome Thermal Management Challenges in
    Rugged System Design to Optimize SWaP-C."
    <https://www.electronicdesign.com/technologies/industrial/boards/article/55233125/advanced-cooling-technologies-overcome-thermal-management-challenges-in-rugged-system-design-to-optimize-swap-c>

29. Advanced Cooling Technologies. "Defense Thermal Solutions."
    <https://www.1-act.com/industries/defense/>

30. Tektronix. "2.5D/3D Packaging."
    <https://www.tek.com/en/component-solutions/2-5d-3d-packaging>

31. DigiKey. "Minimizing IoT Sensor Node Power Consumption."
    <https://www.digikey.com/en/articles/minimizing-iot-sensor-node-power-consumption>

32. CEVA. "2025 Edge AI Technology Report."
    <https://www.ceva-ip.com/wp-content/uploads/2025-Edge-AI-Technology-Report.pdf>

33. Promwad. "Edge AI in Embedded Devices 2025."
    <https://promwad.com/news/edge-ai-embedded-devices-2025>

34. Voler Systems. "Building FDA Approved Wearable Medical Devices."
    <https://www.volersystems.com/blog/building-better-wearable-device-getting-cleared-fda-walt-maclay>

35. DAU / AcqNotes. "Preliminary Design Review."
    <https://aaf.dau.edu/aaf/mca/pdr/>

36. Barnard Microsystems. "UAV Design Guidelines."
    <https://barnardmicrosystems.com/UAV/uav_design/guidelines.html>

37. Connector Supplier. "Soldier Wearable Technologies Advance Military Operations."
    <https://connectorsupplier.com/soldier-wearable-technologies-advance-military-operations/>

38. ExecutiveBiz. "Galorath SEER Suite for SWaP-C Optimization." January 2025.
    <https://executivebiz.com/2025/01/galorath-matt-mcdonald-swap-c-optimization-seer/>
