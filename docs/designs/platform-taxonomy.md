# Embodied AI Platform Registry

**Date**: 2026-04-03
**Revised**: 2026-04-03 — full gap analysis pass: added healthcare non-surgical,
prosthetics/assistive, waste/recycling, food processing, underground/pipe,
ship/hull, forestry, snow/ice, data center/telecom, mining, oil/gas, aerospace
manufacturing, textile, municipal services; renamed to Platform Registry
**Status**: Draft — enumeration and attribute analysis complete, pending schema design

## Purpose

Define a comprehensive, file-based **platform registry** that enables the embodied
AI architect to infer the correct platform from low-information prompts like
"edge device running edge detection" or "autonomous sprayer for vineyards".

The current system has 3 domain templates (drone, ugv, robot\_arm) with ~12 keywords
each. This is insufficient — "edge device" matches nothing. We need a dense registry
of ~290 platform definitions with rich keyword sets and attribute ranges so that
even vague prompts land on a reasonable starting point.

The registry serves as the **canonical catalog** for the agentic qualification
pipeline: when a user describes a goal, the matching engine scores it against every
entry in the registry, returns ranked candidates, and pre-fills the system spec
with realistic attribute ranges for the matched platform.

---

## 1. Platform Enumeration

### 1.1 Aerial (16 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `aerial.multirotor_delivery` | Multirotor Delivery Drone | Last-mile package delivery, 2-5kg payload | Payload drop mechanism, geofencing |
| `aerial.multirotor_inspection` | Multirotor Inspection Drone | Infrastructure/building/tower inspection | Zoom camera, GPS-denied flight |
| `aerial.multirotor_racing` | FPV Racing Drone | High-speed FPV racing and freestyle | <5ms perception latency, analog video |
| `aerial.multirotor_photography` | Cinematography Drone | Film/real-estate/event aerial photography | Gimbal stabilization, 4K+ video |
| `aerial.multirotor_search_rescue` | SAR Drone | Search and rescue, disaster response | Thermal imaging, loudspeaker, drop payload |
| `aerial.fixed_wing_survey` | Fixed-Wing Survey UAV | Mapping, photogrammetry, corridor survey | Long endurance, large area coverage |
| `aerial.fixed_wing_cargo` | Fixed-Wing Cargo UAV | Long-range heavy cargo transport | >5kg payload, >60min endurance |
| `aerial.fixed_wing_military_isr` | Military ISR UAV | Intelligence/surveillance/reconnaissance | EO/IR, SIGINT, long loiter |
| `aerial.agricultural_sprayer` | Agricultural Spray Drone | Precision crop spraying | Tank/nozzle, GPS-guided swath |
| `aerial.micro_indoor` | Micro Indoor Drone | Sub-250g warehouse/confined-space inspection | Prop guards, optical flow nav |
| `aerial.vtol_air_taxi` | eVTOL Air Taxi | Urban air mobility, passenger transport | Passenger safety SIL-4, redundant flight control |
| `aerial.tethered_persistent` | Tethered Surveillance Drone | Persistent aerial overwatch via power tether | Unlimited endurance, tethered power |
| `aerial.swarm_surveillance` | Surveillance Drone Swarm | Coordinated multi-drone area coverage | Swarm coordination, mesh comms, distributed perception |
| `aerial.swarm_counter_uas` | Counter-UAS Interceptor Swarm | Intercept/neutralize hostile drones | Target tracking, high-speed intercept, coordinated engagement |
| `aerial.swarm_light_show` | Drone Light Show Swarm | Coordinated aerial light displays | Precision formation, RTK-GPS, choreography engine |
| `aerial.high_altitude_pseudo_satellite` | HAPS / Solar Drone | Stratospheric long-endurance comms/ISR relay | Solar power, months-long endurance, >60k ft |

### 1.2 Ground — Wheeled (18 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `ground_wheeled.warehouse_amr` | Warehouse AMR | Shelf-to-person, goods transport | SLAM in structured aisles, fleet coordination |
| `ground_wheeled.sidewalk_delivery` | Sidewalk Delivery Robot | Last-mile sidewalk package delivery | Pedestrian avoidance, curb navigation |
| `ground_wheeled.security_patrol` | Security Patrol Robot | Campus/facility autonomous patrol | 24/7 operation, ANPR, anomaly detection |
| `ground_wheeled.hospital_logistics` | Hospital Logistics Robot | Medication/supply/lab sample delivery | Elevator interface, sterile corridors |
| `ground_wheeled.retail_inventory` | Retail Inventory Robot | Shelf scanning, stocktaking | Barcode/RFID reading, aisle navigation |
| `ground_wheeled.cleaning_commercial` | Commercial Cleaning Robot | Floor scrubbing, vacuuming large areas | Wet/dry operation, obstacle mapping |
| `ground_wheeled.autonomous_truck` | Autonomous Truck (L4) | Highway freight, hub-to-hub | Highway ODD, platooning, >80k lbs GVW |
| `ground_wheeled.autonomous_shuttle` | Autonomous Shuttle | Low-speed campus/district people mover | Passenger safety, fixed route, <25 mph |
| `ground_wheeled.autonomous_car` | Autonomous Car (Robotaxi) | L4 urban ride-hailing | Full urban ODD, passenger comfort |
| `ground_wheeled.mining_haul` | Autonomous Mining Hauler | Open-pit autonomous haulage | 200+ ton payload, dust/vibration, no-GPS zones |
| `ground_wheeled.forklift_agv` | Autonomous Forklift | Pallet transport, warehouse racking | Fork manipulation, 3D pallet detection |
| `ground_wheeled.hotel_service` | Hotel Service Robot | Room delivery, concierge assistance | Elevator integration, guest interaction |
| `ground_wheeled.restaurant_service` | Restaurant Service Robot | Food delivery table-to-table | Tray carrying, dynamic crowd navigation |
| `ground_wheeled.airport_baggage` | Airport Baggage Vehicle | Autonomous baggage tug on apron/tarmac | GPS+LiDAR on apron, aircraft proximity safety |
| `ground_wheeled.floor_scrubber_industrial` | Industrial Floor Scrubber | Autonomous floor cleaning in factories/warehouses | Large area coverage, wet floor handling |
| `ground_wheeled.swarm_warehouse` | Warehouse Robot Swarm | Kiva-style coordinated shelf-moving fleet | Central planner, swarm coordination, 100+ units |
| `ground_wheeled.autonomous_bus` | Autonomous City Bus | Full-size autonomous public transit | Passenger capacity >40, urban ODD, ADA compliance |
| `ground_wheeled.golf_cart_autonomous` | Autonomous Golf Cart / Campus Shuttle | Low-speed open-air transport | Low speed, simple ODD, passenger interaction |

### 1.3 Ground — Legged (8 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `ground_legged.quadruped_inspection` | Quadruped Inspection Robot | Industrial plant walkthrough/inspection | Stair climbing, confined spaces, sensor payload |
| `ground_legged.quadruped_security` | Quadruped Security Patrol | Perimeter patrol on rough terrain | All-weather, terrain adaptation, thermal camera |
| `ground_legged.quadruped_research` | Quadruped Research Platform | General-purpose legged research | ROS2, configurable payload, open SDK |
| `ground_legged.quadruped_military` | Military Quadruped | Load carrying, reconnaissance in rough terrain | Heavy payload, quiet operation, GPS-denied nav |
| `ground_legged.biped_humanoid_warehouse` | Warehouse Humanoid | Humanoid for pick/pack/place in human spaces | Human-scale manipulation, stair climbing |
| `ground_legged.biped_humanoid_companion` | Companion Humanoid | Social/eldercare/assistant humanoid | Natural language, gesture, emotional modeling |
| `ground_legged.biped_humanoid_construction` | Construction Humanoid | Humanoid for construction site tasks | Heavy-duty manipulation, harsh environment |
| `ground_legged.hexapod_rough_terrain` | Hexapod Explorer | All-terrain exploration, disaster sites | 6-leg stability, payload, extreme terrain |

### 1.4 Ground — Tracked (6 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `ground_tracked.bomb_disposal` | EOD / Bomb Disposal Robot | Explosive ordnance disposal | Manipulator arm, X-ray, teleoperated + AI assist |
| `ground_tracked.military_ugv` | Military UGV | Armed/logistics ground combat | Weapon mount, C4ISR integration, ruggedized |
| `ground_tracked.exploration_rover` | Exploration Rover | Planetary/cave/nuclear site exploration | Extreme autonomy (signal delay), radiation tolerance |
| `ground_tracked.demolition_robot` | Remote Demolition Robot | Controlled demolition in hazardous areas | Hydraulic breaker/crusher, heavy vibration tolerance |
| `ground_tracked.firefighting_robot` | Firefighting Robot | Fire suppression in hazardous environments | Thermal protection, water cannon, thermal imaging |
| `ground_tracked.nuclear_decommission` | Nuclear Decommissioning Robot | Radioactive site cleanup | Radiation-hardened electronics, remote teleoperation |

### 1.5 Manipulation — Industrial (22 platforms)

Organized by process type. Each robot handles fundamentally different loads,
speeds, precisions, and environmental conditions.

#### Material Handling & Assembly

| ID | Platform | Description | Load Class | Key Differentiator |
|----|----------|-------------|-----------|-------------------|
| `manipulation.cobot_tabletop` | Tabletop Cobot | Desktop collaborative assembly | light (1-10kg) | Force-limited, power-and-force limiting mode |
| `manipulation.cobot_industrial` | Industrial Cobot | Floor-mount collaborative | medium (10-35kg) | Dual nervous system, speed/separation monitoring |
| `manipulation.industrial_caged` | Caged Industrial Arm | High-speed behind safety cage | medium-heavy (10-200kg) | Maximum speed/force, no human proximity limits |
| `manipulation.scara_assembly` | SCARA Assembly Robot | 4-axis electronics/small-part assembly | micro-light (0.1-5kg) | High-speed, high-repeatability, planar motion |
| `manipulation.delta_pick_place` | Delta Pick-and-Place | Parallel robot, 100+ picks/min | micro-light (0.01-3kg) | Extreme speed, vision-guided sorting |
| `manipulation.palletizing_robot` | Palletizing Robot | End-of-line pallet stacking | heavy (50-500kg) | Pallet pattern planning, long reach |
| `manipulation.heavy_forge_press` | Heavy Press / Forge Tending | Loading 1-10 ton billets into forge/press | ultra-heavy (1,000-10,000kg) | Extreme payload, heat-resistant tooling, overhead gantry |
| `manipulation.dual_arm_assembly` | Dual-Arm Assembly | Bimanual coordinated manipulation | light-medium (1-20kg) | Coordinated bi-arm planning, force balancing |
| `manipulation.mobile_manipulator` | Mobile Manipulator | Arm mounted on mobile base | light-medium (3-25kg) | Combined navigation + manipulation planning |
| `manipulation.bin_picking` | Bin Picking Robot | Pick random parts from bulk bin | light (0.1-10kg) | 3D vision in clutter, grasp planning on unknown pose |
| `manipulation.smt_pick_place` | SMT Pick-and-Place | PCB component placement | micro (<0.01kg) | Sub-0.05mm accuracy, 20k+ CPH, feeder integration |

#### Process Robots (tool-on-arm)

| ID | Platform | Description | Load Class | Key Differentiator |
|----|----------|-------------|-----------|-------------------|
| `manipulation.welding_arc` | Arc Welding Robot | MIG/TIG/stick arc welding | tool (5-15kg torch) | Seam tracking, weld pool monitoring, fume tolerance |
| `manipulation.welding_spot` | Spot Welding Robot | Resistance spot welding (automotive) | tool-heavy (80-200kg gun) | Heavy weld gun, electrode wear monitoring |
| `manipulation.welding_laser` | Laser Welding Robot | Precision laser welding/cutting | tool (10-30kg head) | Beam focus control, zero-contact, high speed |
| `manipulation.painting_robot` | Painting / Coating Robot | Automotive/industrial surface coating | tool (5-15kg gun) | ATEX zone, path coverage, spray pattern control |
| `manipulation.dispensing_robot` | Dispensing Robot | Adhesive, sealant, potting compound | tool (3-10kg head) | Bead consistency, flow rate control, cure time |
| `manipulation.grinding_polishing` | Grinding / Polishing / Deburring | Surface finishing, edge deburring | tool + force (5-30kg) | Force-controlled, compliant contact, dust extraction |
| `manipulation.laser_cutting` | Laser Cutting Robot | 3D laser cutting, trimming | tool (15-40kg head) | 3D path, kerf compensation, fume extraction |
| `manipulation.waterjet_cutting` | Waterjet Cutting Robot | Abrasive waterjet cutting | tool (20-50kg head) | 60k PSI, garnet abrasive, any material |
| `manipulation.cnc_tending` | Machine Tending Robot | CNC/press/injection mold loading | medium (5-50kg parts) | Part orientation, machine integration (OPC-UA) |
| `manipulation.packaging_robot` | Packaging / Case Packing | Product into box/carton/wrap | light-medium (1-30kg) | Mixed-SKU, carton erecting, film wrapping |
| `manipulation.composite_layup` | Composite Layup Robot | Carbon fiber / prepreg tape laying | tool (20-50kg head) | Ply-by-ply, compaction force, laser heating |

#### Domestic / Light Manipulation

| ID | Platform | Description | Load Class | Key Differentiator |
|----|----------|-------------|-----------|-------------------|
| `manipulation.laundry_folding` | Laundry Folding Robot | Fold clothes, sort, load/unload baskets | deformable-light (0.1-5kg) | Deformable object handling, fabric classification |
| `manipulation.kitchen_robot` | Kitchen / Food Prep Robot | Cooking, plating, ingredient handling | deformable-light (0.1-3kg) | Food-safe, varied object shapes, temperature |
| `manipulation.elderly_assist_arm` | Assistive Manipulation Arm | Fetch objects, feed, assist with daily tasks | light (0.1-3kg) | Gentle force control, human-safe, voice-commanded |

### 1.5b Manipulation — Surgical (12 platforms)

Surgical robots are a distinct specialization — each surgical domain has
radically different workspace geometry, precision requirements, instrument
types, and imaging modalities. A single "surgical robot" entry is insufficient.

| ID | Platform | Description | Precision | Key Differentiator |
|----|----------|-------------|----------|-------------------|
| `surgical.laparoscopic` | Laparoscopic Surgery Robot | Abdominal MIS (da Vinci-class) | 0.1-1mm | Trocar access, 4-arm, wristed instruments, 3D stereo endoscope |
| `surgical.orthopedic` | Orthopedic Surgery Robot | Joint replacement, bone cutting | 0.5-1mm | Haptic feedback, bone registration, CT-guided, high-force cutting |
| `surgical.spinal` | Spinal Surgery Robot | Pedicle screw placement, spinal fusion | 0.5-1mm | Fluoroscopy/CT navigation, screw trajectory planning |
| `surgical.neurosurgery` | Neurosurgery Robot | Brain biopsy, DBS electrode placement, tumor resection | 0.01-0.1mm | Stereotactic frame or frameless, MRI-guided, sub-mm mandatory |
| `surgical.ophthalmic` | Ophthalmic Surgery Robot | Retinal membrane peel, cataract, glaucoma | 0.001-0.01mm | Micron-level, tremor cancellation, microscope-integrated |
| `surgical.cardiac` | Cardiac Surgery Robot | Coronary bypass, valve repair | 0.1-0.5mm | Beating-heart motion compensation, 4D tracking |
| `surgical.ent` | ENT Surgery Robot | Transoral, sinus, skull base | 0.1-0.5mm | Narrow-access, flexible instruments, endoscopic |
| `surgical.urological` | Urological Surgery Robot | Prostatectomy, nephrectomy | 0.1-1mm | Confined pelvic workspace, nerve-sparing planning |
| `surgical.dental` | Dental Surgery Robot | Implant placement, guided surgery | 0.1-0.5mm | Jaw registration, drill guide, patient awake |
| `surgical.microsurgery` | Microsurgery Robot | Vessel anastomosis, nerve repair | 0.01-0.1mm | Microscope-integrated, tremor filter, sub-mm sutures |
| `surgical.endovascular` | Endovascular / Interventional Robot | Catheter navigation, stent deployment | 1-2mm | Fluoroscopy-guided, force-limited catheter, radiation shielding |
| `surgical.radiation_therapy` | Radiation Therapy Robot | Real-time tumor tracking (CyberKnife-class) | 0.5-1mm | 6-DOF linac mount, respiratory gating, real-time image guidance |

### 1.5c Manipulation — Veterinary & Husbandry (8 platforms)

| ID | Platform | Description | Load Class | Key Differentiator |
|----|----------|-------------|-----------|-------------------|
| `veterinary.surgical_robot` | Veterinary Surgical Robot | Minimally invasive animal surgery | light (instruments) | Adapted ergonomics for animal anatomy, laparoscopic |
| `veterinary.ultrasound_scanner` | Livestock Ultrasound Robot | Automated pregnancy/health scanning | contact (probe) | Animal restraint integration, automatic organ detection |
| `veterinary.sheep_shearer` | Robotic Sheep Shearer | Automated wool shearing | deformable-contact | Animal shape adaptation, non-planar surface following |
| `veterinary.hoof_trimmer` | Robotic Hoof Trimmer | Cattle hoof care automation | contact-force | 3D hoof scanning, precision cutting, animal comfort |
| `veterinary.vaccination_robot` | Livestock Vaccination Robot | Automated injection in herds | contact (needle) | Animal ID, injection site targeting, dose tracking |
| `veterinary.poultry_processing` | Poultry Processing Robot | Evisceration, deboning, portioning | light (1-5kg) | High speed, food-safe, carcass variation handling |
| `veterinary.egg_collection` | Egg Collection Robot | Automated egg gathering from nests | fragile-light (<0.1kg) | Gentle handling, crack detection, nest navigation |
| `veterinary.aquaculture_sampling` | Fish Sampling Robot | Automated fish health measurement | live-animal | Gentle fish handling, size/weight measurement, non-lethal |

### 1.5d Manipulation — Laboratory Automation (12 platforms)

Lab automation is a major embodied AI domain — wet labs, biobanks, and
high-throughput screening facilities run thousands of robotic operations daily.

| ID | Platform | Description | Load Class | Key Differentiator |
|----|----------|-------------|-----------|-------------------|
| `lab.liquid_handler` | Liquid Handling Robot | Pipetting, dilution, reagent dispensing | micro-fluid (µL-mL) | 96/384/1536-well plates, nL precision, contamination-free |
| `lab.plate_handler` | Microplate Handler / Shuttle | Move plates between instruments | light (0.05-0.3kg) | Stackers, hotel integration, barcode tracking |
| `lab.colony_picker` | Colony Picker | Pick microbial colonies from agar | micro-contact | Camera-guided, pin/tip selection, sterile |
| `lab.sample_prep` | Sample Preparation Robot | Centrifuge, vortex, aliquot | light (tubes/vials) | Multi-instrument orchestration, protocol engine |
| `lab.hts_screening` | High-Throughput Screening Robot | Compound library screening | micro-fluid | 100k+ compounds/day, assay integration, hit detection |
| `lab.biobank_storage` | Biobank Sample Management | Automated -80C/LN2 sample storage/retrieval | light (vial/tube) | Cryogenic operation, sample chain-of-custody, 2D barcode |
| `lab.histology_prep` | Histology / Pathology Prep Robot | Tissue sectioning, slide staining | fragile-micro | Microtome integration, stain protocols, slide tracking |
| `lab.pcr_setup` | PCR Setup Robot | Automated PCR plate preparation | micro-fluid (µL) | Template/primer dispensing, contamination prevention |
| `lab.mass_spec_prep` | Mass Spec Sample Prep Robot | Protein digest, LC-MS prep | micro-fluid | Magnetic bead handling, solvent dispensing, evaporation |
| `lab.compound_management` | Compound Management System | Chemical library storage and dispensing | micro-fluid/solid | Desiccated storage, acoustic dispensing, cherry-picking |
| `lab.cell_culture` | Cell Culture Robot | Automated cell passaging, feeding, imaging | sterile-fluid | Laminar flow integration, incubator access, confluency detection |
| `lab.dna_synthesis` | DNA / Oligo Synthesis Robot | Automated nucleotide synthesis | micro-fluid (nL-µL) | Phosphoramidite chemistry, column management, purification |

### 1.6 Marine — Surface (6 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `marine_surface.usv_survey` | Survey USV | Hydrographic survey, bathymetry | Multibeam sonar, RTK-GPS, long endurance |
| `marine_surface.usv_patrol` | Patrol USV | Harbor/coast patrol and surveillance | Radar, EO/IR, AIS integration |
| `marine_surface.usv_cargo` | Autonomous Cargo Ferry | Short-range autonomous cargo transport | Large payload, docking automation |
| `marine_surface.usv_aquaculture` | Aquaculture USV | Fish farm monitoring and feeding | Underwater camera, feeding mechanism |
| `marine_surface.usv_environmental` | Environmental Monitoring USV | Water quality, oil spill detection | Chemical sensors, water sampling |
| `marine_surface.swarm_minesweep` | Mine Countermeasure USV Swarm | Coordinated naval mine detection/neutralization | Swarm coordination, sonar, expendable |

### 1.7 Marine — Underwater (6 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `marine_underwater.auv_inspection` | Inspection AUV | Pipeline/hull/infrastructure inspection | Close-range sonar, camera, dead reckoning |
| `marine_underwater.auv_survey` | Survey AUV | Seabed mapping, oceanography | Side-scan sonar, long endurance |
| `marine_underwater.auv_mine_countermeasures` | MCM AUV | Naval mine detection | High-res sonar, classification AI |
| `marine_underwater.rov_deep_sea` | Deep-Sea ROV | Deep-sea exploration, intervention | Manipulator arms, tethered, >3000m depth |
| `marine_underwater.glider` | Underwater Glider | Long-endurance ocean monitoring | Buoyancy-driven, months-long endurance |
| `marine_underwater.swarm_survey` | AUV Survey Swarm | Coordinated seabed survey fleet | Acoustic comms, distributed mapping |

### 1.8 Space (5 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `space.planetary_rover` | Planetary Rover | Mars/lunar surface exploration | Extreme autonomy (signal delay), rad-hard |
| `space.orbital_servicing` | Orbital Servicing Robot | Satellite repair, refueling, de-orbit | Microgravity manipulation, rendezvous/dock |
| `space.free_flying_inspector` | Free-Flying Inspector | ISS/station exterior inspection | Micro-thrusters, zero-G navigation |
| `space.lunar_lander` | Autonomous Lander | Precision planetary landing | Terrain-relative navigation, hazard avoidance |
| `space.debris_removal` | Space Debris Removal | Active debris removal from orbit | Non-cooperative target capture, deorbit |

### 1.9 Stationary Sensing & Edge Vision (24 platforms)

Surveillance and monitoring is the largest cross-cutting use case in embodied AI.
It spans security, safety, health, environmental, infrastructure, and counting
applications. The common thread: persistent sensing + edge inference + alerting,
with no actuation beyond PTZ.

Organized by surveillance **purpose**, not just sensor type.

#### Security & Access Control

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `surveillance.perimeter_intrusion` | Perimeter Intrusion Detection | Fence-line / boundary breach detection | Multi-sensor (fence vibration, IR beam, radar, camera), zone-based alerting |
| `surveillance.access_control_vision` | Visual Access Control | Face/badge/vehicle gate entry | Face recognition, ANPR, tailgating detection, turnstile integration |
| `surveillance.campus_security` | Campus / Facility Surveillance | Multi-camera security monitoring | VMS integration, person re-ID across cameras, anomaly detection |
| `surveillance.critical_infrastructure` | Critical Infrastructure Surveillance | Power plant / dam / water treatment monitoring | 24/7, tamper detection, SCADA integration, multi-spectral |
| `surveillance.prison_monitoring` | Correctional Facility Monitoring | Inmate/yard/cell block surveillance | Behavior analytics, fight detection, headcount, contraband detection |

#### Public Safety & Situational Awareness

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `surveillance.crowd_analytics` | Crowd Density & Flow Analytics | Stadium / transit / public space crowd monitoring | Density estimation, flow heatmap, crush risk alerting, evacuation routing |
| `surveillance.gunshot_detection` | Acoustic Threat Detection | Gunshot / explosion / glass-break detection | Microphone array, acoustic classification, triangulation, <2s alert |
| `surveillance.city_wide_video` | City-Wide Video Analytics | Urban CCTV network with AI analytics | Federated edge inference, cross-camera tracking, incident detection |
| `surveillance.disaster_situational` | Disaster Situational Awareness | Post-earthquake / flood / wildfire damage assessment | Aerial + ground fusion, structural damage classification, survivor detection |
| `surveillance.border_surveillance` | Border / Coastline Surveillance | Long-range border monitoring | Radar + EO/IR + seismic, wide-area, day/night, classification at range |

#### Traffic & Transportation Monitoring

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `edge_vision.smart_camera_traffic` | Traffic Smart Camera | Intersection monitoring, ANPR, flow analysis | Weatherproof, ANPR, multi-lane counting, signal integration |
| `surveillance.highway_monitoring` | Highway / Tunnel Traffic Monitoring | Incident detection, wrong-way driver, congestion | Long-range, fog/rain tolerance, queue detection, V2X integration |
| `surveillance.parking_occupancy` | Parking Occupancy Monitor | Space-by-space availability detection | Overhead or per-space sensor, real-time availability API |
| `surveillance.rail_track_monitor` | Rail Track / Platform Monitor | Track intrusion, platform edge safety | Track obstacle detection, gap monitoring, passenger flow |

#### Environmental & Ecological Monitoring

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `surveillance.wildfire_detection` | Wildfire Detection Network | Early smoke/fire detection over large areas | Distributed tower-mount cameras, IR + visible, smoke classification |
| `surveillance.air_quality` | Air Quality Monitoring Station | Particulate, gas, pollen sensing with AI | PM2.5/PM10/O3/NO2, source attribution, forecast integration |
| `surveillance.wildlife_camera_trap` | Wildlife Camera Trap / Census | Biodiversity monitoring, population counting | Motion-triggered, animal species classification, weather-sealed |
| `surveillance.ocean_monitoring` | Coastal / Ocean Environmental Monitor | Wave height, water quality, marine life tracking | Radar + camera + chemical sensor, storm surge alerting |
| `surveillance.wastewater_epidemiology` | Wastewater Epidemiological Monitor | Pathogen surveillance via sewage sampling | Automated sampling, PCR/sequencing integration, outbreak early warning |

#### Counting & Census (dedicated perception task)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `surveillance.people_counting` | People Counting / Occupancy | Retail footfall, building occupancy, event attendance | Overhead stereo/ToF, bi-directional, occupancy limit alerting |
| `surveillance.vehicle_counting` | Vehicle Counting & Classification | Traffic studies, toll plazas, weigh stations | Per-lane count, vehicle class (car/truck/bus/bike), speed |
| `surveillance.livestock_counting` | Livestock Counting & Identification | Herd census, individual animal ID | Aerial or gate-mounted, ear tag / face / body ID, count reconciliation |

#### Multi-Spectral & Sensor Fusion

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `surveillance.multispectral_fusion` | Multi-Spectral Surveillance Platform | Fused visible + thermal + SWIR + radar | Sensor fusion engine, target detection across spectra, day/night/fog |
| `surveillance.ground_radar` | Ground Surveillance Radar | Wide-area moving target detection | 360° scan, Doppler classification, camera slew-to-cue, 10km+ range |

#### Industrial & Workplace Monitoring

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `edge_vision.industrial_quality` | Industrial Quality Inspection | Production line defect detection | High-res, strobe lighting, sub-mm defect detection |
| `surveillance.ppe_compliance` | PPE Compliance Monitor | Hard hat / vest / goggle detection on workers | Real-time alerting, zone-based rules, compliance reporting |
| `surveillance.ergonomic_monitor` | Workplace Ergonomic Monitor | Posture analysis, repetitive motion risk | Skeleton tracking, risk scoring, fatigue estimation |
| `surveillance.process_monitor` | Industrial Process Monitor | Visual process parameter monitoring | Gauge/dial reading, flame color, fill level, conveyor speed |

#### Structural & Infrastructure Health

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `surveillance.structural_health` | Structural Health Monitor | Bridge / building / dam vibration and crack monitoring | Accelerometers + strain gauges + cameras, modal analysis, long-term drift |
| `surveillance.facade_inspection` | Building Facade Inspection | Exterior cladding, window, balcony condition | Thermal + visible, crack/spalling detection, BIM integration |

#### Retail & Commercial

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `edge_vision.smart_camera_retail` | Retail Smart Camera | Checkout-free store, customer analytics | Person re-ID, action recognition, planogram compliance |
| `surveillance.loss_prevention` | Loss Prevention / Shrinkage Monitor | Theft detection, suspicious behavior | Concealment detection, POS exception correlation, alert escalation |
| `surveillance.shelf_analytics` | Shelf / Planogram Analytics | Product availability, placement compliance | SKU recognition, out-of-stock detection, compliance scoring |

#### Residential / Consumer

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `edge_vision.smart_doorbell` | Smart Doorbell / Home Camera | Home security with person/package detection | Battery or wired, cloud + edge hybrid, familiar face |
| `edge_vision.baby_monitor_ai` | AI Baby Monitor | Infant breathing, position, cry detection | Non-contact vital signs, sleep analytics, SIDS risk alerting |
| `edge_vision.elderly_monitor` | Elderly / Fall Risk Monitor | In-home activity and fall monitoring | Privacy-preserving (radar/skeleton, no video), inactivity alerting |

#### Specialized Sensing

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `edge_vision.medical_imaging_edge` | Medical Imaging Edge Device | Bedside/point-of-care AI diagnostic | FDA-cleared, HIPAA, DICOM integration |
| `edge_vision.edge_vision_box` | General Edge AI Appliance | Multi-purpose edge inference box | Multi-stream, model-agnostic, RTSP/ONVIF |
| `edge_vision.sports_analytics` | Sports Analytics Camera | Player tracking, event detection | Multi-camera stitching, real-time broadcast overlay |
| `surveillance.acoustic_monitoring` | Industrial Acoustic Monitor | Machine health via sound analysis | Microphone array, anomaly detection, bearing/motor fault |
| `surveillance.radiation_monitor` | Radiation Monitoring Network | Area radiation dose rate monitoring | Gamma spectroscopy, neutron detection, ALARA alerting |

### 1.10 Wearable / Portable (12 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `wearable.ar_headset` | AR/VR Headset | Spatial computing, mixed reality | 6-DOF tracking, hand tracking, <20ms motion-to-photon |
| `wearable.smart_glasses` | Smart Glasses | Hands-free AR assistance | Lightweight, all-day wear, voice + visual |
| `wearable.exoskeleton_industrial` | Industrial Exoskeleton | Load-bearing, fatigue reduction | Torso/arm/leg assist, 8-hour shift |
| `wearable.exoskeleton_medical` | Medical Exoskeleton | Rehabilitation, mobility assist | Gait training, adaptive impedance control |
| `wearable.body_camera_ai` | AI Body Camera | Law enforcement, hands-free inspection | Always-on recording, real-time detection |
| `wearable.cgm_ai` | Continuous Glucose Monitor + AI | Glucose trend prediction, insulin dosing advice | Biocompatible sensor, predictive alerts, FDA Class II |
| `wearable.smart_insulin_pump` | Smart Insulin Pump | Closed-loop automated insulin delivery | Continuous dosing, CGM integration, SIL-2+ |
| `wearable.cardiac_monitor` | Wearable Cardiac Monitor | Continuous ECG/arrhythmia detection | Multi-lead ECG, AF detection, clinical-grade |
| `wearable.fall_detection` | Fall Detection Device | Elderly fall detection and alert | Accelerometer/gyro, auto-alert, low power |
| `wearable.seizure_detector` | Seizure Detection Wearable | Epilepsy seizure detection and alert | EDA/accelerometer, pre-ictal detection |
| `wearable.respiratory_monitor` | Respiratory Monitor | SpO2, respiratory rate, sleep apnea | Pulse ox, continuous monitoring, clinical accuracy |
| `wearable.tremor_monitor` | Tremor / Movement Disorder Monitor | Parkinson's tremor tracking and medication timing | IMU-based tremor quantification, medication reminders |

### 1.11 Consumer / Home (8 platforms, incl. 1.31 extensions)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `consumer.robot_vacuum` | Robot Vacuum | Autonomous floor cleaning | SLAM, obstacle avoidance, dock return |
| `consumer.robot_lawn_mower` | Robot Lawn Mower | Autonomous lawn care | Boundary wire or vision-based, rain detection |
| `consumer.pool_cleaner` | Autonomous Pool Cleaner | Autonomous pool cleaning | Underwater navigation, wall climbing |
| `consumer.home_companion` | Home Companion Robot | Social robot, eldercare, family assistant | NLU, face recognition, emotional modeling |
| `consumer.educational_robot` | Educational Robot | STEM teaching, programmable platform | Block programming, sensors, extensible |
| `consumer.pet_robot` | Robot Pet / Entertainment | Companion entertainment robot | Personality engine, play behaviors |

### 1.12 Agriculture (14 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `agriculture.autonomous_tractor` | Autonomous Tractor | Plowing, seeding, tilling | RTK-GPS, implement control, large power |
| `agriculture.autonomous_harvester` | Autonomous Combine Harvester | Grain/crop harvesting | Yield mapping, auto-steering, header control |
| `agriculture.fruit_picker` | Fruit Picking Robot | Selective fruit/vegetable harvesting | Soft gripper, ripeness detection, arm reach |
| `agriculture.weeding_robot` | Precision Weeding Robot | Mechanical/laser/micro-spray weed removal | Per-plant detection, cm-level accuracy |
| `agriculture.pesticide_ground_sprayer` | Autonomous Ground Sprayer | Precision pesticide/herbicide application | Variable-rate nozzle, drift control, GPS-guided |
| `agriculture.crop_scout` | Crop Scouting Robot | Disease/pest/nutrient deficiency detection | Multispectral camera, NDVI, plant-level ID |
| `agriculture.livestock_monitor` | Livestock Monitoring Robot | Cattle/poultry health, behavior tracking | Animal ID, lameness detection, thermal imaging |
| `agriculture.dairy_milking` | Robotic Milking System | Autonomous dairy milking | Teat detection, gentle manipulation, hygiene |
| `agriculture.greenhouse_robot` | Greenhouse Robot | Indoor growing automation | Pruning, pollination, climate-controlled |
| `agriculture.vineyard_robot` | Vineyard Management Robot | Grape leaf analysis, selective harvesting | Row navigation, canopy analysis |
| `agriculture.aquaculture_feeder` | Aquaculture Feeding Robot | Fish farm automated feeding | Feed dispensing, biomass estimation |
| `agriculture.soil_sampler` | Soil Sampling Robot | Autonomous soil core collection | Drill mechanism, GPS-logged samples |
| `agriculture.irrigation_robot` | Irrigation Management Robot | Precision irrigation, moisture monitoring | Soil moisture sensing, valve control |
| `agriculture.biosurveillance_drone` | Crop Biosurveillance Drone | Aerial disease/pest detection over large areas | Multispectral/hyperspectral, fleet coverage |

### 1.13 Construction (7 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `construction.autonomous_excavator` | Autonomous Excavator | Earthmoving, grading, trenching | 3D terrain modeling, bucket control |
| `construction.bricklaying_robot` | Bricklaying Robot | Automated masonry | Mortar application, placement accuracy |
| `construction.3d_printing` | Construction 3D Printer | Additive construction (concrete/polymer) | Extrusion head, toolpath planning |
| `construction.rebar_tying` | Rebar Tying Robot | Reinforcement automation | Wire tying, rebar detection |
| `construction.site_survey_robot` | Construction Site Survey Robot | Progress monitoring, BIM comparison | 3D scanning, SLAM, BIM overlay |
| `construction.demolition_robot` | Construction Demolition Robot | Selective interior demolition | Hydraulic tools, dust/debris tolerance |
| `construction.autonomous_crane` | Autonomous Tower Crane | Automated load lifting and placement | Anti-sway, load path planning, wind compensation |

### 1.14 Logistics & Warehousing (14 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `logistics.parcel_sorter` | Parcel Sorting Robot | Last-mile hub package sorting | Barcode reading, tilt-tray/cross-belt, high throughput |
| `logistics.goods_to_person` | Goods-to-Person Shuttle | Shelf-moving robot (Kiva-style) | Central fleet planner, 100+ unit coordination |
| `logistics.autonomous_forklift` | Autonomous Forklift | Pallet handling in warehouse | Fork manipulation, 3D pallet detection, high racks |
| `logistics.autonomous_reach_truck` | Autonomous Reach Truck | High-rack narrow-aisle pallet storage | 10m+ lift height, narrow aisle navigation |
| `logistics.autonomous_pallet_jack` | Autonomous Pallet Jack | Low-level pallet transport | Low-profile, dock plate traversal |
| `logistics.truck_loading` | Truck Loading Robot | Loading/unloading trailers | Mixed-SKU handling, trailer mapping, floor-to-ceiling |
| `logistics.container_handler` | Container Handling Robot | Port container stacking/transport | Spreader lock, 40-ton capacity, GPS+LiDAR |
| `logistics.conveyor_pick` | Conveyor Pick Robot | Pick-from-conveyor to tote/box | Vision-guided, moving-belt tracking |
| `logistics.depalletizing` | Depalletizing Robot | Unstacking mixed pallets layer-by-layer | 3D box detection, suction/clamp gripper |
| `logistics.loading_dock` | Loading Dock Robot | Cross-dock transfer, dock-to-staging | Dock leveler traversal, mixed freight |
| `logistics.cold_storage` | Cold Storage Robot | Frozen/chilled warehouse operations | -25C operation, condensation handling |
| `logistics.e_commerce_fulfillment` | E-Commerce Fulfillment Robot | Pick-pack-ship for online orders | Item recognition, multi-bin picking |
| `logistics.mail_sorting` | Mail/Parcel Sorting System | High-speed letter/flat/parcel sorting | OCR, barcode, 10k+ items/hour |
| `logistics.fleet_yard_truck` | Autonomous Yard Truck | Container/trailer moving within yards | Outdoor GPS, coupling/uncoupling, 24/7 |

### 1.15 Energy & Utilities (6 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `energy.wind_turbine_inspector` | Wind Turbine Inspection Robot | Blade surface defect detection | Rope/crawler, blade-specific AI, 80m+ height |
| `energy.solar_panel_cleaner` | Solar Panel Cleaning Robot | Autonomous PV array cleaning | Water-free or minimal water, panel detection |
| `energy.pipeline_inspection` | Pipeline Inspection Robot (PIG) | Internal pipeline inspection | In-pipe navigation, ultrasonic/MFL sensing |
| `energy.power_line_inspector` | Power Line Inspection Drone/Robot | Transmission line and tower inspection | LiDAR, thermal hotspot, wire following |
| `energy.substation_patrol` | Substation Patrol Robot | Electrical substation monitoring | Thermal imaging, dial/gauge reading, EMI tolerant |
| `energy.nuclear_inspection` | Nuclear Facility Inspection Robot | Reactor/containment inspection | Radiation-hardened, remote teleoperation |

### 1.16 Entertainment & Hospitality (4 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `entertainment.animatronic` | Animatronic / Theme Park Robot | Life-like character animation | Multi-DOF face/body, show programming |
| `entertainment.tour_guide` | Tour Guide Robot | Museum/venue autonomous guide | NLU, wayfinding, crowd-aware navigation |
| `entertainment.telepresence` | Telepresence Robot | Remote presence, meetings, tours | Video call integration, remote driving |
| `entertainment.event_photographer` | Autonomous Event Photographer | Event/party photo/video capture | Face detection, composition, autonomous roaming |

### 1.17 Military & Defense — Swarm / Multi-Agent (6 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `military.swarm_recon` | Reconnaissance Drone Swarm | Distributed ISR over contested area | Mesh networking, resilient to attrition |
| `military.swarm_c_uas` | Counter-UAS Interceptor Swarm | Intercept incoming enemy drones/munitions | Threat classification, pursuit algorithms, expendable |
| `military.swarm_ground_logistics` | Ground Logistics Swarm | Supply convoy following lead vehicle | Leader-follower, convoy formation |
| `military.swarm_naval_surface` | Naval Surface Swarm | Coordinated surface patrol/interdiction | Maritime AIS, threat prioritization |
| `military.loitering_munition` | Loitering Munition | Autonomous target acquisition and engagement | Long loiter, target recognition, terminal guidance |
| `military.mule_robot` | Autonomous Load Carrier (Mule) | Soldier squad logistics support | Follow-me mode, terrain following, quiet operation |

### 1.18 Healthcare — Non-Surgical, Non-Wearable (6 platforms)

Facility-based healthcare robots that are neither surgical systems nor
body-worn devices. Distinct compute/safety/regulatory profiles.

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `healthcare.uv_disinfection` | UV-C Disinfection Robot | Autonomous room sterilization with UV-C | Human-exclusion zone enforcement, dose mapping, room coverage path |
| `healthcare.pharmacy_dispensing` | Pharmacy Dispensing Robot | Hospital automated medication dispensing | Barcode verification, controlled substance tracking, error-proof |
| `healthcare.patient_transport` | Patient Transport Robot | Autonomous bed/gurney/wheelchair mover | Gentle motion, elevator integration, IV-pole accommodation |
| `healthcare.rehabilitation` | Rehabilitation Robot | Stationary hand/ankle/shoulder therapy | Adaptive impedance, progress tracking, therapist override |
| `healthcare.telepresence_medical` | Medical Telepresence Robot | Remote physician rounding/consultation | HD video, stethoscope/vitals integration, HIPAA-compliant |
| `healthcare.radiology_positioning` | Radiology Positioning Robot | Patient positioning for CT/MRI/X-ray | Sub-mm repeatability, radiation-zone tolerant, DICOM integration |

### 1.19 Prosthetics & Assistive Devices (4 platforms)

Human-worn or human-integrated devices with AI-driven intent prediction.
Fundamentally different from exoskeletons — these replace or augment
missing/impaired function rather than enhancing existing capability.

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `assistive.prosthetic_hand` | AI Prosthetic Hand | Myoelectric hand with intent prediction | EMG pattern recognition, grasp type selection, tactile feedback |
| `assistive.prosthetic_leg` | AI Prosthetic Leg | Powered leg with terrain adaptation | Gait phase detection, stair/ramp adaptation, stumble recovery |
| `assistive.smart_wheelchair` | Smart Wheelchair | Semi-autonomous navigation wheelchair | Obstacle avoidance, doorway navigation, shared autonomy with user |
| `assistive.communication_device` | Communication Assistive Device | Eye-tracking / BCI input for ALS/locked-in | Eye gaze tracking, brain-computer interface, predictive text |

### 1.20 Waste & Recycling (4 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `waste.recycling_sorter` | Recycling Sorting Robot | Pick recyclables from mixed waste stream | Material classification (plastic type, metal, paper), high-speed pick |
| `waste.collection_robot` | Waste Collection Robot | Autonomous bin pickup/emptying | Heavy lift, outdoor nav, bin detection, fleet-operated |
| `waste.hazardous_handling` | Hazardous Waste Robot | Chemical/biological/radioactive waste | Full containment, remote operation, decontaminable |
| `waste.ewaste_disassembly` | E-Waste Disassembly Robot | Component recovery from electronics | PCB component ID, selective disassembly, precious metal recovery |

### 1.21 Food Processing & Commercial Kitchen (5 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `food.grading_sorting` | Food Grading / Sorting Robot | Fruit/vegetable/seafood quality grading | Hyperspectral/color vision, defect detection, weight sorting |
| `food.meat_processing` | Meat Processing Robot | Cutting, deboning, portioning (large carcass) | Force-controlled cutting, 3D carcass scanning, food-safe |
| `food.commercial_kitchen` | Commercial Kitchen Robot | High-volume food prep, cooking, plating | Heat-tolerant, washdown-rated, recipe execution engine |
| `food.bakery_automation` | Bakery Automation Robot | Dough handling, shaping, decorating | Deformable material handling, proofing-aware timing |
| `food.beverage_dispensing` | Beverage / Barista Robot | Automated coffee/cocktail/smoothie | Liquid handling, customer interaction, recipe management |

### 1.22 Underground / Pipe / Confined Space (4 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `underground.sewer_inspection` | Sewer Inspection Robot | Municipal sewer pipe camera robot | In-pipe crawler, defect classification, NASSCO-coded reporting |
| `underground.water_main_inspection` | Water Main Inspection Robot | Pressurized pipe inspection | Live-main insertion, leak detection, wall thickness measurement |
| `underground.mine_robot` | Underground Mine Robot | Confined mine tunnel operation | Methane-rated, no-GPS SLAM, rock-fall detection, ATEX |
| `underground.tunnel_inspection` | Tunnel Inspection Robot | Road/rail tunnel lining inspection | Crack/spalling detection, laser profiling, long-range |

### 1.23 Ship / Hull / Offshore (3 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `marine.hull_cleaning` | Ship Hull Cleaning Robot | Underwater hull scrubbing, biofouling removal | Magnetic adhesion, cavitation cleaning, paint-safe |
| `marine.ship_hold_inspection` | Ship Hold Inspection Robot | Cargo hold / ballast tank inspection | Confined space, corrosion detection, coating assessment |
| `marine.offshore_inspection` | Offshore Platform Inspection Robot | Oil rig / wind farm structural inspection | Climbing/rope, salt spray IP68, cathodic protection check |

### 1.24 Forestry (3 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `forestry.tree_planting` | Tree Planting Robot | Automated seedling planting at scale | Drill/dibble mechanism, GPS-logged, terrain traversal |
| `forestry.inventory_drone` | Forest Inventory Drone | Tree counting, species ID, canopy health | LiDAR + multispectral, individual tree segmentation, biomass est. |
| `forestry.timber_harvesting` | Autonomous Timber Harvester | Felling, delimbing, bucking | Heavy hydraulic, tree detection, cut optimization |

### 1.25 Snow / Ice / Weather Operations (3 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `weather.autonomous_snowplow` | Autonomous Snow Plow | Road/runway snow clearing | Blade control, GPS-guided edge detection, salt/sand spreading |
| `weather.residential_snow` | Residential Snow Removal Robot | Driveway/sidewalk snow clearing | Consumer-scale, battery, boundary detection |
| `weather.deicing_robot` | De-Icing Robot | Aircraft / wind turbine de-icing | Fluid spray/heat, surface temp sensing, ice thickness detection |

### 1.26 Data Center & Telecom (2 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `datacenter.inspection_robot` | Data Center Inspection Robot | Thermal monitoring, cable tracing, asset audit | Hot/cold aisle nav, rack-level thermal map, asset tag reading |
| `telecom.tower_inspector` | Cell Tower Inspection Robot | Tower climbing, antenna inspection | Climbing mechanism, RF survey, structural assessment |

### 1.27 Mining — Extended (2 platforms)

Beyond the existing `ground_wheeled.mining_haul` (open-pit surface hauler).

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `mining.underground_loader` | Underground Mine Loader/Hauler | Confined tunnel load-haul-dump | Methane-rated, low-profile, no-GPS, collision avoidance |
| `mining.survey_robot` | Mine Survey / Mapping Robot | 3D cavity scanning, geological assessment | LiDAR SLAM, void detection, gas monitoring |

### 1.28 Oil & Gas (2 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `oil_gas.refinery_inspection` | Refinery Inspection Robot | ATEX-rated confined-space inspection | Explosion-proof, gas detection, corrosion measurement |
| `oil_gas.downhole_inspection` | Downhole / Wellbore Robot | In-well inspection and intervention | Extreme pressure/temp, small diameter, wireline deployment |

### 1.29 Aerospace Manufacturing (2 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `aerospace.riveting_robot` | Aircraft Riveting / Drilling Robot | Fuselage drilling and fastening | Large-envelope gantry, hole quality inspection, countersink depth |
| `aerospace.aircraft_painting` | Aircraft Painting Robot | Full-airframe coating application | ATEX paint booth, 30m+ reach, thickness measurement |

### 1.30 Textile / Garment (3 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `textile.sewing_robot` | Automated Sewing Robot | Garment assembly, seam stitching | Deformable fabric handling, needle guidance, pattern following |
| `textile.fabric_cutting` | Fabric Cutting Robot | Pattern cutting from rolls | Multi-ply cutting, nesting optimization, laser/blade |
| `textile.inspection` | Textile Inspection Robot | Fabric defect detection in weaving/dyeing | High-speed line scan, color consistency, weave defect AI |

### 1.31 Consumer / Home — Extended (2 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `consumer.window_cleaner` | Window Cleaning Robot | Exterior glass cleaning (high-rise or residential) | Vacuum/magnetic adhesion, edge detection, safety tether |
| `consumer.gutter_cleaner` | Gutter Cleaning Robot | Roof-edge debris removal | Roof-edge navigation, debris extraction, remote monitoring |

### 1.32 Municipal Services (2 platforms)

| ID | Platform | Description | Key Differentiator |
|----|----------|-------------|-------------------|
| `municipal.street_sweeper` | Autonomous Street Sweeper | Road/sidewalk sweeping | Large area, debris collection, curb following, night operation |
| `municipal.pothole_inspector` | Road Surface Inspector / Repair | Pothole detection, road condition mapping | 3D surface profiling, severity classification, repair dispatching |

---

## 2. Attribute Analysis

### 2.1 New Attributes Introduced by Expanded Platforms

The expanded taxonomy introduces attributes not covered by the current 8-subsystem
`SystemSpec` model:

#### Load / Payload Characterization (new subsystem)

The single most important missing attribute dimension. A laundry-folding robot
and a forge-tending robot both "manipulate objects" but share almost nothing
in their mechanical, sensing, or control requirements.

**Load Class Taxonomy:**

| Class | Mass Range | Examples | Grip Type | Typical Precision |
|-------|-----------|----------|-----------|------------------|
| `micro` | <10g | SMT components, lab pipette tips | vacuum, tweezers | 0.01-0.1mm |
| `micro_fluid` | µL-mL volumes | Pipetting, reagent dispensing | positive displacement | nL-µL accuracy |
| `fragile_micro` | <100g | Tissue sections, eggs, glass slides | soft contact, vacuum | 0.1-1mm |
| `light` | 0.1-10kg | Packaged goods, tools, small parts | pinch, vacuum, magnetic | 0.1-1mm |
| `deformable_light` | 0.1-5kg | Laundry, food, fabric, cable | soft gripper, multi-finger | 1-5mm |
| `medium` | 10-50kg | Boxes, subassemblies, engine parts | power grip, fork, clamp | 0.5-2mm |
| `heavy` | 50-500kg | Pallets, car bodies, steel beams | clamp, magnetic, vacuum pad | 1-5mm |
| `ultra_heavy` | 500-50,000kg | Forge billets, containers, airframes | hydraulic, crane, spreader | 5-50mm |
| `live_animal` | 0.1-1000kg | Livestock, fish, poultry | restraint, gentle contact | 10-50mm |
| `contact_tool` | N/A (tool mass) | Weld gun, spray gun, drill, grinder | rigidly mounted tool | depends on process |
| `sterile_fluid` | µL-L volumes | Cell culture, biobank samples | aseptic pipette, peristaltic | µL-mL accuracy |
| `hazardous` | varies | Explosives, radioactive material, chemicals | remote gripper, shielded | varies |

**Load Characterization Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `load_class` | enum | From taxonomy above |
| `load_mass_min_kg` | float | Minimum object mass |
| `load_mass_max_kg` | float | Maximum object mass |
| `load_rigidity` | enum | rigid, semi_rigid, deformable, fluid, granular |
| `load_fragility` | enum | robust, normal, fragile, ultra_fragile |
| `load_temperature_c` | range | Object temperature (-196 for cryo to +1200 for forge) |
| `load_sterility_required` | bool | Sterile handling required |
| `load_food_safe` | bool | Food-contact materials required |
| `load_hazardous` | bool | Hazardous material (chemical, radioactive, explosive) |
| `grip_type` | list[enum] | vacuum, pinch, power, magnetic, soft, hydraulic, adhesive, tweezers, fork, clamp, spreader |
| `grip_force_n` | range | Required grip force |
| `grasp_planning_required` | bool | AI-based grasp planning (vs fixed grasp) |
| `deformable_object_handling` | bool | Must handle objects that change shape |
| `object_variety` | enum | single_sku, few_sku, mixed_sku, unknown_objects |
| `throughput_objects_per_hour` | float | Required manipulation throughput |
| `placement_precision_mm` | float | Required placement accuracy |

This subsystem is what differentiates:
- A laundry robot (deformable_light, soft gripper, 5mm precision, unknown_objects)
- A forge tender (ultra_heavy, hydraulic, 20mm precision, single_sku, 1200C)
- A pipetting robot (micro_fluid, positive_displacement, µL precision, sterile)
- A palletizer (heavy, vacuum_pad, 5mm precision, few_sku, 500 pph)

#### Surgical / Clinical Procedure (new subsystem)

Attributes specific to surgical and medical procedure robots, beyond the
general medical device attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `procedure_domain` | enum | laparoscopic, orthopedic, spinal, neuro, ophthalmic, cardiac, ent, urological, dental, microsurgery, endovascular, radiation |
| `workspace_geometry` | enum | open_cavity, trocar_access, percutaneous, intravascular, intraoral, transcranial |
| `instrument_dof` | int | Degrees of freedom per instrument |
| `instrument_diameter_mm` | float | Instrument shaft diameter (5-12mm laparoscopic, <1mm ophthalmic) |
| `position_accuracy_mm` | float | Required tip position accuracy |
| `force_range_n` | range | Force range (0.001N ophthalmic to 500N orthopedic) |
| `imaging_modality` | list[enum] | endoscope, fluoroscopy, ct, mri, ultrasound, microscope, oct |
| `motion_compensation` | enum | none, respiratory_gating, cardiac_gating, tremor_cancellation |
| `sterile_barrier` | enum | draped, sealed, autoclavable |
| `registration_method` | enum | fiducial, surface_matching, image_based, mechanical |
| `regulatory_class` | enum | fda_class_II, fda_class_III, ce_class_IIb, ce_class_III |
| `patient_awake` | bool | Procedure performed on conscious patient |

#### Laboratory Automation (new subsystem)

| Attribute | Type | Description |
|-----------|------|-------------|
| `lab_type` | enum | wet_lab, dry_lab, cleanroom, bsl2, bsl3, bsl4, cryo_facility |
| `plate_format` | list[enum] | 6_well, 12_well, 24_well, 96_well, 384_well, 1536_well, tube_rack, vial_tray |
| `dispensing_volume_min` | float | Minimum dispense volume (nL for acoustic, µL for tip) |
| `dispensing_volume_max` | float | Maximum dispense volume |
| `dispensing_accuracy_cv_pct` | float | Coefficient of variation for dispensing |
| `contamination_control` | enum | none, filter_tips, uv_decon, laminar_flow, positive_displacement |
| `temperature_range_c` | range | Operating temperature (-196 for LN2 to +95 for PCR) |
| `protocol_engine` | bool | Runs multi-step automated protocols |
| `lims_integration` | bool | Integrates with Laboratory Information Management System |
| `sample_tracking` | enum | barcode_1d, barcode_2d, rfid, none |
| `throughput_samples_per_day` | int | Daily sample processing capacity |
| `gmp_compliant` | bool | Good Manufacturing Practice compliant |

#### Veterinary / Husbandry (new subsystem)

| Attribute | Type | Description |
|-----------|------|-------------|
| `animal_type` | list[enum] | cattle, sheep, poultry, swine, fish, equine, canine, feline |
| `animal_restraint` | enum | free_standing, chute, cradle, tank, pen, none |
| `animal_welfare_compliance` | list[str] | Applicable welfare standards |
| `live_animal_contact` | bool | Direct contact with live animals |
| `food_chain_traceability` | bool | Part of food production chain |
| `washdown_rated` | bool | Can withstand high-pressure washdown |
| `bio_containment_level` | enum | none, basic_hygiene, biosecurity_level_1, biosecurity_level_2 |

#### Surveillance / Monitoring (new subsystem)

Surveillance is a cross-cutting capability — it applies to stationary cameras,
mobile patrol robots, aerial drones, and wearable devices. The common attribute
set describes what the system watches for, how it alerts, and how it fuses
multiple sensing modalities.

| Attribute | Type | Description |
|-----------|------|-------------|
| `surveillance_purpose` | list[enum] | security, access_control, safety, traffic, environmental, structural, counting, compliance, health, process |
| `detection_targets` | list[str] | What to detect: person, vehicle, fire, smoke, intrusion, anomaly, PPE, animal, crack, flood, etc. |
| `counting_targets` | list[str] | What to count: people, vehicles, animals, objects (empty if not a counting platform) |
| `spectral_bands` | list[enum] | visible, nir, swir, mwir, lwir_thermal, radar, acoustic, multispectral, hyperspectral |
| `sensor_fusion_type` | enum | single_sensor, early_fusion, late_fusion, decision_fusion |
| `coverage_type` | enum | point (single camera), zone (multi-camera), area (wide-area), perimeter (linear), volumetric (3D) |
| `coverage_range_m` | float | Maximum detection range |
| `persistence` | enum | continuous_24_7, scheduled, triggered, on_demand |
| `alert_latency_ms` | float | Time from event to alert |
| `alert_channels` | list[enum] | visual_overlay, siren, push_notification, sms, email, vms, scada, api_webhook |
| `privacy_mode` | enum | full_video, anonymized, skeleton_only, metadata_only, radar_only |
| `retention_days` | int | How long data/video is stored |
| `integration_protocol` | list[str] | onvif, rtsp, mqtt, modbus, opc_ua, api_rest, syslog |
| `edge_vs_cloud` | enum | edge_only, edge_primary, hybrid, cloud_primary |
| `multi_camera_analytics` | bool | Cross-camera tracking, re-identification |
| `weather_tolerance` | enum | indoor_only, light_weather, all_weather, extreme |

#### Swarm / Fleet Coordination (new subsystem)

Platforms: all `*.swarm_*`, `logistics.goods_to_person`, `logistics.fleet_yard_truck`,
`ground_wheeled.swarm_warehouse`, `military.swarm_*`

| Attribute | Type | Description |
|-----------|------|-------------|
| `fleet_size` | int range | Expected fleet size (2 — 10,000) |
| `coordination_topology` | enum | centralized, decentralized, hierarchical, hybrid |
| `inter_agent_comms` | enum | wifi_mesh, uwb, acoustic, radio, optical |
| `consensus_algorithm` | str | raft, paxos, gossip, none |
| `formation_control` | bool | Does fleet maintain spatial formations? |
| `task_allocation` | enum | central_planner, auction, market, emergent |
| `resilience_to_attrition` | bool | Can swarm lose members and continue? |
| `shared_perception` | bool | Distributed/fused perception across agents? |
| `collision_avoidance_inter_agent` | bool | Inter-agent collision avoidance |
| `swarm_latency_budget_ms` | float | Max inter-agent coordination latency |

#### Medical / Biocompatibility (new subsystem)

Platforms: `wearable.cgm_ai`, `wearable.smart_insulin_pump`, `wearable.cardiac_monitor`,
`wearable.seizure_detector`, `manipulation.surgical_robot`

| Attribute | Type | Description |
|-----------|------|-------------|
| `medical_device_class` | enum | FDA Class I, II, III / EU MDR Class I, IIa, IIb, III |
| `biocompatible` | bool | Direct skin/body contact with biocompatible materials |
| `drug_delivery` | bool | Administers medication |
| `drug_delivery_type` | enum | insulin, IV, topical, inhaled |
| `continuous_monitoring_hours` | float | Required continuous operation time |
| `clinical_accuracy_standard` | str | e.g., "ISO 15197" for glucose, "IEC 60601" for ECG |
| `alert_types` | list[str] | emergency_call, vibration, audible, caregiver_notify |
| `hipaa_compliant` | bool | Handles protected health information |
| `fda_clearance_pathway` | enum | 510k, de_novo, pma, exempt |

#### Heavy Machinery / Large Vehicle (extends actuator subsystem)

Platforms: `construction.*`, `agriculture.autonomous_tractor`, `agriculture.autonomous_harvester`,
`mining_haul`, `logistics.container_handler`

| Attribute | Type | Description |
|-----------|------|-------------|
| `vehicle_gross_weight_kg` | float | Total vehicle weight |
| `implement_control` | bool | Controls attached implements (bucket, plow, header) |
| `hydraulic_system` | bool | Hydraulic actuation (vs electric) |
| `engine_type` | enum | electric, diesel, hybrid |
| `ground_pressure_kpa` | float | For tracked/heavy vehicles |
| `operating_grade_percent` | float | Max slope operation |
| `towing_capacity_kg` | float | Trailer/implement towing |

#### Chemical / Hazardous Material Handling

Platforms: `agriculture.pesticide_ground_sprayer`, `agriculture.biosurveillance_drone`,
`ground_tracked.firefighting_robot`, `ground_tracked.nuclear_decommission`

| Attribute | Type | Description |
|-----------|------|-------------|
| `chemical_handling` | bool | Handles chemicals/hazardous materials |
| `chemical_type` | list[str] | pesticide, herbicide, fertilizer, fire_retardant |
| `atex_rated` | bool | Explosion-proof rating for hazardous atmospheres |
| `radiation_hardened` | bool | Tolerant to ionizing radiation |
| `decontaminable` | bool | Can be decontaminated after exposure |

#### Depth / Pressure Rating (marine/underwater)

| Attribute | Type | Description |
|-----------|------|-------------|
| `max_depth_m` | float | Maximum operating depth |
| `pressure_housing` | bool | Pressure-rated enclosure |
| `buoyancy_system` | enum | passive, active, variable |
| `acoustic_comms` | bool | Underwater acoustic communication |

### 2.2 Revised Subsystem Model

The analysis reveals the current 8-subsystem model needs extension to **16 subsystems**:

```
Current (8):          New (16):
  perception            perception
  compute               compute
  power                 power
  sensors               sensors
  actuators             actuators
  comms                 comms
  autonomy              autonomy
  safety                safety
                      + load            (object/payload characterization — THE key differentiator)
                      + surveillance    (detection targets, spectral bands, alerting, coverage, fusion)
                      + fleet           (swarm/multi-agent coordination)
                      + medical         (biocompatibility, drug delivery, clinical)
                      + surgical        (procedure domain, instruments, imaging, registration)
                      + laboratory      (plate formats, dispensing, LIMS, contamination control)
                      + veterinary      (animal type, restraint, welfare, biosecurity)
                      + environment     (chemical, radiation, pressure, ATEX)
```

The `load` subsystem is the most impactful addition — it is what distinguishes
robots that otherwise look identical at the actuator/DOF level but operate in
completely different domains. A 6-DOF arm is a 6-DOF arm, but whether it's
folding socks or forging crankshafts is entirely determined by what it manipulates.

The `actuators` subsystem also needs extension for heavy machinery attributes
(hydraulics, implements, vehicle weight).

### 2.3 Universal vs Domain-Specific Attributes

**Universal (all 285 platforms have these):**

| Attribute | Range |
|-----------|-------|
| Power budget | 0.01W (wearable sensor) — 50kW (autonomous excavator) |
| Compute budget | 0.1 TOPS — 2000 TOPS |
| Perception latency | 0.5ms (surgical) — 5000ms (ocean glider) |
| Weight | 10g (wearable) — 500,000 kg (mining hauler) |
| Cost | $5 (sensor) — $5M (surgical robot) |
| Operating temperature | -40C — +85C |
| Safety level | none — SIL-4 |
| Autonomy level | teleoperated — full |
| Communication | always (some protocol) |

**Common but not universal:**

| Attribute | Present in | % of platforms |
|-----------|-----------|---------------|
| Load characterization | All manipulation + logistics + lab + veterinary | ~45% |
| Locomotion speed | All mobile platforms | ~55% |
| Fleet coordination | Swarm variants, warehouse, logistics, agriculture | ~20% |
| Manipulation DOF | All manipulation + surgical + lab + veterinary | ~40% |
| Surgical procedure attrs | Surgical specializations only | ~6% |
| Laboratory automation attrs | Lab automation only | ~6% |
| Veterinary/husbandry attrs | Veterinary platforms only | ~4% |
| Medical device certification | Medical wearables + surgical | ~12% |
| Chemical handling | Agriculture, firefighting, nuclear, lab | ~10% |
| Underwater pressure | Marine underwater only | ~3% |

---

## 3. Classification Axes

### 3.1 Primary Axes (4 orthogonal)

Every platform can be located as a point in this 4D space:

#### Axis 1: Locomotion

```
aerial
├── rotary_wing (multirotor, helicopter)
├── fixed_wing (airplane, glider)
├── vtol (tiltrotor, tail-sitter)
├── lighter_than_air (blimp, balloon)
└── tethered

ground
├── wheeled (differential, ackermann, omnidirectional, skid-steer)
├── legged (biped, quadruped, hexapod)
├── tracked
└── hybrid (wheel-leg, wheel-track)

marine
├── surface (displacement, planing, hydrofoil)
├── underwater (propeller, buoyancy, bio-inspired)
└── amphibious

space
├── orbital (thrusters, reaction wheels)
├── surface (wheeled, legged)
└── atmospheric (parafoil, rotorcraft)

stationary
├── fixed_mount (camera, sensor box)
├── pan_tilt (PTZ camera)
├── gantry (moves on fixed rails)
└── conveyor_mounted

wearable
├── head_mounted (glasses, headset)
├── body_worn (vest, belt, watch)
├── limb_attached (exoskeleton, glove)
└── implantable (future: neural interface)
```

#### Axis 2: Manipulation Capability

Two dimensions: **mechanism** and **load class**.

Mechanism:
```
none                    — Sensors only (camera, drone, wearable)
fixed_tool              — Non-dexterous tool (spray nozzle, welding torch)
fluid_dispenser         — Pipette, pump, nozzle (lab, agriculture)
1_dof_gripper           — Simple open/close gripper
multi_dof_hand          — Dexterous end-effector (3+ DOF)
soft_gripper            — Compliant gripper for deformables/fragiles
articulated_arm         — 4-7 DOF arm with end-effector
dual_arm                — Bimanual coordination
mobile_manipulation     — Arm on mobile base
surgical_instrument     — Wristed laparoscopic/micro instrument
full_humanoid           — Anthropomorphic manipulation
heavy_gantry            — Overhead crane/gantry for ultra-heavy loads
```

Load class (orthogonal to mechanism):
```
none                    — No payload (sensing only)
micro                   — <10g (SMT, pipette tips)
micro_fluid             — µL-mL (pipetting, dispensing)
fragile_micro           — <100g fragile (tissue, glass, eggs)
light                   — 0.1-10kg (packages, tools)
deformable_light        — 0.1-5kg deformable (laundry, food, cable)
medium                  — 10-50kg (boxes, subassemblies)
heavy                   — 50-500kg (pallets, car bodies)
ultra_heavy             — 500-50,000kg (containers, forgings)
live_animal             — 0.1-1000kg live (livestock, fish)
sterile_fluid           — Aseptic liquid handling (cell culture, biobank)
hazardous               — Explosive, radioactive, toxic
```

#### Axis 3: Operating Environment

```
indoor_structured       — Warehouse, factory, hospital corridor (known map, controlled)
indoor_unstructured     — Home, office (unknown layout, dynamic occupants)
indoor_cleanroom        — ISO class 1-8 cleanroom (semiconductor, pharma)
indoor_sterile          — Operating room, BSL-2/3/4 lab (aseptic, biocontainment)
indoor_lab              — Wet lab, dry lab (benchtop, chemical fume hoods)
indoor_cold_chain       — Freezer (-25C), cryogenic (-196C LN2 biobank)
outdoor_urban           — Streets, sidewalks, buildings
outdoor_rural           — Fields, forests, open terrain
outdoor_extreme         — Desert, arctic, high altitude
outdoor_farmyard        — Livestock pens, barns, paddocks (animals, mud, wash-down)
underwater_shallow      — <100m depth
underwater_deep         — >100m depth
space_orbital           — Microgravity, vacuum
space_planetary         — Planetary surface
subterranean            — Mines, caves, tunnels
hazardous_chemical      — Chemical plant, ATEX zones
hazardous_radiation     — Nuclear facility, radioactive contamination
hazardous_fire          — Active fire, high-temperature zones
food_processing         — Food-contact, HACCP, washdown-rated
```

#### Axis 4: Human Interaction Proximity

```
no_humans               — Remote, isolated operation
humans_distant          — Humans in vicinity but not in workspace (>5m)
humans_nearby           — Humans in shared space but not direct interaction (1-5m)
humans_contact          — Direct physical interaction (cobot, exoskeleton, medical)
humans_on_body          — Worn on or implanted in human body
humans_inside           — Human is passenger (vehicle, air taxi)
```

### 3.2 Secondary Axes (continuous, define the design space)

| Axis | Range | Unit |
|------|-------|------|
| Power budget | 0.01 — 50,000 | W |
| Perception latency | 0.5 — 5,000 | ms |
| Weight budget | 0.01 — 500,000 | kg |
| Unit cost | 5 — 5,000,000 | USD |
| Autonomy level | 0 — 5 | SAE-like scale |
| Fleet size | 1 — 10,000 | units |
| Mission duration | 0.1 — 8,760 | hours |
| Safety integrity | 0 — 4 | SIL level |

---

## 4. Keyword Density Strategy

For prompt matching to work with low-information queries, each platform needs
**20-50 keywords/phrases** covering:

1. **Platform identity**: official name, abbreviations, brand names
2. **Common descriptions**: how a non-expert would describe it
3. **Application phrases**: what it does in plain English
4. **Industry terms**: domain-specific jargon
5. **Component mentions**: key hardware that implies the platform
6. **Negation anchors**: what it is NOT (to disambiguate similar platforms)

### Example: `edge_vision.industrial_quality`

```yaml
keywords:
  identity:
    - industrial quality inspection
    - visual inspection system
    - automated optical inspection
    - AOI system
    - machine vision inspection
  descriptions:
    - camera checking parts on production line
    - defect detection on factory line
    - quality control camera
    - inline inspection
  application:
    - surface defect detection
    - dimensional measurement
    - label verification
    - color inspection
    - scratch detection
    - print quality check
  industry:
    - AOI
    - SPC vision
    - GigE vision
    - CoaXPress
    - line scan
    - area scan
  components:
    - line scan camera
    - strobe light
    - telecentric lens
    - frame grabber
  negation:
    - NOT security camera
    - NOT surveillance
```

This density means "edge device running edge detection" would match against
"edge device" appearing in the description keywords of `edge_vision.edge_vision_box`
and potentially `edge_vision.industrial_quality`.

---

## 5. Total Platform Count

| # | Category | Count |
|---|----------|-------|
| 1.1 | Aerial | 16 |
| 1.2 | Ground — Wheeled | 18 |
| 1.3 | Ground — Legged | 8 |
| 1.4 | Ground — Tracked | 6 |
| 1.5 | Manipulation — Industrial | 22 |
| 1.5b | Manipulation — Surgical | 12 |
| 1.5c | Manipulation — Veterinary & Husbandry | 8 |
| 1.5d | Manipulation — Laboratory Automation | 12 |
| 1.5 | Manipulation — Domestic / Light | 3 |
| 1.6 | Marine — Surface | 6 |
| 1.7 | Marine — Underwater | 6 |
| 1.8 | Space | 5 |
| 1.9 | Surveillance — Security & Access | 5 |
| 1.9 | Surveillance — Public Safety | 5 |
| 1.9 | Surveillance — Traffic & Transportation | 4 |
| 1.9 | Surveillance — Environmental & Ecological | 5 |
| 1.9 | Surveillance — Counting & Census | 3 |
| 1.9 | Surveillance — Multi-Spectral & Sensor Fusion | 2 |
| 1.9 | Surveillance — Industrial & Workplace | 4 |
| 1.9 | Surveillance — Structural & Infrastructure | 2 |
| 1.9 | Surveillance — Retail & Commercial | 3 |
| 1.9 | Surveillance — Residential / Consumer | 3 |
| 1.9 | Surveillance — Specialized Sensing | 5 |
| 1.10 | Wearable / Portable | 12 |
| 1.11 | Consumer / Home | 8 |
| 1.12 | Agriculture | 14 |
| 1.13 | Construction | 7 |
| 1.14 | Logistics & Warehousing | 14 |
| 1.15 | Energy & Utilities | 6 |
| 1.16 | Entertainment & Hospitality | 4 |
| 1.17 | Military / Swarm | 6 |
| 1.18 | Healthcare — Non-Surgical | 6 |
| 1.19 | Prosthetics & Assistive | 4 |
| 1.20 | Waste & Recycling | 4 |
| 1.21 | Food Processing & Commercial Kitchen | 5 |
| 1.22 | Underground / Pipe / Confined Space | 4 |
| 1.23 | Ship / Hull / Offshore | 3 |
| 1.24 | Forestry | 3 |
| 1.25 | Snow / Ice / Weather | 3 |
| 1.26 | Data Center & Telecom | 2 |
| 1.27 | Mining — Extended | 2 |
| 1.28 | Oil & Gas | 2 |
| 1.29 | Aerospace Manufacturing | 2 |
| 1.30 | Textile / Garment | 3 |
| 1.32 | Municipal Services | 2 |
| | **Total** | **285** |

---

## 6. New Attributes Summary

The expanded taxonomy introduces **8 new subsystems** and **~90 new attributes**
beyond the current `SystemSpec` model:

| New Subsystem | Attribute Count | Triggered By |
|---------------|----------------|--------------|
| **Load / Payload** | 16 | ~45% of platforms (all manipulation, logistics, lab, vet) |
| **Surveillance / Monitoring** | 16 | ~25% of platforms (all 41 surveillance + mobile patrol platforms) |
| **Surgical / Clinical Procedure** | 12 | ~5% of platforms (12 surgical specializations) |
| **Laboratory Automation** | 12 | ~5% of platforms (12 lab automation types) |
| **Veterinary / Husbandry** | 7 | ~3% of platforms (8 vet/husbandry types) |
| Fleet / Swarm | 10 | ~15% of platforms |
| Medical Device | 9 | ~10% of platforms |
| Hazardous Environment | 5 | ~8% of platforms |
| Heavy Machinery (actuator ext.) | 7 | ~8% of platforms |

The **Load/Payload subsystem is the highest-impact addition**. It is the single
attribute dimension that most strongly differentiates platforms that are otherwise
identical at the actuator level. Without it, a pipetting robot and a forge-tending
robot are both "6-DOF arms" — with it, they are in completely different design spaces.

Plus extensions to existing subsystems:
- `sensors`: add `acoustic_comms`, `multispectral`, `hyperspectral`, `oct`, `endoscope`
- `comms`: add `uwb`, `acoustic`, `satellite_link`, `ethercat`, `opc_ua`
- `safety`: add `atex_zone`, `radiation_tolerance_gray`
- `actuators`: add `hydraulic_system`, `implement_control`, `engine_type`

---

## 7. Next Steps

1. **Design YAML schema** — Define the platform registry schema incorporating
   the 16-subsystem model, keyword density requirements, and attribute ranges.
   The schema itself lives at `data/platforms/schema.yaml`.

2. **Populate platform registry** — Create YAML files for all 285 platforms with
   rich keyword sets (20-50 per platform) and attribute range specifications.
   Organized under `data/platforms/` by category.

3. **Populate configuration registry** — Create concrete instances mapping to real
   products (DJI Matrice 350, Boston Dynamics Spot, Franka Emika Panda, NVIDIA
   Isaac AMR, da Vinci Xi, Intuitive Surgical Ion, Kiva Systems, etc.) with
   specific attribute values. Target: 200+ configurations.
   Lives under `data/platforms/configurations/`.

4. **Build matching engine** — Replace `detect_domain()` keyword counter with
   TF-IDF or embedding-based scorer against the full registry corpus. Should
   return ranked candidates with confidence scores, not just the best match.

5. **Integrate with qualification flow** — Wire the platform registry into
   `GoalQualifier` so that matched platforms pre-fill the spec with realistic
   attribute ranges and relevant questions. The qualification question trees
   become registry-driven rather than hard-coded per domain template.

6. **CLI commands** — Add `branes platform list`, `branes platform show <id>`,
   `branes platform search <query>` for inspecting and managing the registry.
