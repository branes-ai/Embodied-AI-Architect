"""Drone domain template — single compute node, battery-constrained.

Complexity level: MODERATE
- Single SoC doing perception + control
- Tight power/weight/volume constraints from battery + airframe
- Perception drives flight controller directly
- Environmental: outdoor, vibration, thermal cycling

The drone is the "canonical" embodied AI platform: one compute node
with a perception pipeline feeding a control loop, all under strict
SWaP-C constraints from the laws of physics (flight).
"""

from embodied_ai_architect.qualification.models import (
    DomainTemplate,
    Question,
    QuestionType,
)

TEMPLATE = DomainTemplate(
    domain="drone",
    display_name="Unmanned Aerial Vehicle (Drone)",
    description=(
        "Aerial platforms from micro-drones to large fixed-wing UAVs. "
        "Battery-constrained, single compute node, perception drives "
        "flight controller. SWaP-C is dominated by flight physics."
    ),
    keywords=[
        "drone",
        "uav",
        "quadrotor",
        "multirotor",
        "fixed-wing",
        "aerial",
        "flying",
        "flight",
        "quadcopter",
        "hexacopter",
        "octocopter",
        "vtol",
    ],
    base_implications={
        "platform_type": "drone",
        "sensors.modalities": ["imu", "barometer", "gps"],
        "comms.protocols": ["wifi", "mavlink"],
    },
    questions=[
        # --- Platform ---
        Question(
            id="drone_type",
            dimension="platform",
            text="What type of drone is this?",
            explanation=(
                "Drone subtype determines the power budget, payload capacity, "
                "speed envelope, and mission duration — which constrain every "
                "downstream design choice."
            ),
            question_type=QuestionType.SINGLE_CHOICE,
            options=[
                "multirotor_delivery",
                "multirotor_inspection",
                "multirotor_racing",
                "fixed_wing_survey",
                "fixed_wing_cargo",
                "agricultural_sprayer",
                "micro_indoor",
            ],
            option_descriptions={
                "multirotor_delivery": "Last-mile package delivery (2-5kg payload, 15-25min endurance)",
                "multirotor_inspection": "Infrastructure inspection (cameras/LiDAR, 20-30min hover)",
                "multirotor_racing": "FPV racing or high-speed maneuvers (<200ms total latency)",
                "fixed_wing_survey": "Mapping/survey (long endurance, large area coverage)",
                "fixed_wing_cargo": "Heavy cargo transport (>5kg payload, >60min endurance)",
                "agricultural_sprayer": "Precision spraying (GPS-guided, multi-acre coverage)",
                "micro_indoor": "Sub-250g indoor drone (warehouse, inspection)",
            },
            required=True,
            implications={
                "multirotor_delivery": {
                    "power.compute_power_watts": 5.0,
                    "power.mission_duration_min": 25,
                    "actuators.max_speed_mps": 15,
                    "actuators.payload_kg": 2.5,
                    "power.battery_wh": 100,
                    "autonomy.level": "conditional",
                    "sensors.environmental_rating": "ip54",
                },
                "multirotor_inspection": {
                    "power.compute_power_watts": 8.0,
                    "power.mission_duration_min": 25,
                    "actuators.max_speed_mps": 8,
                    "actuators.payload_kg": 1.0,
                    "power.battery_wh": 80,
                    "autonomy.level": "partial",
                    "sensors.environmental_rating": "outdoor",
                },
                "multirotor_racing": {
                    "power.compute_power_watts": 3.0,
                    "power.mission_duration_min": 5,
                    "actuators.max_speed_mps": 40,
                    "actuators.payload_kg": 0.0,
                    "power.battery_wh": 30,
                    "perception.max_latency_ms": 10.0,
                    "autonomy.level": "assisted",
                },
                "fixed_wing_survey": {
                    "power.compute_power_watts": 3.0,
                    "power.mission_duration_min": 60,
                    "actuators.max_speed_mps": 25,
                    "actuators.payload_kg": 0.5,
                    "power.battery_wh": 200,
                    "autonomy.level": "high",
                    "sensors.environmental_rating": "outdoor",
                },
                "fixed_wing_cargo": {
                    "power.compute_power_watts": 10.0,
                    "power.mission_duration_min": 90,
                    "actuators.max_speed_mps": 30,
                    "actuators.payload_kg": 10.0,
                    "power.battery_wh": 500,
                    "autonomy.level": "high",
                },
                "agricultural_sprayer": {
                    "power.compute_power_watts": 5.0,
                    "power.mission_duration_min": 15,
                    "actuators.max_speed_mps": 10,
                    "actuators.payload_kg": 10.0,
                    "power.battery_wh": 150,
                    "autonomy.level": "high",
                    "sensors.environmental_rating": "outdoor",
                },
                "micro_indoor": {
                    "power.compute_power_watts": 1.5,
                    "power.mission_duration_min": 10,
                    "actuators.max_speed_mps": 5,
                    "actuators.payload_kg": 0.0,
                    "power.battery_wh": 15,
                    "autonomy.level": "partial",
                    "sensors.environmental_rating": "indoor",
                },
            },
        ),
        # --- Perception ---
        Question(
            id="perception_tasks",
            dimension="perception",
            text="Which perception tasks does this drone need?",
            explanation=(
                "Each perception task requires specific compute kernels, model "
                "architectures, and latency budgets. Obstacle avoidance at 15m/s "
                "needs <20ms; mapping can tolerate 200ms. This determines the SoC."
            ),
            question_type=QuestionType.MULTI_CHOICE,
            options=[
                "obstacle_avoidance",
                "object_detection",
                "landing_zone_detection",
                "visual_odometry",
                "slam",
                "tracking",
                "terrain_mapping",
                "payload_inspection",
                "person_detection",
            ],
            option_descriptions={
                "obstacle_avoidance": "Detect and avoid obstacles in flight path (latency-critical)",
                "object_detection": "Identify specific objects (packages, vehicles, people)",
                "landing_zone_detection": "Find safe landing areas from aerial view",
                "visual_odometry": "Estimate motion from camera frames (GPS-denied nav)",
                "slam": "Simultaneous localization and mapping",
                "tracking": "Multi-object tracking across frames",
                "terrain_mapping": "Build elevation/occupancy maps for survey",
                "payload_inspection": "Inspect payload or delivery target",
                "person_detection": "Detect people for safety/avoidance (regulatory requirement)",
            },
            required=True,
            min_selections=1,
            implications={
                "obstacle_avoidance": {
                    "perception.detection_classes": ["obstacle"],
                    "perception.max_latency_ms": 20.0,
                    "perception.min_fps": 30.0,
                    "autonomy.obstacle_avoidance": True,
                },
                "object_detection": {
                    "perception.tracking": False,
                    "perception.min_fps": 15.0,
                },
                "visual_odometry": {
                    "perception.camera_types": ["stereo"],
                    "sensors.modalities": ["imu", "barometer"],
                    "sensors.imu_rate_hz": 200.0,
                },
                "slam": {
                    "perception.camera_types": ["stereo"],
                    "sensors.modalities": ["imu"],
                    "autonomy.navigation": "slam",
                },
                "tracking": {
                    "perception.tracking": True,
                    "perception.min_fps": 30.0,
                },
                "person_detection": {
                    "perception.detection_classes": ["person"],
                    "safety.level": "basic",
                },
            },
        ),
        # --- Control output ---
        Question(
            id="control_output",
            dimension="control",
            text="What does the perception output drive?",
            explanation=(
                "The downstream consumer determines the latency budget. A flight "
                "controller at 50Hz needs perception within 20ms. A ground station "
                "alert can tolerate seconds."
            ),
            question_type=QuestionType.MULTI_CHOICE,
            options=[
                "flight_controller",
                "path_planner",
                "mission_abort",
                "ground_station_alert",
                "payload_release",
                "gimbal_tracking",
            ],
            option_descriptions={
                "flight_controller": "Direct input to PX4/ArduPilot (latency-critical, 50-400Hz)",
                "path_planner": "Feed local/global path planning (moderate latency, 5-20Hz)",
                "mission_abort": "Trigger emergency landing or return-to-home",
                "ground_station_alert": "Send alerts to operator (tolerant latency)",
                "payload_release": "Trigger package drop at delivery point",
                "gimbal_tracking": "Control camera gimbal to track objects",
            },
            required=True,
            implications={
                "flight_controller": {
                    "actuators.control_rate_hz": 50.0,
                    "actuators.actuator_types": ["brushless_motor"],
                },
                "path_planner": {
                    "autonomy.planning": "path_planning",
                    "autonomy.decision_rate_hz": 10.0,
                },
                "mission_abort": {
                    "safety.failsafe_action": "land",
                },
            },
        ),
        # --- Environment ---
        Question(
            id="operating_environment",
            dimension="environment",
            text="What is the primary operating environment?",
            explanation=(
                "Environment determines sensor modalities (IR for night, stereo "
                "for GPS-denied), environmental protection (IP rating), and thermal "
                "constraints."
            ),
            question_type=QuestionType.SINGLE_CHOICE,
            options=[
                "outdoor_urban",
                "outdoor_rural",
                "outdoor_maritime",
                "indoor_warehouse",
                "indoor_office",
                "gps_denied",
                "night_operations",
            ],
            option_descriptions={
                "outdoor_urban": "Cities — GPS available, obstacles, people, regulations",
                "outdoor_rural": "Open areas — GPS available, weather exposure",
                "outdoor_maritime": "Over water — salt spray, no landmarks, GPS available",
                "indoor_warehouse": "Structured indoor — no GPS, controlled lighting",
                "indoor_office": "Indoor with people — safety critical, no GPS",
                "gps_denied": "GPS unavailable — requires VIO/SLAM for navigation",
                "night_operations": "Low light — requires IR or active illumination",
            },
            required=False,
            default="outdoor_urban",
            implications={
                "outdoor_urban": {
                    "sensors.environmental_rating": "ip54",
                    "sensors.modalities": ["gps"],
                },
                "outdoor_rural": {
                    "sensors.environmental_rating": "outdoor",
                    "sensors.modalities": ["gps"],
                },
                "outdoor_maritime": {
                    "sensors.environmental_rating": "ip67",
                    "sensors.modalities": ["gps"],
                },
                "indoor_warehouse": {
                    "sensors.environmental_rating": "indoor",
                    "autonomy.navigation": "slam",
                    "perception.camera_types": ["stereo"],
                },
                "indoor_office": {
                    "sensors.environmental_rating": "indoor",
                    "safety.level": "basic",
                    "autonomy.navigation": "slam",
                },
                "gps_denied": {
                    "autonomy.navigation": "vio",
                    "perception.camera_types": ["stereo"],
                    "sensors.imu_rate_hz": 200.0,
                },
                "night_operations": {
                    "perception.camera_types": ["thermal"],
                },
            },
        ),
        # --- Regulatory ---
        Question(
            id="regulatory",
            dimension="safety",
            text="What regulatory requirements apply?",
            explanation=(
                "Regulations constrain weight (<250g for hobby exemption), "
                "operations (BVLOS requires detect-and-avoid), and safety "
                "(over-people requires parachute/redundancy)."
            ),
            question_type=QuestionType.MULTI_CHOICE,
            options=[
                "part_107_vlos",
                "part_107_bvlos",
                "over_people",
                "sub_250g",
                "no_specific_regulation",
            ],
            option_descriptions={
                "part_107_vlos": "FAA Part 107 visual line-of-sight (standard commercial)",
                "part_107_bvlos": "FAA Part 107 beyond visual line-of-sight (requires DAA)",
                "over_people": "Operations over people (Category 2-4, requires safety systems)",
                "sub_250g": "Under 250g total weight (hobbyist exemption in many jurisdictions)",
                "no_specific_regulation": "No specific regulation applies (research/military)",
            },
            required=False,
            default="no_specific_regulation",
            implications={
                "part_107_bvlos": {
                    "safety.level": "sil_1",
                    "perception.detection_classes": ["aircraft"],
                    "autonomy.obstacle_avoidance": True,
                },
                "over_people": {
                    "safety.level": "sil_2",
                    "safety.redundancy": "dual",
                    "safety.failsafe_action": "land",
                },
                "sub_250g": {
                    "power.compute_power_watts": 1.5,
                    "power.battery_wh": 15,
                },
            },
        ),
    ],
)
