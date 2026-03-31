"""UGV (Unmanned Ground Vehicle) domain template — multi-sensor, indoor/outdoor.

Complexity level: MODERATE-HIGH
- Multiple sensor modalities (camera, LiDAR, ultrasonic, bumpers)
- Navigation in structured or semi-structured environments
- Optional manipulation (pick-and-place, door opening)
- Power from large battery packs (longer missions than drones)
- Less constrained on weight/volume than aerial platforms

UGVs add multi-sensor fusion and optional manipulation to the drone baseline.
The navigation problem is fundamentally different: 2D path planning with
obstacle avoidance instead of 3D flight control.
"""

from embodied_ai_architect.qualification.models import (
    DomainTemplate,
    Question,
    QuestionType,
)

TEMPLATE = DomainTemplate(
    domain="ugv",
    display_name="Unmanned Ground Vehicle (UGV / AMR)",
    description=(
        "Ground-based mobile platforms including warehouse AMRs, delivery "
        "robots, agricultural ground vehicles, and exploration rovers. "
        "Multi-sensor, 2D navigation, optional manipulation."
    ),
    keywords=[
        "ugv",
        "amr",
        "ground vehicle",
        "mobile robot",
        "warehouse robot",
        "delivery robot",
        "rover",
        "ground-based",
        "wheeled",
        "tracked",
        "agv",
        "autonomous vehicle",
        "sidewalk",
    ],
    base_implications={
        "platform_type": "amr",
        "sensors.modalities": ["imu", "wheel_encoder"],
        "actuators.actuator_types": ["brushless_motor"],
        "comms.protocols": ["wifi"],
    },
    questions=[
        # --- Platform ---
        Question(
            id="ugv_type",
            dimension="platform",
            text="What type of ground vehicle is this?",
            explanation=(
                "UGV subtype determines size class, payload capacity, speed, "
                "and whether indoor navigation (SLAM) or outdoor navigation "
                "(GPS + path planning) is the primary mode."
            ),
            question_type=QuestionType.SINGLE_CHOICE,
            options=[
                "warehouse_amr",
                "sidewalk_delivery",
                "agricultural_ground",
                "security_patrol",
                "exploration_rover",
                "hospital_logistics",
                "retail_inventory",
            ],
            option_descriptions={
                "warehouse_amr": "Indoor warehouse picking/transport (50-500kg, structured environment)",
                "sidewalk_delivery": "Last-mile sidewalk delivery (10-30kg, urban outdoors)",
                "agricultural_ground": "Field operations (crop monitoring, spraying, harvesting assist)",
                "security_patrol": "Autonomous security patrol (indoor/outdoor, 24/7)",
                "exploration_rover": "Unstructured terrain exploration (mining, disaster response)",
                "hospital_logistics": "Hospital supply transport (sterile corridors, elevators)",
                "retail_inventory": "Retail store inventory scanning (aisles, shelves, barcodes)",
            },
            required=True,
            implications={
                "warehouse_amr": {
                    "power.compute_power_watts": 15.0,
                    "power.battery_wh": 500,
                    "power.mission_duration_min": 480,
                    "actuators.max_speed_mps": 2.0,
                    "actuators.payload_kg": 200.0,
                    "sensors.environmental_rating": "indoor",
                    "autonomy.level": "high",
                    "autonomy.navigation": "slam",
                    "safety.level": "basic",
                },
                "sidewalk_delivery": {
                    "power.compute_power_watts": 10.0,
                    "power.battery_wh": 300,
                    "power.mission_duration_min": 120,
                    "actuators.max_speed_mps": 2.0,
                    "actuators.payload_kg": 15.0,
                    "sensors.environmental_rating": "ip54",
                    "autonomy.level": "conditional",
                    "autonomy.navigation": "gps_waypoint",
                    "safety.level": "basic",
                },
                "agricultural_ground": {
                    "power.compute_power_watts": 10.0,
                    "power.battery_wh": 1000,
                    "power.mission_duration_min": 300,
                    "actuators.max_speed_mps": 3.0,
                    "actuators.payload_kg": 50.0,
                    "sensors.environmental_rating": "ip67",
                    "autonomy.level": "high",
                    "autonomy.navigation": "gps_waypoint",
                },
                "security_patrol": {
                    "power.compute_power_watts": 15.0,
                    "power.battery_wh": 400,
                    "power.mission_duration_min": 240,
                    "actuators.max_speed_mps": 3.0,
                    "actuators.payload_kg": 5.0,
                    "sensors.environmental_rating": "ip54",
                    "autonomy.level": "high",
                    "autonomy.navigation": "slam",
                    "safety.level": "basic",
                },
                "exploration_rover": {
                    "power.compute_power_watts": 20.0,
                    "power.battery_wh": 800,
                    "power.mission_duration_min": 180,
                    "actuators.max_speed_mps": 1.0,
                    "actuators.payload_kg": 30.0,
                    "sensors.environmental_rating": "ip67",
                    "autonomy.level": "conditional",
                    "autonomy.navigation": "slam",
                },
                "hospital_logistics": {
                    "power.compute_power_watts": 10.0,
                    "power.battery_wh": 300,
                    "power.mission_duration_min": 480,
                    "actuators.max_speed_mps": 1.5,
                    "actuators.payload_kg": 50.0,
                    "sensors.environmental_rating": "indoor",
                    "autonomy.level": "high",
                    "autonomy.navigation": "slam",
                    "safety.level": "sil_1",
                },
                "retail_inventory": {
                    "power.compute_power_watts": 8.0,
                    "power.battery_wh": 200,
                    "power.mission_duration_min": 300,
                    "actuators.max_speed_mps": 1.0,
                    "actuators.payload_kg": 5.0,
                    "sensors.environmental_rating": "indoor",
                    "autonomy.level": "high",
                    "autonomy.navigation": "slam",
                },
            },
        ),
        # --- Perception ---
        Question(
            id="perception_tasks",
            dimension="perception",
            text="Which perception tasks does this vehicle need?",
            explanation=(
                "Ground vehicles face different perception challenges than aerial: "
                "2D obstacle avoidance at ground level, pedestrian detection for "
                "safety, and potentially manipulation-related perception."
            ),
            question_type=QuestionType.MULTI_CHOICE,
            options=[
                "obstacle_avoidance_2d",
                "pedestrian_detection",
                "lane_following",
                "object_recognition",
                "barcode_reading",
                "slam_navigation",
                "person_following",
                "manipulation_vision",
                "terrain_classification",
            ],
            option_descriptions={
                "obstacle_avoidance_2d": "Detect and avoid obstacles at ground level",
                "pedestrian_detection": "Detect people for safety (critical in shared spaces)",
                "lane_following": "Follow painted lines or virtual lanes",
                "object_recognition": "Identify specific objects (packages, pallets, targets)",
                "barcode_reading": "Read barcodes/QR codes on packages or shelves",
                "slam_navigation": "Build and use maps for autonomous navigation",
                "person_following": "Follow a specific person (delivery handoff, guide mode)",
                "manipulation_vision": "Guide a manipulator arm for pick-and-place",
                "terrain_classification": "Classify traversable vs non-traversable terrain",
            },
            required=True,
            min_selections=1,
            implications={
                "obstacle_avoidance_2d": {
                    "perception.detection_classes": ["obstacle"],
                    "perception.max_latency_ms": 50.0,
                    "perception.min_fps": 15.0,
                    "autonomy.obstacle_avoidance": True,
                    "sensors.modalities": ["lidar", "ultrasonic"],
                },
                "pedestrian_detection": {
                    "perception.detection_classes": ["person"],
                    "perception.max_latency_ms": 30.0,
                    "perception.min_fps": 20.0,
                    "safety.level": "basic",
                },
                "slam_navigation": {
                    "autonomy.navigation": "slam",
                    "sensors.modalities": ["lidar"],
                    "sensors.lidar_points_per_sec": 300000,
                },
                "manipulation_vision": {
                    "perception.camera_types": ["depth"],
                    "perception.min_accuracy": 0.85,
                    "actuators.dof": 6,
                },
                "barcode_reading": {
                    "perception.camera_types": ["monocular"],
                    "perception.resolution": "1920x1080",
                },
            },
        ),
        # --- Control output ---
        Question(
            id="control_output",
            dimension="control",
            text="What does the perception output drive?",
            explanation=(
                "Ground vehicles typically feed a path planner rather than a direct "
                "motor controller. The latency budget is more relaxed than aerial "
                "(seconds for path planning vs milliseconds for flight control)."
            ),
            question_type=QuestionType.MULTI_CHOICE,
            options=[
                "path_planner",
                "motor_controller",
                "manipulator_controller",
                "fleet_management",
                "human_alert",
                "elevator_interface",
            ],
            option_descriptions={
                "path_planner": "Local or global path planning (5-20Hz decision rate)",
                "motor_controller": "Direct wheel motor control (safety stop, speed adjust)",
                "manipulator_controller": "Guide robotic arm for manipulation tasks",
                "fleet_management": "Report status/location to fleet management system",
                "human_alert": "Alert nearby humans (audible/visual warnings)",
                "elevator_interface": "Interface with elevators for multi-floor operation",
            },
            required=True,
            implications={
                "path_planner": {
                    "autonomy.planning": "path_planning",
                    "autonomy.decision_rate_hz": 10.0,
                },
                "motor_controller": {
                    "actuators.control_rate_hz": 50.0,
                },
                "manipulator_controller": {
                    "actuators.dof": 6,
                    "actuators.control_rate_hz": 100.0,
                },
                "fleet_management": {
                    "comms.protocols": ["wifi", "mqtt"],
                    "autonomy.multi_agent": True,
                },
            },
        ),
        # --- Sensor suite ---
        Question(
            id="sensor_suite",
            dimension="environment",
            text="What sensor modalities beyond cameras?",
            explanation=(
                "Ground vehicles often use multi-sensor fusion. LiDAR provides "
                "reliable depth. Ultrasonic handles close-range. Bumpers are the "
                "last line of defense. The sensor suite drives compute requirements."
            ),
            question_type=QuestionType.MULTI_CHOICE,
            options=[
                "2d_lidar",
                "3d_lidar",
                "ultrasonic",
                "bumper_switches",
                "wheel_encoders",
                "depth_camera",
                "thermal_camera",
                "radar",
            ],
            option_descriptions={
                "2d_lidar": "2D scanning LiDAR for SLAM and obstacle detection",
                "3d_lidar": "3D LiDAR for rich environment mapping",
                "ultrasonic": "Short-range proximity sensors (0.2-3m)",
                "bumper_switches": "Contact sensors as last line of defense",
                "wheel_encoders": "Odometry from wheel rotation",
                "depth_camera": "RGB-D camera (Intel RealSense, etc.)",
                "thermal_camera": "Thermal imaging for night/people detection",
                "radar": "Millimeter-wave radar for weather-independent detection",
            },
            required=False,
            default="2d_lidar",
            implications={
                "3d_lidar": {
                    "sensors.lidar_points_per_sec": 600000,
                    "sensors.data_rate_mbps": 100.0,
                },
                "2d_lidar": {
                    "sensors.lidar_points_per_sec": 300000,
                },
            },
        ),
        # --- Safety ---
        Question(
            id="human_interaction",
            dimension="safety",
            text="Does this vehicle operate around people?",
            explanation=(
                "Vehicles operating in shared human spaces need pedestrian "
                "detection, safety-rated stopping, and often audible/visual "
                "warnings. This significantly impacts the safety architecture."
            ),
            question_type=QuestionType.SINGLE_CHOICE,
            options=[
                "no_humans",
                "humans_nearby",
                "humans_close_contact",
            ],
            option_descriptions={
                "no_humans": "Restricted area, no human interaction expected",
                "humans_nearby": "Shared space, humans walk near but don't touch",
                "humans_close_contact": "Direct interaction (delivery handoff, following)",
            },
            required=True,
            implications={
                "no_humans": {
                    "safety.level": "basic",
                },
                "humans_nearby": {
                    "safety.level": "sil_1",
                    "safety.failsafe_action": "stop",
                    "perception.detection_classes": ["person"],
                },
                "humans_close_contact": {
                    "safety.level": "sil_2",
                    "safety.failsafe_action": "stop",
                    "safety.redundancy": "dual",
                    "perception.detection_classes": ["person", "hand"],
                },
            },
        ),
    ],
)
