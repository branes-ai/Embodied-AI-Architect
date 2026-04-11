#!/usr/bin/env python3
"""Enrich platform YAML files with additional keywords, attributes, and implications.

Targets platforms with <20 total keywords and enriches them to 20-50 keywords.
Preserves all existing content and structure.
"""

import pathlib
import re
import yaml
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
PLATFORMS_DIR = ROOT / "data" / "platforms"

# ---------------------------------------------------------------------------
# Category knowledge base: domain-specific keywords to draw from
# ---------------------------------------------------------------------------

CATEGORY_KNOWLEDGE = {
    "aerial": {
        "identity_extras": [
            "UAV", "unmanned aerial vehicle", "drone", "multirotor", "quadcopter",
            "RPAS", "remotely piloted aircraft", "UAS", "unmanned aircraft system",
            "VTOL", "vertical takeoff", "flying robot",
        ],
        "application_extras": [
            "aerial survey", "aerial inspection", "aerial mapping", "photogrammetry",
            "aerial photography", "crop monitoring", "infrastructure inspection",
            "search and rescue", "surveillance", "package delivery", "precision agriculture",
            "disaster response", "environmental monitoring", "power line inspection",
            "wind turbine inspection", "bridge inspection",
        ],
        "industry_extras": [
            "aerospace", "defense", "agriculture", "energy", "construction",
            "logistics", "public safety", "telecommunications", "mining",
            "oil and gas", "environmental services", "film production",
        ],
        "component_extras": [
            "flight controller", "ESC", "brushless motor", "propeller", "GPS module",
            "IMU", "barometer", "magnetometer", "LiPo battery", "gimbal",
            "FPV camera", "telemetry radio", "receiver", "power distribution board",
            "obstacle avoidance sensor", "RTK GPS", "downward vision sensor",
        ],
        "related_extras": [
            "flight planning", "waypoint navigation", "geofencing", "return to home",
            "autonomous flight", "BVLOS", "sense and avoid", "airspace management",
            "flight logging", "battery management", "payload integration",
            "ground control station", "MAVLink", "PX4", "ArduPilot",
        ],
        "descriptions_extras": [
            "unmanned aerial platform for autonomous missions",
            "drone system with onboard perception and navigation",
            "aerial robot for beyond visual line of sight operations",
        ],
        "default_perception": {
            "camera_types": ["monocular", "stereo"],
            "detection_classes": ["obstacle", "person", "vehicle", "landing_zone"],
            "max_latency_ms": 100,
            "min_fps": 15,
        },
    },
    "aerospace": {
        "identity_extras": [
            "spacecraft", "satellite", "launch vehicle", "space system",
            "orbital platform", "space robot", "aerospace system",
        ],
        "application_extras": [
            "orbital maneuvering", "satellite servicing", "space debris removal",
            "on-orbit assembly", "Earth observation", "deep space exploration",
            "space station maintenance", "rendezvous and docking", "attitude control",
        ],
        "industry_extras": [
            "space industry", "defense aerospace", "satellite communications",
            "space exploration", "NewSpace", "launch services",
        ],
        "component_extras": [
            "star tracker", "reaction wheel", "sun sensor", "radiation-hardened processor",
            "solar panel", "thruster", "docking mechanism", "thermal radiator",
        ],
        "related_extras": [
            "orbital mechanics", "space radiation", "thermal vacuum",
            "microgravity", "debris tracking", "TLE", "CCSDS",
        ],
        "descriptions_extras": [
            "space-grade autonomous system for orbital operations",
            "radiation-hardened platform for space missions",
        ],
        "default_perception": {
            "camera_types": ["monocular", "star_tracker"],
            "detection_classes": ["debris", "satellite", "docking_target"],
            "max_latency_ms": 200,
            "min_fps": 10,
        },
    },
    "agriculture": {
        "identity_extras": [
            "agribot", "farm robot", "agricultural robot", "precision agriculture system",
            "smart farming platform", "agri-tech robot", "autonomous farm vehicle",
        ],
        "application_extras": [
            "crop monitoring", "weed detection", "precision spraying", "yield estimation",
            "soil analysis", "planting", "harvesting", "fruit picking", "pruning",
            "crop scouting", "irrigation management", "phenotyping",
            "pest detection", "disease detection", "nutrient mapping",
        ],
        "industry_extras": [
            "precision agriculture", "smart farming", "agri-tech", "horticulture",
            "viticulture", "organic farming", "controlled environment agriculture",
            "vertical farming", "food production",
        ],
        "component_extras": [
            "multispectral camera", "NDVI sensor", "RTK GPS", "spray nozzle",
            "soil moisture sensor", "weather station", "hyperspectral camera",
            "robotic arm", "conveyor", "seed dispenser",
        ],
        "related_extras": [
            "NDVI", "vegetation index", "variable rate application", "site-specific management",
            "crop health", "field mapping", "row following", "headland turning",
        ],
        "descriptions_extras": [
            "autonomous platform for precision agriculture tasks",
            "farm robot for automated crop management and monitoring",
        ],
        "default_perception": {
            "camera_types": ["monocular", "multispectral"],
            "detection_classes": ["crop", "weed", "pest", "disease", "obstacle"],
            "max_latency_ms": 200,
            "min_fps": 10,
        },
    },
    "assistive": {
        "identity_extras": [
            "assistive robot", "companion robot", "care robot", "mobility aid",
            "rehabilitation robot", "service robot", "social robot",
        ],
        "application_extras": [
            "mobility assistance", "daily living support", "rehabilitation therapy",
            "companionship", "medication reminder", "fall detection", "gait assistance",
            "object fetching", "communication aid", "cognitive assistance",
            "elder care", "disability support",
        ],
        "industry_extras": [
            "assistive technology", "healthcare robotics", "elder care",
            "rehabilitation", "disability services", "social robotics",
        ],
        "component_extras": [
            "force sensor", "voice interface", "touch screen", "proximity sensor",
            "speaker", "microphone array", "soft gripper", "load cell",
        ],
        "related_extras": [
            "human-robot interaction", "HRI", "accessibility", "universal design",
            "adaptive interface", "gesture recognition", "emotion detection",
        ],
        "descriptions_extras": [
            "robot system that assists people with daily activities",
            "assistive platform for mobility or cognitive support",
        ],
        "default_perception": {
            "camera_types": ["monocular", "depth"],
            "detection_classes": ["person", "face", "gesture", "object", "obstacle"],
            "max_latency_ms": 100,
            "min_fps": 15,
        },
    },
    "construction": {
        "identity_extras": [
            "construction robot", "building robot", "construction automation system",
            "autonomous construction vehicle", "construction drone",
        ],
        "application_extras": [
            "site survey", "progress monitoring", "bricklaying", "concrete pouring",
            "demolition", "excavation", "rebar tying", "welding", "painting",
            "3D printing", "structural inspection", "as-built verification",
            "safety monitoring", "dust suppression",
        ],
        "industry_extras": [
            "construction", "civil engineering", "architecture", "building",
            "infrastructure", "real estate development", "heavy industry",
        ],
        "component_extras": [
            "total station", "laser scanner", "BIM integration",
            "hydraulic actuator", "concrete pump", "robotic arm",
        ],
        "related_extras": [
            "BIM", "digital twin", "site safety", "progress tracking",
            "point cloud", "as-built model", "scheduling",
        ],
        "descriptions_extras": [
            "autonomous construction platform for building site operations",
            "robot system for automated construction tasks",
        ],
        "default_perception": {
            "camera_types": ["monocular", "stereo", "lidar"],
            "detection_classes": ["person", "vehicle", "structure", "obstacle", "PPE"],
            "max_latency_ms": 100,
            "min_fps": 15,
        },
    },
    "consumer": {
        "identity_extras": [
            "consumer robot", "home robot", "personal robot", "household robot",
            "smart home device", "domestic robot",
        ],
        "application_extras": [
            "home cleaning", "lawn care", "pool cleaning", "entertainment",
            "home security", "pet care", "cooking assistance", "education",
            "telepresence", "home automation", "personal assistant",
        ],
        "industry_extras": [
            "consumer electronics", "home appliances", "smart home",
            "IoT", "consumer robotics", "home automation",
        ],
        "component_extras": [
            "Wi-Fi module", "speaker", "microphone", "LED display",
            "rechargeable battery", "charging dock", "touch sensor",
        ],
        "related_extras": [
            "app control", "voice assistant", "smart home integration",
            "OTA update", "user-friendly", "companion app",
        ],
        "descriptions_extras": [
            "consumer robotic device for everyday home use",
            "smart home robot with autonomous operation capabilities",
        ],
        "default_perception": {
            "camera_types": ["monocular"],
            "detection_classes": ["person", "pet", "furniture", "obstacle"],
            "max_latency_ms": 200,
            "min_fps": 10,
        },
    },
    "datacenter": {
        "identity_extras": [
            "datacenter robot", "server room robot", "data center automation",
            "rack management system", "datacenter operations platform",
        ],
        "application_extras": [
            "server inspection", "cable management", "thermal monitoring",
            "asset tracking", "hardware inventory", "rack deployment",
            "environmental monitoring", "hot spot detection", "capacity planning",
        ],
        "industry_extras": [
            "data center operations", "cloud infrastructure", "IT operations",
            "colocation", "hyperscale", "edge computing",
        ],
        "component_extras": [
            "thermal camera", "barcode scanner", "RFID reader",
            "environmental sensor", "humidity sensor", "air flow sensor",
        ],
        "related_extras": [
            "DCIM", "PUE", "airflow management", "hot aisle",
            "cold aisle", "rack unit", "uptime", "redundancy",
        ],
        "descriptions_extras": [
            "autonomous data center platform for infrastructure management",
            "robot for automated data center inspection and monitoring",
        ],
        "default_perception": {
            "camera_types": ["monocular", "thermal"],
            "detection_classes": ["server", "cable", "hot_spot", "person"],
            "max_latency_ms": 500,
            "min_fps": 5,
        },
    },
    "edge_vision": {
        "identity_extras": [
            "edge vision system", "smart camera", "AI camera", "vision appliance",
            "edge AI device", "intelligent camera", "video analytics device",
        ],
        "application_extras": [
            "video analytics", "object detection", "people counting", "face recognition",
            "license plate recognition", "anomaly detection", "behavior analysis",
            "occupancy monitoring", "queue detection", "safety compliance",
            "retail analytics", "traffic monitoring",
        ],
        "industry_extras": [
            "computer vision", "edge computing", "security", "retail",
            "smart city", "transportation", "industrial automation",
        ],
        "component_extras": [
            "CMOS sensor", "ISP", "NPU", "AI accelerator", "lens",
            "IR LED", "PoE module", "edge processor", "FPGA",
        ],
        "related_extras": [
            "inference at edge", "model optimization", "TensorRT",
            "OpenVINO", "ONNX", "INT8 quantization", "pruning",
            "on-device AI", "low-latency inference",
        ],
        "descriptions_extras": [
            "edge-deployed vision system with on-device AI inference",
            "smart camera platform for real-time video analytics",
        ],
        "default_perception": {
            "camera_types": ["monocular"],
            "detection_classes": ["person", "vehicle", "object"],
            "max_latency_ms": 50,
            "min_fps": 25,
        },
    },
    "energy": {
        "identity_extras": [
            "energy inspection robot", "power grid robot", "utility robot",
            "energy sector platform", "power plant robot",
        ],
        "application_extras": [
            "power line inspection", "substation monitoring", "solar panel inspection",
            "wind turbine inspection", "transformer monitoring", "grid maintenance",
            "pipeline inspection", "thermal inspection", "insulator inspection",
            "vegetation management", "fault detection",
        ],
        "industry_extras": [
            "energy", "utilities", "power generation", "renewable energy",
            "oil and gas", "nuclear", "grid operations", "solar", "wind",
        ],
        "component_extras": [
            "thermal camera", "corona camera", "LiDAR", "gas sensor",
            "current sensor", "partial discharge sensor", "UV camera",
        ],
        "related_extras": [
            "SCADA", "grid reliability", "outage prevention",
            "predictive maintenance", "asset management", "condition monitoring",
        ],
        "descriptions_extras": [
            "autonomous inspection platform for energy infrastructure",
            "robot for power grid monitoring and maintenance",
        ],
        "default_perception": {
            "camera_types": ["monocular", "thermal"],
            "detection_classes": ["defect", "hot_spot", "vegetation", "corrosion"],
            "max_latency_ms": 200,
            "min_fps": 10,
        },
    },
    "entertainment": {
        "identity_extras": [
            "entertainment robot", "show robot", "animatronic", "performance robot",
            "theme park robot", "interactive robot",
        ],
        "application_extras": [
            "live performance", "audience interaction", "theme park attraction",
            "light show", "dance performance", "character performance",
            "interactive exhibit", "educational entertainment", "drone show",
        ],
        "industry_extras": [
            "entertainment", "theme parks", "events", "media production",
            "live events", "experiential marketing", "museum",
        ],
        "component_extras": [
            "LED array", "speaker system", "servo motor", "motion controller",
            "DMX controller", "fog machine", "projection mapping",
        ],
        "related_extras": [
            "choreography", "synchronized motion", "show control",
            "audience engagement", "motion capture", "expressive movement",
        ],
        "descriptions_extras": [
            "robot platform for live entertainment and audience interaction",
            "automated performance system for events and shows",
        ],
        "default_perception": {
            "camera_types": ["monocular", "depth"],
            "detection_classes": ["person", "gesture", "face"],
            "max_latency_ms": 100,
            "min_fps": 15,
        },
    },
    "food": {
        "identity_extras": [
            "food robot", "food processing robot", "kitchen robot",
            "food service robot", "culinary automation system",
        ],
        "application_extras": [
            "food preparation", "cooking", "plating", "ingredient handling",
            "food sorting", "quality inspection", "food packaging",
            "food delivery", "beverage dispensing", "baking",
        ],
        "industry_extras": [
            "food service", "food processing", "restaurant", "catering",
            "food manufacturing", "hospitality", "quick service restaurant",
        ],
        "component_extras": [
            "food-grade gripper", "temperature sensor", "weight scale",
            "conveyor belt", "dispensing nozzle", "heating element",
        ],
        "related_extras": [
            "food safety", "HACCP", "sanitation", "food-grade materials",
            "allergen handling", "portion control", "recipe execution",
        ],
        "descriptions_extras": [
            "automated food preparation and handling platform",
            "robot for commercial kitchen or food processing operations",
        ],
        "default_perception": {
            "camera_types": ["monocular"],
            "detection_classes": ["food_item", "ingredient", "container", "person"],
            "max_latency_ms": 100,
            "min_fps": 15,
        },
    },
    "forestry": {
        "identity_extras": [
            "forestry robot", "timber robot", "logging robot",
            "forest management platform", "silviculture robot",
        ],
        "application_extras": [
            "tree inventory", "timber harvesting", "reforestation", "fire monitoring",
            "pest detection", "trail maintenance", "wildlife monitoring",
            "tree health assessment", "canopy analysis", "biomass estimation",
        ],
        "industry_extras": [
            "forestry", "timber", "logging", "conservation",
            "wildlife management", "fire prevention", "environmental services",
        ],
        "component_extras": [
            "LiDAR", "multispectral camera", "chainsaw attachment",
            "tree diameter sensor", "GPS/GNSS", "rugged chassis",
        ],
        "related_extras": [
            "DBH measurement", "canopy cover", "forest inventory",
            "timber volume", "species classification", "fire risk index",
        ],
        "descriptions_extras": [
            "autonomous platform for forestry operations and monitoring",
            "robot for forest management and timber operations",
        ],
        "default_perception": {
            "camera_types": ["monocular", "multispectral", "lidar"],
            "detection_classes": ["tree", "obstacle", "person", "fire", "wildlife"],
            "max_latency_ms": 200,
            "min_fps": 10,
        },
    },
    "ground_legged": {
        "identity_extras": [
            "legged robot", "walking robot", "quadruped", "biped",
            "humanoid robot", "legged platform", "spot-like robot",
        ],
        "application_extras": [
            "rough terrain traversal", "stair climbing", "industrial inspection",
            "search and rescue", "patrol", "mapping", "load carrying",
            "hazardous environment exploration", "construction site inspection",
        ],
        "industry_extras": [
            "industrial inspection", "defense", "public safety",
            "construction", "energy", "mining", "research",
        ],
        "component_extras": [
            "actuated leg", "IMU", "LiDAR", "depth camera", "force-torque sensor",
            "joint encoder", "servo motor", "battery pack",
        ],
        "related_extras": [
            "gait control", "dynamic balance", "SLAM", "foothold planning",
            "terrain adaptation", "compliant locomotion", "whole-body control",
        ],
        "descriptions_extras": [
            "legged robot platform for traversing unstructured terrain",
            "walking robot with dynamic balance for inspection tasks",
        ],
        "default_perception": {
            "camera_types": ["stereo", "depth", "lidar"],
            "detection_classes": ["obstacle", "person", "stair", "terrain"],
            "max_latency_ms": 50,
            "min_fps": 20,
        },
    },
    "ground_tracked": {
        "identity_extras": [
            "tracked robot", "tracked vehicle", "crawler robot",
            "tracked platform", "UGV", "unmanned ground vehicle",
        ],
        "application_extras": [
            "rough terrain navigation", "bomb disposal", "mine clearance",
            "pipeline inspection", "agricultural tillage", "demolition",
            "hazmat response", "military reconnaissance", "cargo transport",
        ],
        "industry_extras": [
            "defense", "mining", "agriculture", "hazmat response",
            "construction", "public safety", "EOD",
        ],
        "component_extras": [
            "track assembly", "drive sprocket", "idler wheel", "track tensioner",
            "robotic arm", "manipulator", "thermal camera", "radiation sensor",
        ],
        "related_extras": [
            "terrain mobility", "slip compensation", "skid steering",
            "payload capacity", "rugged design", "all-terrain",
        ],
        "descriptions_extras": [
            "tracked ground robot for challenging terrain operations",
            "unmanned tracked vehicle for hazardous environment tasks",
        ],
        "default_perception": {
            "camera_types": ["monocular", "stereo", "thermal"],
            "detection_classes": ["obstacle", "person", "vehicle", "hazard"],
            "max_latency_ms": 100,
            "min_fps": 15,
        },
    },
    "ground_wheeled": {
        "identity_extras": [
            "wheeled robot", "mobile robot", "AGV", "AMR", "autonomous mobile robot",
            "wheeled platform", "UGV", "unmanned ground vehicle", "self-driving vehicle",
        ],
        "application_extras": [
            "material transport", "warehouse automation", "last-mile delivery",
            "campus delivery", "patrol", "floor cleaning", "inventory scanning",
            "guided vehicle", "tugger", "sortation",
        ],
        "industry_extras": [
            "logistics", "warehousing", "manufacturing", "retail",
            "hospitality", "healthcare logistics", "e-commerce fulfillment",
        ],
        "component_extras": [
            "wheel encoder", "LiDAR", "ultrasonic sensor", "bumper sensor",
            "motor driver", "caster wheel", "battery management system",
        ],
        "related_extras": [
            "path planning", "SLAM", "fleet management", "traffic control",
            "docking station", "obstacle avoidance", "map building",
        ],
        "descriptions_extras": [
            "wheeled mobile robot for autonomous transport and navigation",
            "autonomous ground vehicle for logistics or patrol tasks",
        ],
        "default_perception": {
            "camera_types": ["monocular", "lidar"],
            "detection_classes": ["person", "obstacle", "pallet", "vehicle"],
            "max_latency_ms": 100,
            "min_fps": 15,
        },
    },
    "healthcare": {
        "identity_extras": [
            "healthcare robot", "medical robot", "hospital robot",
            "clinical robot", "health-tech platform",
        ],
        "application_extras": [
            "patient monitoring", "telemedicine", "medication dispensing",
            "surgical assistance", "disinfection", "specimen transport",
            "rehabilitation", "vital signs monitoring", "patient lifting",
            "clinical workflow automation",
        ],
        "industry_extras": [
            "healthcare", "hospital", "clinical", "medical devices",
            "pharma", "telemedicine", "health-tech",
        ],
        "component_extras": [
            "UV-C lamp", "vital signs sensor", "touch screen", "antimicrobial surface",
            "medical-grade enclosure", "sterile interface",
        ],
        "related_extras": [
            "HIPAA", "FDA clearance", "clinical workflow", "EHR integration",
            "patient safety", "infection control", "regulatory compliance",
        ],
        "descriptions_extras": [
            "robot platform for healthcare delivery and clinical support",
            "medical automation system for hospital environments",
        ],
        "default_perception": {
            "camera_types": ["monocular", "depth"],
            "detection_classes": ["person", "patient", "medical_device", "obstacle"],
            "max_latency_ms": 100,
            "min_fps": 15,
        },
    },
    "lab": {
        "identity_extras": [
            "lab robot", "laboratory automation", "lab automation system",
            "scientific robot", "research robot", "analytical robot",
        ],
        "application_extras": [
            "sample handling", "pipetting", "plate handling", "centrifugation",
            "PCR setup", "high-throughput screening", "compound management",
            "cell culture", "assay preparation", "liquid handling",
        ],
        "industry_extras": [
            "life sciences", "pharmaceutical", "biotech", "clinical laboratory",
            "academic research", "analytical chemistry", "genomics",
        ],
        "component_extras": [
            "pipette tip", "microplate", "liquid handler", "plate reader",
            "barcode scanner", "incubator", "centrifuge", "robotic arm",
        ],
        "related_extras": [
            "LIMS", "SiLA", "sample tracking", "GLP", "throughput",
            "reproducibility", "assay development", "automation protocol",
        ],
        "descriptions_extras": [
            "laboratory automation platform for high-throughput experiments",
            "robotic system for automated lab sample processing",
        ],
        "default_perception": {
            "camera_types": ["monocular"],
            "detection_classes": ["vial", "plate", "tip", "barcode"],
            "max_latency_ms": 200,
            "min_fps": 10,
        },
    },
    "logistics": {
        "identity_extras": [
            "logistics robot", "warehouse robot", "fulfillment robot",
            "delivery robot", "sortation robot", "material handling robot",
        ],
        "application_extras": [
            "order picking", "goods-to-person", "sortation", "palletizing",
            "depalletizing", "container unloading", "truck loading",
            "inventory management", "cycle counting", "cross-docking",
        ],
        "industry_extras": [
            "logistics", "supply chain", "e-commerce", "3PL",
            "warehousing", "distribution", "freight", "parcel delivery",
        ],
        "component_extras": [
            "barcode scanner", "conveyor interface", "lift mechanism",
            "roller deck", "tote handler", "pallet jack", "safety scanner",
        ],
        "related_extras": [
            "WMS integration", "fleet management", "pick rate",
            "throughput", "order accuracy", "zone picking", "wave planning",
        ],
        "descriptions_extras": [
            "automated logistics platform for warehouse operations",
            "robot for material handling and order fulfillment",
        ],
        "default_perception": {
            "camera_types": ["monocular", "depth"],
            "detection_classes": ["package", "person", "pallet", "barcode", "obstacle"],
            "max_latency_ms": 100,
            "min_fps": 15,
        },
    },
    "manipulation": {
        "identity_extras": [
            "manipulation robot", "robotic arm", "manipulator", "cobot",
            "collaborative robot", "industrial robot arm", "pick and place robot",
        ],
        "application_extras": [
            "pick and place", "assembly", "welding", "painting", "palletizing",
            "machine tending", "bin picking", "packaging", "polishing",
            "screw driving", "gluing", "inspection handling",
        ],
        "industry_extras": [
            "manufacturing", "automotive", "electronics", "aerospace",
            "consumer goods", "packaging", "metal fabrication",
        ],
        "component_extras": [
            "servo joint", "gripper", "force-torque sensor", "end effector",
            "teach pendant", "controller cabinet", "vision system",
        ],
        "related_extras": [
            "inverse kinematics", "motion planning", "grasp planning",
            "payload capacity", "reach", "repeatability", "cycle time",
        ],
        "descriptions_extras": [
            "robotic manipulation platform for industrial automation",
            "robot arm system for automated material handling and assembly",
        ],
        "default_perception": {
            "camera_types": ["monocular", "depth"],
            "detection_classes": ["object", "part", "bin", "person"],
            "max_latency_ms": 50,
            "min_fps": 20,
        },
    },
    "marine": {
        "identity_extras": [
            "marine robot", "underwater robot", "aquatic robot",
            "maritime autonomous system", "ocean robot",
        ],
        "application_extras": [
            "underwater inspection", "hull inspection", "pipeline survey",
            "marine research", "fish farming", "port security",
            "environmental monitoring", "bathymetric survey", "salvage",
        ],
        "industry_extras": [
            "maritime", "offshore", "aquaculture", "naval",
            "ocean science", "port operations", "subsea",
        ],
        "component_extras": [
            "thruster", "depth sensor", "sonar", "DVL", "acoustic modem",
            "pressure housing", "buoyancy module", "manipulator arm",
        ],
        "related_extras": [
            "underwater navigation", "acoustic positioning", "USBL",
            "depth rating", "corrosion resistance", "marine grade",
        ],
        "descriptions_extras": [
            "autonomous marine platform for underwater or surface operations",
            "marine robot for inspection, survey, or environmental monitoring",
        ],
        "default_perception": {
            "camera_types": ["monocular", "sonar"],
            "detection_classes": ["obstacle", "pipe", "structure", "marine_life"],
            "max_latency_ms": 200,
            "min_fps": 10,
        },
    },
    "marine_surface": {
        "identity_extras": [
            "surface vessel", "USV", "unmanned surface vehicle",
            "autonomous boat", "surface drone", "marine surface robot",
        ],
        "application_extras": [
            "hydrographic survey", "water quality monitoring", "harbor patrol",
            "search and rescue", "oil spill response", "coastal survey",
            "aquaculture monitoring", "cargo transport", "ferry service",
        ],
        "industry_extras": [
            "maritime", "hydrography", "aquaculture", "port operations",
            "environmental services", "coast guard", "naval",
        ],
        "component_extras": [
            "hull", "propulsion motor", "GPS antenna", "radar",
            "AIS transponder", "weather station", "echo sounder",
        ],
        "related_extras": [
            "COLREGs", "collision avoidance", "waypoint navigation",
            "sea state", "station keeping", "autonomous docking",
        ],
        "descriptions_extras": [
            "unmanned surface vessel for autonomous maritime missions",
            "autonomous boat platform for survey and monitoring",
        ],
        "default_perception": {
            "camera_types": ["monocular", "radar"],
            "detection_classes": ["vessel", "obstacle", "buoy", "person"],
            "max_latency_ms": 200,
            "min_fps": 10,
        },
    },
    "marine_underwater": {
        "identity_extras": [
            "AUV", "autonomous underwater vehicle", "ROV", "remotely operated vehicle",
            "submersible", "underwater drone", "subsea robot",
        ],
        "application_extras": [
            "subsea inspection", "pipeline inspection", "hull inspection",
            "oceanographic survey", "mine countermeasures", "wreck exploration",
            "coral reef monitoring", "dam inspection", "cable survey",
        ],
        "industry_extras": [
            "subsea", "offshore oil and gas", "oceanography",
            "marine archaeology", "defense", "aquaculture",
        ],
        "component_extras": [
            "depth sensor", "DVL", "sonar", "thruster", "pressure hull",
            "manipulator", "sample collector", "acoustic modem",
        ],
        "related_extras": [
            "depth rating", "acoustic navigation", "USBL", "INS",
            "buoyancy control", "tether management", "subsea tooling",
        ],
        "descriptions_extras": [
            "autonomous underwater vehicle for subsea inspection and survey",
            "remotely operated underwater robot for deep-sea operations",
        ],
        "default_perception": {
            "camera_types": ["monocular", "sonar"],
            "detection_classes": ["pipe", "structure", "crack", "marine_life"],
            "max_latency_ms": 200,
            "min_fps": 10,
        },
    },
    "military": {
        "identity_extras": [
            "military robot", "tactical robot", "defense platform",
            "combat robot", "military UGV", "MULE", "military drone",
        ],
        "application_extras": [
            "reconnaissance", "ISR", "bomb disposal", "EOD", "logistics resupply",
            "perimeter security", "CBRN detection", "mine clearance",
            "target acquisition", "force protection", "counter-UAS",
        ],
        "industry_extras": [
            "defense", "military", "homeland security", "intelligence",
            "force protection", "C4ISR", "special operations",
        ],
        "component_extras": [
            "ruggedized enclosure", "encrypted radio", "FLIR camera",
            "NBC sensor", "ballistic armor", "weapons mount",
        ],
        "related_extras": [
            "MIL-STD", "ITAR", "C2 system", "tactical network",
            "STANAG", "blue force tracking", "mission planning",
        ],
        "descriptions_extras": [
            "military-grade autonomous platform for tactical operations",
            "defense robot for reconnaissance and force protection",
        ],
        "default_perception": {
            "camera_types": ["monocular", "thermal", "FLIR"],
            "detection_classes": ["person", "vehicle", "threat", "IED"],
            "max_latency_ms": 50,
            "min_fps": 25,
        },
    },
    "mining": {
        "identity_extras": [
            "mining robot", "mine automation system", "autonomous mining vehicle",
            "underground mining robot", "mining drone",
        ],
        "application_extras": [
            "ore extraction", "tunnel inspection", "drilling", "blasting support",
            "conveyor monitoring", "ventilation monitoring", "geotechnical survey",
            "rock bolting", "shotcreting", "hauling", "stockpile management",
        ],
        "industry_extras": [
            "mining", "quarrying", "tunneling", "mineral processing",
            "underground mining", "open pit mining", "geotechnical",
        ],
        "component_extras": [
            "rock drill", "LiDAR", "gas sensor", "proximity detection",
            "collision avoidance system", "ruggedized chassis", "dust filter",
        ],
        "related_extras": [
            "mine safety", "MSHA compliance", "stope planning",
            "ore body modeling", "fleet management", "machine guidance",
        ],
        "descriptions_extras": [
            "autonomous mining platform for underground or surface operations",
            "robot for mine inspection and extraction support",
        ],
        "default_perception": {
            "camera_types": ["monocular", "thermal", "lidar"],
            "detection_classes": ["person", "vehicle", "rock_fall", "obstacle"],
            "max_latency_ms": 100,
            "min_fps": 15,
        },
    },
    "municipal": {
        "identity_extras": [
            "municipal robot", "city robot", "urban service robot",
            "public works robot", "smart city platform",
        ],
        "application_extras": [
            "street cleaning", "snow removal", "pothole detection",
            "park maintenance", "public safety patrol", "waste collection",
            "graffiti removal", "sidewalk delivery", "traffic management",
        ],
        "industry_extras": [
            "municipal services", "public works", "smart city",
            "urban planning", "waste management", "transportation",
        ],
        "component_extras": [
            "sweeper brush", "snow plow", "salt spreader", "GPS",
            "LiDAR", "horn", "safety lights", "reflectors",
        ],
        "related_extras": [
            "municipal fleet", "public ROW", "ADA compliance",
            "sidewalk navigation", "urban mobility", "city ordinance",
        ],
        "descriptions_extras": [
            "autonomous municipal service platform for city operations",
            "robot for public works and urban maintenance tasks",
        ],
        "default_perception": {
            "camera_types": ["monocular", "lidar"],
            "detection_classes": ["person", "vehicle", "obstacle", "curb", "debris"],
            "max_latency_ms": 100,
            "min_fps": 15,
        },
    },
    "oil_gas": {
        "identity_extras": [
            "oil and gas robot", "offshore robot", "refinery robot",
            "pipeline robot", "upstream automation platform",
        ],
        "application_extras": [
            "pipeline inspection", "flare stack inspection", "tank inspection",
            "leak detection", "corrosion monitoring", "valve operation",
            "wellhead monitoring", "rig floor automation", "gas detection",
        ],
        "industry_extras": [
            "oil and gas", "petroleum", "upstream", "midstream",
            "downstream", "petrochemical", "refining", "offshore",
        ],
        "component_extras": [
            "ATEX-rated enclosure", "gas sensor", "thermal camera",
            "ultrasonic thickness gauge", "explosion-proof motor",
        ],
        "related_extras": [
            "ATEX", "IECEx", "intrinsic safety", "hazardous area classification",
            "SIL rating", "process safety", "SCADA integration",
        ],
        "descriptions_extras": [
            "intrinsically safe robot for oil and gas operations",
            "autonomous platform for pipeline and refinery inspection",
        ],
        "default_perception": {
            "camera_types": ["monocular", "thermal"],
            "detection_classes": ["leak", "corrosion", "flame", "person", "obstacle"],
            "max_latency_ms": 100,
            "min_fps": 15,
        },
    },
    "space": {
        "identity_extras": [
            "space robot", "planetary rover", "orbital servicing robot",
            "space manipulator", "extraterrestrial robot",
        ],
        "application_extras": [
            "planetary exploration", "sample collection", "surface traverse",
            "satellite servicing", "orbital assembly", "debris removal",
            "lunar exploration", "Mars exploration", "regolith analysis",
        ],
        "industry_extras": [
            "space exploration", "planetary science", "NASA",
            "ESA", "commercial space", "lunar economy",
        ],
        "component_extras": [
            "rad-hard processor", "solar panel", "RTG", "rocker-bogie suspension",
            "spectrometer", "drill", "sample container", "antenna",
        ],
        "related_extras": [
            "planetary protection", "autonomous navigation", "sol planning",
            "terrain classification", "wheel slip", "thermal cycling",
        ],
        "descriptions_extras": [
            "space-rated robot for planetary or orbital missions",
            "autonomous rover for surface exploration and sample collection",
        ],
        "default_perception": {
            "camera_types": ["monocular", "stereo", "multispectral"],
            "detection_classes": ["rock", "crater", "hazard", "terrain"],
            "max_latency_ms": 500,
            "min_fps": 5,
        },
    },
    "surgical": {
        "identity_extras": [
            "surgical robot", "robot-assisted surgery system", "teleoperated surgical platform",
            "minimally invasive surgical robot", "surgical automation",
        ],
        "application_extras": [
            "minimally invasive surgery", "laparoscopic surgery", "microsurgery",
            "orthopedic surgery", "neurosurgery", "cardiac surgery",
            "robotic-assisted surgery", "suturing", "tissue manipulation",
        ],
        "industry_extras": [
            "surgical robotics", "medical devices", "healthcare",
            "operating room", "clinical", "hospital",
        ],
        "component_extras": [
            "surgical instrument", "endoscope", "trocar", "force-feedback joystick",
            "sterile drape", "vision cart", "patient cart",
        ],
        "related_extras": [
            "haptic feedback", "tremor filtering", "FDA 510(k)",
            "da Vinci", "surgical planning", "image guidance",
        ],
        "descriptions_extras": [
            "robot-assisted surgical system for minimally invasive procedures",
            "surgical platform with teleoperation and tremor compensation",
        ],
        "default_perception": {
            "camera_types": ["stereo", "endoscope"],
            "detection_classes": ["tissue", "instrument", "vessel", "nerve"],
            "max_latency_ms": 10,
            "min_fps": 30,
        },
    },
    "surveillance": {
        "identity_extras": [
            "surveillance system", "security camera", "monitoring system",
            "CCTV", "video surveillance", "security platform",
        ],
        "application_extras": [
            "intruder detection", "perimeter monitoring", "crowd monitoring",
            "traffic surveillance", "license plate recognition", "face detection",
            "behavior analysis", "anomaly detection", "access control",
            "incident detection", "forensic search",
        ],
        "industry_extras": [
            "physical security", "public safety", "law enforcement",
            "retail loss prevention", "critical infrastructure protection",
            "transportation security", "border security",
        ],
        "component_extras": [
            "PTZ camera", "fixed camera", "thermal camera", "IR illuminator",
            "NVR", "VMS", "video encoder", "PoE switch",
        ],
        "related_extras": [
            "video management system", "ONVIF", "RTSP", "video analytics",
            "NDAA compliant", "cybersecurity", "edge recording",
        ],
        "descriptions_extras": [
            "intelligent surveillance system with real-time analytics",
            "video monitoring platform for security and safety applications",
        ],
        "default_perception": {
            "camera_types": ["monocular", "thermal"],
            "detection_classes": ["person", "vehicle", "face", "anomaly"],
            "max_latency_ms": 50,
            "min_fps": 25,
        },
    },
    "telecom": {
        "identity_extras": [
            "telecom robot", "cell tower robot", "network maintenance robot",
            "telecommunications automation platform",
        ],
        "application_extras": [
            "tower inspection", "antenna alignment", "cable routing",
            "site survey", "spectrum monitoring", "small cell installation",
            "fiber optic inspection", "network maintenance",
        ],
        "industry_extras": [
            "telecommunications", "wireless", "5G", "fiber optics",
            "network infrastructure", "tower companies",
        ],
        "component_extras": [
            "spectrum analyzer", "antenna", "signal meter", "climbing mechanism",
            "cable cutter", "fiber splicer",
        ],
        "related_extras": [
            "RF engineering", "antenna pattern", "VSWR", "tower climbing safety",
            "small cell", "mmWave", "macro site",
        ],
        "descriptions_extras": [
            "autonomous platform for telecommunications infrastructure maintenance",
            "robot for cell tower inspection and network equipment servicing",
        ],
        "default_perception": {
            "camera_types": ["monocular"],
            "detection_classes": ["antenna", "cable", "connector", "corrosion"],
            "max_latency_ms": 200,
            "min_fps": 10,
        },
    },
    "textile": {
        "identity_extras": [
            "textile robot", "fabric handling robot", "garment robot",
            "sewing robot", "textile automation system",
        ],
        "application_extras": [
            "fabric cutting", "sewing", "garment assembly", "quality inspection",
            "fabric handling", "pattern matching", "dyeing", "folding",
            "textile sorting", "embroidery",
        ],
        "industry_extras": [
            "textile manufacturing", "apparel", "fashion", "garment",
            "upholstery", "technical textiles", "home furnishing",
        ],
        "component_extras": [
            "fabric gripper", "sewing head", "cutting blade", "tension sensor",
            "color sensor", "pattern recognition camera",
        ],
        "related_extras": [
            "fabric deformation", "thread tension", "seam quality",
            "pattern nesting", "waste reduction", "lean manufacturing",
        ],
        "descriptions_extras": [
            "robot platform for automated textile handling and assembly",
            "automation system for garment manufacturing operations",
        ],
        "default_perception": {
            "camera_types": ["monocular"],
            "detection_classes": ["fabric_edge", "defect", "pattern", "thread"],
            "max_latency_ms": 50,
            "min_fps": 20,
        },
    },
    "underground": {
        "identity_extras": [
            "underground robot", "tunnel robot", "subterranean robot",
            "sewer robot", "pipe inspection robot", "cave exploration robot",
        ],
        "application_extras": [
            "tunnel inspection", "pipe inspection", "sewer inspection",
            "utility mapping", "underground construction", "cable pulling",
            "confined space inspection", "structural assessment",
        ],
        "industry_extras": [
            "utilities", "tunneling", "civil infrastructure",
            "wastewater", "underground construction", "mining",
        ],
        "component_extras": [
            "crawler tracks", "LED lighting", "sonar", "gas sensor",
            "pipe gripper", "thickness gauge", "waterproof enclosure",
        ],
        "related_extras": [
            "CCTV inspection", "manhole entry", "confined space safety",
            "inflow and infiltration", "structural rating", "NASSCO PACP",
        ],
        "descriptions_extras": [
            "robot for inspecting underground infrastructure and tunnels",
            "autonomous platform for subterranean exploration and maintenance",
        ],
        "default_perception": {
            "camera_types": ["monocular"],
            "detection_classes": ["crack", "corrosion", "obstruction", "joint"],
            "max_latency_ms": 200,
            "min_fps": 10,
        },
    },
    "veterinary": {
        "identity_extras": [
            "veterinary robot", "animal care robot", "livestock robot",
            "animal health platform", "vet-tech robot",
        ],
        "application_extras": [
            "animal health monitoring", "automated milking", "livestock sorting",
            "medication delivery", "herd management", "animal tracking",
            "reproductive monitoring", "weight monitoring", "feed optimization",
        ],
        "industry_extras": [
            "veterinary medicine", "animal husbandry", "dairy farming",
            "livestock management", "poultry", "aquaculture",
        ],
        "component_extras": [
            "RFID reader", "temperature sensor", "weighing platform",
            "teat cup", "sorting gate", "feed dispenser",
        ],
        "related_extras": [
            "herd management software", "animal welfare", "lameness detection",
            "body condition score", "estrus detection", "mastitis detection",
        ],
        "descriptions_extras": [
            "automated platform for animal health monitoring and care",
            "robot for livestock management and veterinary tasks",
        ],
        "default_perception": {
            "camera_types": ["monocular", "thermal"],
            "detection_classes": ["animal", "body_part", "behavior", "anomaly"],
            "max_latency_ms": 200,
            "min_fps": 10,
        },
    },
    "waste": {
        "identity_extras": [
            "waste robot", "recycling robot", "trash sorting robot",
            "waste management platform", "refuse handling robot",
        ],
        "application_extras": [
            "waste sorting", "recycling", "material recovery", "trash collection",
            "contamination detection", "bin emptying", "litter picking",
            "compaction", "hazardous waste handling",
        ],
        "industry_extras": [
            "waste management", "recycling", "circular economy",
            "environmental services", "sanitation", "material recovery",
        ],
        "component_extras": [
            "gripper", "conveyor belt", "NIR sensor", "air jet",
            "compactor", "bin lifter", "metal detector",
        ],
        "related_extras": [
            "material classification", "contamination rate", "MRF",
            "single stream", "diversion rate", "tipping floor",
        ],
        "descriptions_extras": [
            "autonomous waste sorting and recycling platform",
            "robot for automated waste handling and material recovery",
        ],
        "default_perception": {
            "camera_types": ["monocular"],
            "detection_classes": ["plastic", "metal", "paper", "glass", "contaminant"],
            "max_latency_ms": 50,
            "min_fps": 20,
        },
    },
    "wearable": {
        "identity_extras": [
            "wearable robot", "exoskeleton", "powered suit",
            "assistive wearable", "body-worn robot", "wearable device",
        ],
        "application_extras": [
            "load bearing", "gait assistance", "rehabilitation",
            "industrial lifting", "military augmentation", "fall prevention",
            "posture correction", "tremor suppression", "endurance enhancement",
        ],
        "industry_extras": [
            "wearable robotics", "exoskeletons", "rehabilitation",
            "industrial ergonomics", "defense", "assistive technology",
        ],
        "component_extras": [
            "actuated joint", "IMU", "EMG sensor", "strain gauge",
            "battery pack", "harness", "control unit", "force sensor",
        ],
        "related_extras": [
            "human augmentation", "ergonomic support", "fatigue reduction",
            "range of motion", "biomechanics", "gait analysis",
        ],
        "descriptions_extras": [
            "wearable robotic platform for human augmentation or rehabilitation",
            "body-worn exoskeleton for strength or mobility support",
        ],
        "default_perception": {
            "camera_types": [],
            "detection_classes": ["gesture", "posture", "terrain"],
            "max_latency_ms": 10,
            "min_fps": 30,
        },
    },
    "weather": {
        "identity_extras": [
            "weather station", "meteorological platform", "weather monitoring system",
            "atmospheric sensor", "weather drone",
        ],
        "application_extras": [
            "weather monitoring", "atmospheric profiling", "storm tracking",
            "precipitation measurement", "wind measurement", "temperature monitoring",
            "humidity sensing", "air quality monitoring", "fog detection",
        ],
        "industry_extras": [
            "meteorology", "aviation weather", "agriculture weather",
            "marine weather", "renewable energy", "climate research",
        ],
        "component_extras": [
            "anemometer", "rain gauge", "barometer", "hygrometer",
            "thermometer", "ceilometer", "radiosonde", "weather radar",
        ],
        "related_extras": [
            "METAR", "SYNOP", "forecast model", "data assimilation",
            "WMO standards", "automatic weather station", "AWS",
        ],
        "descriptions_extras": [
            "autonomous weather monitoring and sensing platform",
            "meteorological system for real-time atmospheric data collection",
        ],
        "default_perception": {
            "camera_types": ["monocular"],
            "detection_classes": ["cloud", "precipitation", "visibility"],
            "max_latency_ms": 1000,
            "min_fps": 1,
        },
    },
}


def get_total_keywords(data: dict) -> int:
    """Count total keywords across all groups."""
    kw = data.get("keywords", {})
    return sum(len(v) for v in kw.values() if isinstance(v, list))


def normalize_keyword(kw: str) -> str:
    """Normalize keyword for dedup comparison."""
    return kw.strip().lower()


def generate_name_variants(name: str, category: str) -> list[str]:
    """Generate keyword variants from the platform name."""
    variants = []
    name_lower = name.lower()
    words = name_lower.split()

    # Category + name
    if category.replace("_", " ") not in name_lower:
        variants.append(f"{category.replace('_', ' ')} {name_lower}")

    # "autonomous X" variant
    if "autonomous" not in name_lower:
        variants.append(f"autonomous {name_lower}")

    # "smart X" variant
    if "smart" not in name_lower and len(words) <= 3:
        variants.append(f"smart {name_lower}")

    # "X system" variant
    if "system" not in name_lower:
        variants.append(f"{name_lower} system")

    # "X platform" variant
    if "platform" not in name_lower:
        variants.append(f"{name_lower} platform")

    # "automated X"
    if "automated" not in name_lower and "autonomous" not in name_lower:
        variants.append(f"automated {name_lower}")

    # "robotic X" (if category suggests robots)
    robot_categories = {
        "ground_wheeled", "ground_tracked", "ground_legged", "aerial",
        "marine", "marine_surface", "marine_underwater", "manipulation",
        "consumer", "assistive", "surgical", "lab", "logistics",
    }
    if category in robot_categories and "robot" not in name_lower:
        variants.append(f"robotic {name_lower}")
        variants.append(f"{name_lower} robot")

    return variants


def generate_description_variants(name: str, category: str, description: str) -> list[str]:
    """Generate natural-language description keywords."""
    name_lower = name.lower()
    cat_display = category.replace("_", " ")
    variants = [
        f"system for {name_lower} applications",
        f"{cat_display} platform using {name_lower} technology",
        f"intelligent {name_lower} solution",
    ]
    return variants


def generate_application_from_name(name: str, category: str) -> list[str]:
    """Derive application keywords from the name."""
    name_lower = name.lower()
    words = [w for w in name_lower.split() if len(w) > 2]
    variants = []

    # "X automation"
    variants.append(f"{name_lower} automation")

    # "real-time X"
    variants.append(f"real-time {name_lower}")

    # "X monitoring"
    if "monitoring" not in name_lower:
        variants.append(f"{name_lower} monitoring")

    # "X detection"
    if "detection" not in name_lower:
        variants.append(f"{name_lower} detection")

    # "X analysis"
    if "analysis" not in name_lower:
        variants.append(f"{name_lower} analysis")

    return variants


def enrich_keywords(data: dict, category: str) -> dict:
    """Add keywords to reach 20-50 total. Returns modified data dict."""
    kw = data.setdefault("keywords", {})
    name = data.get("name", "")
    description = data.get("description", "")

    # Ensure all groups exist
    for group in ["identity", "descriptions", "application", "industry", "components", "related"]:
        if group not in kw or not isinstance(kw[group], list):
            kw[group] = kw.get(group, []) if isinstance(kw.get(group), list) else []

    # Collect existing keywords (normalized) for dedup
    existing = set()
    for group_kws in kw.values():
        if isinstance(group_kws, list):
            for k in group_kws:
                existing.add(normalize_keyword(str(k)))

    def add_unique(group: str, candidates: list[str], max_add: int = 50) -> int:
        """Add candidates to group if not already present. Returns count added."""
        added = 0
        for c in candidates:
            if added >= max_add:
                break
            norm = normalize_keyword(c)
            if norm and norm not in existing and len(norm) > 2:
                kw[group].append(c)
                existing.add(norm)
                added += 1
        return added

    # Get category knowledge
    cat_info = CATEGORY_KNOWLEDGE.get(category, {})

    # 1. Identity keywords: name variants + category identity extras
    name_variants = generate_name_variants(name, category)
    add_unique("identity", name_variants, 5)
    add_unique("identity", cat_info.get("identity_extras", []), 6)

    # 2. Descriptions
    desc_variants = generate_description_variants(name, category, description)
    add_unique("descriptions", desc_variants, 3)
    add_unique("descriptions", cat_info.get("descriptions_extras", []), 3)

    # 3. Application keywords
    app_variants = generate_application_from_name(name, category)
    add_unique("application", app_variants, 4)
    add_unique("application", cat_info.get("application_extras", []), 8)

    # 4. Industry keywords
    add_unique("industry", cat_info.get("industry_extras", []), 6)

    # 5. Components
    add_unique("components", cat_info.get("component_extras", []), 6)

    # 6. Related
    add_unique("related", cat_info.get("related_extras", []), 6)

    # If still under 20, add more from each category pool
    current = get_total_keywords(data)
    if current < 20:
        deficit = 25 - current  # aim for 25
        per_group = max(2, deficit // 4)
        add_unique("application", cat_info.get("application_extras", []), per_group)
        add_unique("industry", cat_info.get("industry_extras", []), per_group)
        add_unique("components", cat_info.get("component_extras", []), per_group)
        add_unique("related", cat_info.get("related_extras", []), per_group)

    return data


def enrich_attributes(data: dict, category_defaults: dict | None) -> dict:
    """Add default attribute ranges if attributes section is empty."""
    attrs = data.get("attributes")
    if attrs and any(v for k, v in attrs.items() if isinstance(v, dict) and "min" in v):
        return data  # Already has attribute ranges

    if category_defaults:
        data["attributes"] = dict(category_defaults)

    return data


def enrich_implications(data: dict, category: str) -> dict:
    """Add basic perception implications if missing."""
    impl = data.get("implications")

    # Check if perception is already set
    if isinstance(impl, dict) and impl.get("perception"):
        perc = impl["perception"]
        if isinstance(perc, dict) and perc.get("camera_types"):
            return data  # Already has perception

    cat_info = CATEGORY_KNOWLEDGE.get(category, {})
    default_perc = cat_info.get("default_perception", {
        "camera_types": ["monocular"],
        "detection_classes": [],
        "max_latency_ms": 100,
        "min_fps": 15,
    })

    if impl is None:
        impl = {}
        data["implications"] = impl

    if not isinstance(impl, dict):
        # implications might be null/None in YAML
        impl = {}
        data["implications"] = impl

    if "perception" not in impl or not impl["perception"]:
        impl["perception"] = default_perc

    return data


def load_category_defaults(category_dir: pathlib.Path) -> dict | None:
    """Load default attributes from _category.yaml."""
    cat_file = category_dir / "_category.yaml"
    if cat_file.exists():
        cat_data = yaml.safe_load(cat_file.read_text())
        if isinstance(cat_data, dict):
            return cat_data.get("default_attributes")
    return None


class FlowStyleDumper(yaml.SafeDumper):
    """Custom YAML dumper that preserves inline dict style for attribute ranges."""
    pass


def represent_ordered_mapping(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


FlowStyleDumper.add_representer(dict, represent_ordered_mapping)


def main():
    skip_names = {"schema.yaml", "taxonomy.yaml"}
    total_files = 0
    enriched_count = 0
    already_rich = 0
    skipped = 0

    for yaml_file in sorted(PLATFORMS_DIR.rglob("*.yaml")):
        if yaml_file.name in skip_names or yaml_file.name.startswith("_"):
            continue
        if "configurations" in yaml_file.parts:
            continue

        text = yaml_file.read_text()
        data = yaml.safe_load(text)

        if not isinstance(data, dict) or "id" not in data:
            skipped += 1
            continue

        total_files += 1
        current_kw = get_total_keywords(data)

        if current_kw >= 20:
            already_rich += 1
            continue

        category = data.get("category", "")

        # Load category defaults
        category_defaults = load_category_defaults(yaml_file.parent)

        # Enrich
        data = enrich_keywords(data, category)
        data = enrich_attributes(data, category_defaults)
        data = enrich_implications(data, category)

        new_kw = get_total_keywords(data)

        # Write back
        yaml_text = yaml.dump(
            data,
            Dumper=FlowStyleDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )
        yaml_file.write_text(yaml_text)
        enriched_count += 1

        print(f"  {yaml_file.relative_to(PLATFORMS_DIR)}: {current_kw} -> {new_kw} keywords")

    print(f"\n{'='*60}")
    print(f"Total platform files: {total_files}")
    print(f"Already rich (20+):   {already_rich}")
    print(f"Enriched:             {enriched_count}")
    print(f"Skipped (non-platform): {skipped}")

    # Verification
    rich = 0
    poor = 0
    for yaml_file in sorted(PLATFORMS_DIR.rglob("*.yaml")):
        if yaml_file.name in skip_names or yaml_file.name.startswith("_"):
            continue
        if "configurations" in yaml_file.parts:
            continue
        data = yaml.safe_load(yaml_file.read_text())
        if not isinstance(data, dict) or "id" not in data:
            continue
        kw = get_total_keywords(data)
        if kw >= 20:
            rich += 1
        else:
            poor += 1

    print(f"\nVerification: {rich} platforms with 20+ keywords, {poor} with <20")
    if rich >= 200:
        print("SUCCESS: Target of 200+ rich platforms met!")
    else:
        print(f"WARNING: Only {rich} rich platforms, need at least 200")


if __name__ == "__main__":
    main()
