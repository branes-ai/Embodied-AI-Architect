#!/usr/bin/env python3
"""Generate platform YAML files from the taxonomy document.

Reads platform IDs from the taxonomy doc and generates YAML files with
realistic attributes, keywords, and context. Uses the existing platforms
as reference for the format.

Usage:
    python scripts/generate_platforms.py [--dry-run]
"""

import re
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Platform data: ID → (name, description, category-specific attributes)
# ---------------------------------------------------------------------------

PLATFORMS = {
    # === AERIAL ===
    "aerial.fixed_wing_cargo": {
        "name": "Fixed-Wing Cargo UAV",
        "description": "Long-range heavy cargo transport UAV with >5kg payload and >60min endurance.",
        "keywords": {
            "identity": ["fixed-wing cargo drone", "cargo UAV", "delivery UAV", "freight drone"],
            "descriptions": ["large drone carrying heavy packages long distance", "fixed-wing cargo aircraft"],
            "application": ["cargo delivery", "medical supply delivery", "remote logistics", "humanitarian aid"],
            "industry": ["BVLOS", "cargo drone", "freight UAV", "long-range delivery"],
            "components": ["fixed wing", "cargo bay", "parachute delivery", "RTK-GPS"],
            "related": ["logistics", "humanitarian", "island delivery", "rural supply"],
        },
        "classification": {"locomotion": "aerial.fixed_wing", "manipulation": "none", "load_class": "none", "environment": ["outdoor_rural", "outdoor_extreme"], "human_proximity": "no_humans"},
        "attributes": {"power_watts": {"min": 50, "max": 500, "typical": 200}, "compute_tops": {"min": 2, "max": 20, "typical": 8}, "latency_ms": {"min": 50, "max": 500, "typical": 200}, "weight_kg": {"min": 10, "max": 200, "typical": 50}, "cost_usd": {"min": 5000, "max": 200000, "typical": 50000}, "autonomy_level": "high", "safety_level": "sil_2", "mission_duration_hours": {"min": 1, "max": 8, "typical": 3}},
        "implications": {"platform_type": "drone", "perception": {"camera_types": ["monocular", "thermal"], "detection_classes": ["obstacle", "landing_zone"], "max_latency_ms": 200, "min_fps": 10}, "sensors": {"modalities": ["imu", "barometer", "gps", "airspeed"]}, "actuators": {"actuator_types": ["brushless_motor", "servo"], "dof": None, "control_rate_hz": 50}, "comms": {"protocols": ["mavlink", "lte", "satellite"]}, "power": {"compute_power_watts": 10}, "safety": {"level": "sil_2", "certifications": ["DO-178C"]}},
        "context": {
            "typical_architecture": "Fixed-wing airframe with pusher or tractor propeller, autopilot (PX4/ArduPilot), and cargo bay with release mechanism. Compute typically Jetson-class for terrain-following and landing zone detection.",
            "common_pitfalls": ["Wind loading on large cargo shifts CG significantly", "BVLOS requires detect-and-avoid system", "Battery capacity limits range more than payload weight"],
            "regulatory": ["FAA Part 107 BVLOS waiver", "EASA specific category", "ICAO unmanned aircraft systems"],
            "reference_designs": ["Zipline P2 Zip", "Wingcopter 198", "Volansi VOLY C10"],
            "design_considerations": "Range is the primary constraint — optimize for aerodynamic efficiency over speed. Payload release mechanism reliability is critical for medical/humanitarian delivery. Terrain-following radar or LiDAR required for low-altitude BVLOS in varied terrain.",
        },
        "qualification": {"suggested_questions": ["payload_weight_kg", "range_km", "delivery_mechanism"], "critical_constraints": ["Payload capacity", "Range", "BVLOS capability"], "optional_constraints": ["Parachute delivery", "Medical cold chain"]},
    },
    "aerial.fixed_wing_military_isr": {
        "name": "Military ISR UAV",
        "description": "Intelligence, surveillance, and reconnaissance fixed-wing UAV with EO/IR sensors and long loiter time.",
        "keywords": {
            "identity": ["ISR drone", "surveillance UAV", "reconnaissance drone", "military UAV"],
            "descriptions": ["military spy drone", "long-endurance surveillance aircraft", "intelligence gathering UAV"],
            "application": ["intelligence gathering", "surveillance", "reconnaissance", "border patrol", "target acquisition"],
            "industry": ["ISR", "SIGINT", "ELINT", "COMINT", "C4ISR", "GEOINT"],
            "components": ["EO/IR gimbal", "SAR radar", "SIGINT payload", "datalink", "SATCOM"],
            "related": ["military", "defense", "border security", "maritime patrol"],
        },
        "classification": {"locomotion": "aerial.fixed_wing", "manipulation": "none", "load_class": "none", "environment": ["outdoor_rural", "outdoor_extreme"], "human_proximity": "no_humans"},
        "attributes": {"power_watts": {"min": 100, "max": 5000, "typical": 1000}, "compute_tops": {"min": 10, "max": 200, "typical": 50}, "latency_ms": {"min": 10, "max": 200, "typical": 50}, "weight_kg": {"min": 20, "max": 15000, "typical": 500}, "cost_usd": {"min": 50000, "max": 50000000, "typical": 2000000}, "autonomy_level": "conditional", "safety_level": "sil_2", "mission_duration_hours": {"min": 4, "max": 40, "typical": 16}},
        "implications": {"platform_type": "drone", "perception": {"camera_types": ["eo_ir_gimbal", "sar_radar"], "detection_classes": ["vehicle", "person", "building"], "max_latency_ms": 50, "min_fps": 30}, "sensors": {"modalities": ["imu", "gps", "sigint", "eo_ir"]}, "actuators": {"actuator_types": ["brushless_motor", "servo"], "dof": None, "control_rate_hz": 50}, "comms": {"protocols": ["mavlink", "satcom", "datalink"]}, "power": {"compute_power_watts": 50}, "safety": {"level": "sil_2", "certifications": []}},
        "context": {
            "typical_architecture": "Large fixed-wing with turbojet or piston engine, multi-sensor payload (EO/IR gimbal, SAR, SIGINT), redundant flight control, encrypted datalinks, and onboard edge AI for real-time target detection and tracking.",
            "common_pitfalls": ["Sensor fusion latency between EO/IR and radar degrades target tracking", "Encrypted datalink bandwidth limits real-time video quality", "GPS jamming/spoofing requires INS fallback"],
            "regulatory": ["ITAR export controls", "National airspace integration", "Frequency allocation for datalinks"],
            "reference_designs": ["General Atomics MQ-9 Reaper", "Northrop Grumman RQ-4 Global Hawk", "AeroVironment Puma AE"],
            "design_considerations": "Onboard AI for automatic target recognition (ATR) is the key differentiator — reduces datalink bandwidth requirements and enables autonomous operation in contested RF environments. Multi-INT fusion (EO/IR + SAR + SIGINT) requires significant compute but dramatically improves intelligence quality.",
        },
        "qualification": {"suggested_questions": ["sensor_payload", "endurance_hours", "altitude_ft"], "critical_constraints": ["Loiter endurance", "Sensor resolution", "Datalink security"], "optional_constraints": ["SAR capability", "SIGINT payload", "Weapon integration"]},
    },
    "aerial.fixed_wing_survey": {
        "name": "Fixed-Wing Survey UAV",
        "description": "Mapping, photogrammetry, and corridor survey UAV with long endurance and large area coverage.",
        "keywords": {
            "identity": ["survey drone", "mapping drone", "photogrammetry UAV", "survey UAV"],
            "descriptions": ["drone that maps large areas", "aerial survey aircraft", "photogrammetry drone"],
            "application": ["aerial mapping", "photogrammetry", "orthomosaic", "corridor survey", "topographic survey", "volumetric survey"],
            "industry": ["GSD", "orthomosaic", "DSM", "DTM", "LiDAR survey", "photogrammetry"],
            "components": ["survey camera", "PPK GPS", "LiDAR scanner", "fixed wing airframe"],
            "related": ["mining survey", "construction progress", "land survey", "forestry inventory"],
        },
        "classification": {"locomotion": "aerial.fixed_wing", "manipulation": "none", "load_class": "none", "environment": ["outdoor_rural"], "human_proximity": "no_humans"},
        "attributes": {"power_watts": {"min": 20, "max": 200, "typical": 80}, "compute_tops": {"min": 1, "max": 10, "typical": 4}, "latency_ms": {"min": 100, "max": 1000, "typical": 500}, "weight_kg": {"min": 2, "max": 25, "typical": 8}, "cost_usd": {"min": 3000, "max": 50000, "typical": 15000}, "autonomy_level": "high", "safety_level": "basic", "mission_duration_hours": {"min": 0.5, "max": 3, "typical": 1.5}},
        "implications": {"platform_type": "drone", "perception": {"camera_types": ["survey_camera", "multispectral"], "detection_classes": [], "max_latency_ms": 500, "min_fps": 1}, "sensors": {"modalities": ["imu", "gps", "barometer"]}, "actuators": {"actuator_types": ["brushless_motor", "servo"], "dof": None, "control_rate_hz": 50}, "comms": {"protocols": ["mavlink", "wifi"]}, "power": {"compute_power_watts": 5}, "safety": {"level": "basic", "certifications": []}},
        "context": {
            "typical_architecture": "Small fixed-wing with electric propulsion, PPK/RTK GPS, high-resolution survey camera (42-61MP), and optional LiDAR. Onboard computer handles flight planning, image triggering, and basic telemetry. Post-processing in cloud (Pix4D, Agisoft).",
            "common_pitfalls": ["Image overlap must be >70% front, >60% side for photogrammetry quality", "Wind affects GSD consistency on long corridors", "PPK base station must be within 30km"],
            "regulatory": ["FAA Part 107", "EASA open/specific category", "Local flight restrictions near airports"],
            "reference_designs": ["senseFly eBee X", "WingtraOne", "Trinity F90+"],
            "design_considerations": "Survey accuracy is determined by GSD (ground sample distance) which is a function of altitude, camera resolution, and sensor size. Real-time compute requirements are minimal — the heavy lifting is post-processing. Focus on flight endurance, camera trigger accuracy, and PPK GPS precision.",
        },
        "qualification": {"suggested_questions": ["survey_area_km2", "gsd_cm", "sensor_type"], "critical_constraints": ["GSD resolution", "Area coverage per flight", "GPS accuracy"], "optional_constraints": ["LiDAR capability", "Multispectral sensors"]},
    },
}

# For platforms not in the detailed dict above, generate a minimal but valid YAML
# This is a fallback — the detailed dict should be expanded over time.


def _generate_minimal_platform(platform_id: str) -> dict:
    """Generate a minimal but valid platform YAML from just the ID."""
    category, name_slug = platform_id.split(".", 1)
    human_name = name_slug.replace("_", " ").title()

    # Sensible defaults by category
    CATEGORY_DEFAULTS = {
        "aerial": {"locomotion": "aerial.rotary_wing", "power": {"min": 5, "max": 500, "typical": 50}, "env": ["outdoor_rural"], "proximity": "no_humans", "platform_type": "drone", "sensors": ["imu", "gps", "barometer"], "comms": ["mavlink", "wifi"], "actuators": ["brushless_motor"]},
        "ground_wheeled": {"locomotion": "ground.wheeled", "power": {"min": 10, "max": 2000, "typical": 100}, "env": ["indoor_structured", "outdoor_urban"], "proximity": "humans_nearby", "platform_type": "amr", "sensors": ["lidar", "imu", "wheel_encoder"], "comms": ["wifi", "ethernet"], "actuators": ["brushless_motor"]},
        "ground_legged": {"locomotion": "ground.legged", "power": {"min": 50, "max": 1000, "typical": 200}, "env": ["indoor_structured", "outdoor_urban"], "proximity": "humans_nearby", "platform_type": "quadruped", "sensors": ["lidar", "imu", "depth_camera"], "comms": ["wifi", "ethernet"], "actuators": ["servo_motor"]},
        "ground_tracked": {"locomotion": "ground.tracked", "power": {"min": 50, "max": 5000, "typical": 500}, "env": ["outdoor_extreme", "hazardous"], "proximity": "no_humans", "platform_type": "ugv", "sensors": ["lidar", "imu", "gps"], "comms": ["wifi", "radio"], "actuators": ["brushless_motor"]},
        "manipulation": {"locomotion": "stationary.fixed_mount", "power": {"min": 100, "max": 5000, "typical": 500}, "env": ["indoor_structured"], "proximity": "humans_nearby", "platform_type": "industrial_arm", "sensors": ["force_torque", "joint_encoder"], "comms": ["ethercat", "ethernet"], "actuators": ["servo_motor"]},
        "surgical": {"locomotion": "stationary.fixed_mount", "power": {"min": 200, "max": 2000, "typical": 800}, "env": ["indoor_sterile"], "proximity": "humans_contact", "platform_type": "industrial_arm", "sensors": ["force_torque", "endoscope"], "comms": ["ethernet"], "actuators": ["servo_motor"]},
        "veterinary": {"locomotion": "stationary.fixed_mount", "power": {"min": 50, "max": 500, "typical": 200}, "env": ["outdoor_farmyard"], "proximity": "humans_nearby", "platform_type": "custom", "sensors": ["camera", "ultrasound"], "comms": ["wifi", "ethernet"], "actuators": ["servo_motor"]},
        "lab": {"locomotion": "stationary.benchtop", "power": {"min": 50, "max": 500, "typical": 200}, "env": ["indoor_lab"], "proximity": "humans_nearby", "platform_type": "custom", "sensors": ["camera", "barcode_reader"], "comms": ["ethernet", "usb"], "actuators": ["stepper_motor"]},
        "marine_surface": {"locomotion": "marine.surface", "power": {"min": 50, "max": 5000, "typical": 500}, "env": ["outdoor_extreme"], "proximity": "no_humans", "platform_type": "custom", "sensors": ["gps", "imu", "radar", "sonar"], "comms": ["wifi", "satellite"], "actuators": ["brushless_motor"]},
        "marine_underwater": {"locomotion": "marine.underwater", "power": {"min": 50, "max": 2000, "typical": 300}, "env": ["underwater_shallow", "underwater_deep"], "proximity": "no_humans", "platform_type": "custom", "sensors": ["imu", "sonar", "depth_sensor"], "comms": ["acoustic"], "actuators": ["thruster"]},
        "marine": {"locomotion": "marine.surface", "power": {"min": 100, "max": 5000, "typical": 500}, "env": ["outdoor_extreme"], "proximity": "no_humans", "platform_type": "custom", "sensors": ["gps", "imu", "sonar"], "comms": ["wifi", "satellite"], "actuators": ["brushless_motor"]},
        "space": {"locomotion": "space.orbital", "power": {"min": 10, "max": 500, "typical": 100}, "env": ["space_orbital"], "proximity": "no_humans", "platform_type": "custom", "sensors": ["imu", "star_tracker", "lidar"], "comms": ["radio", "satellite"], "actuators": ["reaction_wheel", "thruster"]},
        "surveillance": {"locomotion": "stationary.fixed_mount", "power": {"min": 5, "max": 100, "typical": 30}, "env": ["indoor_structured", "outdoor_urban"], "proximity": "no_humans", "platform_type": "fixed_camera", "sensors": ["camera"], "comms": ["ethernet", "wifi"], "actuators": []},
        "edge_vision": {"locomotion": "stationary.fixed_mount", "power": {"min": 5, "max": 100, "typical": 30}, "env": ["indoor_structured", "outdoor_urban"], "proximity": "no_humans", "platform_type": "fixed_camera", "sensors": ["camera"], "comms": ["ethernet", "wifi"], "actuators": []},
        "wearable": {"locomotion": "wearable.body_worn", "power": {"min": 0.01, "max": 5, "typical": 0.5}, "env": ["indoor_unstructured", "outdoor_urban"], "proximity": "humans_on_body", "platform_type": "handheld", "sensors": ["imu", "ppg"], "comms": ["bluetooth", "wifi"], "actuators": []},
        "consumer": {"locomotion": "ground.wheeled", "power": {"min": 10, "max": 100, "typical": 30}, "env": ["indoor_unstructured"], "proximity": "humans_nearby", "platform_type": "amr", "sensors": ["lidar", "camera", "imu"], "comms": ["wifi", "bluetooth"], "actuators": ["brushless_motor"]},
        "agriculture": {"locomotion": "ground.wheeled", "power": {"min": 50, "max": 5000, "typical": 500}, "env": ["outdoor_rural"], "proximity": "no_humans", "platform_type": "custom", "sensors": ["gps", "imu", "camera"], "comms": ["wifi", "lora"], "actuators": ["brushless_motor"]},
        "construction": {"locomotion": "stationary.gantry", "power": {"min": 500, "max": 50000, "typical": 5000}, "env": ["outdoor_urban"], "proximity": "humans_distant", "platform_type": "custom", "sensors": ["gps", "lidar", "imu"], "comms": ["wifi", "radio"], "actuators": ["hydraulic"]},
        "logistics": {"locomotion": "ground.wheeled", "power": {"min": 50, "max": 5000, "typical": 500}, "env": ["indoor_structured"], "proximity": "humans_nearby", "platform_type": "amr", "sensors": ["lidar", "camera", "imu"], "comms": ["wifi", "ethernet"], "actuators": ["brushless_motor"]},
        "energy": {"locomotion": "ground.wheeled", "power": {"min": 50, "max": 500, "typical": 100}, "env": ["outdoor_extreme"], "proximity": "no_humans", "platform_type": "custom", "sensors": ["thermal_camera", "lidar"], "comms": ["wifi", "lte"], "actuators": ["brushless_motor"]},
        "entertainment": {"locomotion": "ground.wheeled", "power": {"min": 20, "max": 200, "typical": 50}, "env": ["indoor_structured"], "proximity": "humans_nearby", "platform_type": "custom", "sensors": ["camera", "microphone"], "comms": ["wifi", "bluetooth"], "actuators": ["servo_motor"]},
        "military": {"locomotion": "aerial.rotary_wing", "power": {"min": 10, "max": 5000, "typical": 500}, "env": ["outdoor_extreme"], "proximity": "no_humans", "platform_type": "drone", "sensors": ["imu", "gps", "eo_ir"], "comms": ["radio", "mesh"], "actuators": ["brushless_motor"]},
        "healthcare": {"locomotion": "ground.wheeled", "power": {"min": 50, "max": 500, "typical": 100}, "env": ["indoor_structured"], "proximity": "humans_nearby", "platform_type": "custom", "sensors": ["lidar", "camera"], "comms": ["wifi", "ethernet"], "actuators": ["brushless_motor"]},
        "assistive": {"locomotion": "wearable.limb_attached", "power": {"min": 1, "max": 100, "typical": 20}, "env": ["indoor_unstructured", "outdoor_urban"], "proximity": "humans_on_body", "platform_type": "custom", "sensors": ["imu", "emg"], "comms": ["bluetooth", "wifi"], "actuators": ["servo_motor"]},
        "waste": {"locomotion": "stationary.fixed_mount", "power": {"min": 200, "max": 2000, "typical": 500}, "env": ["indoor_structured"], "proximity": "humans_distant", "platform_type": "custom", "sensors": ["camera", "hyperspectral"], "comms": ["ethernet"], "actuators": ["servo_motor"]},
        "food": {"locomotion": "stationary.fixed_mount", "power": {"min": 200, "max": 2000, "typical": 500}, "env": ["food_processing"], "proximity": "humans_nearby", "platform_type": "custom", "sensors": ["camera", "hyperspectral"], "comms": ["ethernet"], "actuators": ["servo_motor"]},
        "underground": {"locomotion": "ground.wheeled", "power": {"min": 50, "max": 500, "typical": 100}, "env": ["subterranean"], "proximity": "no_humans", "platform_type": "custom", "sensors": ["lidar", "camera", "gas_detector"], "comms": ["wifi", "tethered"], "actuators": ["brushless_motor"]},
        "forestry": {"locomotion": "ground.wheeled", "power": {"min": 100, "max": 10000, "typical": 1000}, "env": ["outdoor_rural"], "proximity": "no_humans", "platform_type": "custom", "sensors": ["gps", "lidar", "camera"], "comms": ["lte", "satellite"], "actuators": ["hydraulic"]},
        "weather": {"locomotion": "ground.wheeled", "power": {"min": 500, "max": 20000, "typical": 5000}, "env": ["outdoor_extreme"], "proximity": "no_humans", "platform_type": "custom", "sensors": ["gps", "lidar", "camera"], "comms": ["lte", "wifi"], "actuators": ["hydraulic"]},
        "datacenter": {"locomotion": "ground.wheeled", "power": {"min": 50, "max": 200, "typical": 100}, "env": ["indoor_structured"], "proximity": "no_humans", "platform_type": "amr", "sensors": ["lidar", "thermal_camera"], "comms": ["wifi", "ethernet"], "actuators": ["brushless_motor"]},
        "mining": {"locomotion": "ground.tracked", "power": {"min": 500, "max": 50000, "typical": 5000}, "env": ["subterranean"], "proximity": "no_humans", "platform_type": "custom", "sensors": ["lidar", "gas_detector", "camera"], "comms": ["wifi", "mesh"], "actuators": ["hydraulic"]},
        "oil_gas": {"locomotion": "ground.wheeled", "power": {"min": 50, "max": 500, "typical": 100}, "env": ["hazardous_chemical"], "proximity": "no_humans", "platform_type": "custom", "sensors": ["gas_detector", "thermal_camera", "ultrasonic"], "comms": ["wifi", "radio"], "actuators": ["brushless_motor"]},
        "aerospace": {"locomotion": "stationary.gantry", "power": {"min": 1000, "max": 10000, "typical": 3000}, "env": ["indoor_structured"], "proximity": "humans_distant", "platform_type": "custom", "sensors": ["force_torque", "laser_tracker"], "comms": ["ethernet", "ethercat"], "actuators": ["servo_motor"]},
        "textile": {"locomotion": "stationary.fixed_mount", "power": {"min": 100, "max": 1000, "typical": 300}, "env": ["indoor_structured"], "proximity": "humans_nearby", "platform_type": "custom", "sensors": ["camera", "force_sensor"], "comms": ["ethernet"], "actuators": ["servo_motor"]},
        "municipal": {"locomotion": "ground.wheeled", "power": {"min": 500, "max": 20000, "typical": 5000}, "env": ["outdoor_urban"], "proximity": "humans_distant", "platform_type": "custom", "sensors": ["lidar", "camera", "gps"], "comms": ["lte", "wifi"], "actuators": ["brushless_motor", "hydraulic"]},
        "telecom": {"locomotion": "stationary.fixed_mount", "power": {"min": 50, "max": 500, "typical": 100}, "env": ["outdoor_extreme"], "proximity": "no_humans", "platform_type": "custom", "sensors": ["camera", "rf_analyzer"], "comms": ["lte", "wifi"], "actuators": ["climbing_mechanism"]},
    }

    defaults = CATEGORY_DEFAULTS.get(category, CATEGORY_DEFAULTS["surveillance"])
    power = defaults["power"]

    # Generate keywords from the name
    words = name_slug.replace("_", " ").split()
    keywords = {
        "identity": [name_slug.replace("_", " "), human_name.lower(), f"{category} {name_slug.replace('_', ' ')}"],
        "descriptions": [f"{human_name.lower()} system", f"autonomous {name_slug.replace('_', ' ')}"],
        "application": [name_slug.replace("_", " "), f"{category} automation"],
        "industry": [category, name_slug.replace("_", " ")],
        "components": defaults.get("sensors", ["camera"])[:3],
        "related": [category, words[0] if words else category],
    }

    return {
        "id": platform_id,
        "name": human_name,
        "category": category,
        "description": f"{human_name} platform for {category.replace('_', ' ')} applications.",
        "aliases": [name_slug.replace("_", " "), human_name],
        "keywords": keywords,
        "classification": {
            "locomotion": defaults["locomotion"],
            "manipulation": "none",
            "load_class": "none",
            "environment": defaults["env"],
            "human_proximity": defaults["proximity"],
        },
        "attributes": {
            "power_watts": power,
            "compute_tops": {"min": 1, "max": 100, "typical": 10},
            "latency_ms": {"min": 10, "max": 500, "typical": 50},
            "weight_kg": {"min": 1, "max": 1000, "typical": 50},
            "cost_usd": {"min": 1000, "max": 500000, "typical": 50000},
            "autonomy_level": "conditional",
            "safety_level": "basic",
            "mission_duration_hours": {"min": 1, "max": 24, "typical": 8},
        },
        "implications": {
            "platform_type": defaults["platform_type"],
            "perception": {"camera_types": ["monocular"], "detection_classes": [], "max_latency_ms": 50, "min_fps": 15},
            "sensors": {"modalities": defaults["sensors"]},
            "actuators": {"actuator_types": defaults["actuators"], "dof": None, "control_rate_hz": None},
            "comms": {"protocols": defaults["comms"]},
            "power": {"compute_power_watts": power["typical"] * 0.3},
            "safety": {"level": "basic", "certifications": []},
        },
        "context": {
            "typical_architecture": f"Standard {category.replace('_', ' ')} architecture for {name_slug.replace('_', ' ')} applications.",
            "common_pitfalls": [f"Domain-specific challenges for {name_slug.replace('_', ' ')} platforms"],
            "regulatory": [],
            "reference_designs": [],
            "design_considerations": f"Design considerations specific to {name_slug.replace('_', ' ')} in the {category.replace('_', ' ')} domain.",
        },
        "qualification": {
            "suggested_questions": ["power_budget", "latency_requirement", "deployment_environment"],
            "critical_constraints": ["Power budget", "Perception latency"],
            "optional_constraints": [],
        },
    }


def generate_platform_file(platform_id: str, output_dir: Path, dry_run: bool = False) -> bool:
    """Generate a single platform YAML file."""
    category, name_slug = platform_id.split(".", 1)

    # Use detailed data if available, otherwise generate minimal
    if platform_id in PLATFORMS:
        data = PLATFORMS[platform_id]
    else:
        data = _generate_minimal_platform(platform_id)

    # Determine output path
    cat_dir = output_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    filepath = cat_dir / f"{name_slug}.yaml"

    if filepath.exists():
        return False  # Already exists

    if dry_run:
        print(f"  Would create: {filepath}")
        return True

    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)

    return True


def main():
    dry_run = "--dry-run" in sys.argv
    output_dir = Path(__file__).resolve().parents[1] / "data" / "platforms"

    # Read needed platforms
    needed_file = Path("/tmp/real_needed.txt")
    if not needed_file.exists():
        print("ERROR: /tmp/real_needed.txt not found. Run the platform diff command first.")
        sys.exit(1)

    platform_ids = [line.strip() for line in needed_file.read_text().splitlines() if line.strip()]

    print(f"Generating {len(platform_ids)} platform files...")
    created = 0
    skipped = 0

    for pid in platform_ids:
        if "." not in pid:
            continue
        if generate_platform_file(pid, output_dir, dry_run):
            created += 1
        else:
            skipped += 1

    print(f"Done: {created} created, {skipped} skipped (already exist)")
    total = len(list(output_dir.rglob("*.yaml"))) - 2  # Exclude schema.yaml and taxonomy.yaml
    print(f"Total platform files: {total}")


if __name__ == "__main__":
    main()
