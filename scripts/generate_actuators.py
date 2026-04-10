#!/usr/bin/env python3
"""Generate actuator YAML definition files for the actuator registry.

Creates 80+ YAML files across 8 categories under data/actuators/<category>/<type>.yaml.
Each file follows the schema defined in data/actuators/schema.yaml.

Usage:
    .venv/bin/python scripts/generate_actuators.py
"""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ACTUATOR_DIR = ROOT / "data" / "actuators"


# ---------------------------------------------------------------------------
# Actuator definitions — every entry becomes one YAML file.
# ---------------------------------------------------------------------------

ACTUATORS = [
    # ======================================================================
    # MOTOR category
    # ======================================================================
    {
        "id": "motor.brushless_dc",
        "name": "Brushless DC Motor",
        "category": "motor",
        "actuator_type": "brushless_dc",
        "description": (
            "Electronically commutated motor with permanent magnets on the rotor "
            "and stator windings driven by an ESC. High efficiency, long lifespan, "
            "and excellent power-to-weight ratio make it the dominant choice for "
            "drone propulsion, robot joints, and precision drives."
        ),
        "aliases": ["BLDC motor", "EC motor", "electronically commutated motor"],
        "keywords": {
            "identity": ["brushless dc motor", "BLDC", "EC motor", "outrunner", "inrunner"],
            "descriptions": [
                "electric motor", "rotary actuator", "high efficiency motor",
                "permanent magnet motor", "sensorless motor",
            ],
            "application": [
                "drone propulsion", "robot joint", "CNC spindle", "conveyor drive",
                "gimbal motor", "electric vehicle", "fan drive",
            ],
            "industry": ["robotics", "aerospace", "automotive", "manufacturing", "consumer electronics"],
            "components": ["permanent magnet", "hall sensor", "encoder", "esc", "stator winding"],
            "related": ["field-oriented control", "back-emf", "sinusoidal commutation", "trapezoidal drive"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 0.01, "max": 50.0, "typical": 0.5},
            "speed_rpm": {"min": 0, "max": 100000, "typical": 5000},
            "efficiency_percent": {"min": 80, "max": 95, "typical": 90},
            "peak_power_watts": {"min": 5, "max": 5000, "typical": 200},
            "continuous_power_watts": {"min": 2, "max": 3000, "typical": 100},
            "weight_grams": {"min": 10, "max": 5000, "typical": 150},
            "cost_usd": {"min": 10, "max": 2000, "typical": 80},
            "response_time_ms": {"min": 0.1, "max": 5, "typical": 1},
            "lifetime_hours": {"min": 10000, "max": 100000, "typical": 30000},
        },
        "interface": {
            "protocol": ["pwm", "can", "step_dir"],
            "voltage_v": {"min": 7, "max": 48, "typical": 24},
            "current_a": {"min": 0.5, "max": 80, "typical": 10},
            "feedback": ["hall_effect", "encoder_incremental", "back_emf"],
        },
        "reference_products": [
            {
                "id": "tmotor_mn5212",
                "name": "T-Motor MN5212",
                "vendor": "T-Motor",
                "cost_usd": 90,
                "torque_nm": 2.5,
                "speed_rpm": 340,
                "protocol": ["pwm"],
                "feedback": ["back_emf"],
            },
            {
                "id": "hobbywing_x9",
                "name": "Hobbywing X9 Plus",
                "vendor": "Hobbywing",
                "cost_usd": 180,
                "torque_nm": 5.0,
                "speed_rpm": 170,
                "protocol": ["pwm", "can"],
                "feedback": ["back_emf"],
            },
            {
                "id": "cyber_gear_m8",
                "name": "Xiaomi Cyber Gear M8",
                "vendor": "Xiaomi",
                "cost_usd": 120,
                "torque_nm": 12.0,
                "speed_rpm": 320,
                "protocol": ["can"],
                "feedback": ["encoder_absolute"],
            },
        ],
    },
    {
        "id": "motor.brushless_dc_micro",
        "name": "Micro Brushless DC Motor",
        "category": "motor",
        "actuator_type": "brushless_dc",
        "description": (
            "Miniature BLDC motor optimized for small drones, micro-robots, and "
            "handheld devices. Sub-50 gram weight with diameters under 30mm."
        ),
        "aliases": ["micro BLDC", "miniature brushless motor"],
        "keywords": {
            "identity": ["micro brushless motor", "miniature BLDC", "coreless BLDC"],
            "descriptions": [
                "small electric motor", "lightweight motor", "compact rotary actuator",
            ],
            "application": [
                "micro drone", "nano UAV", "micro robot", "handheld gimbal",
                "dental handpiece", "micro pump drive",
            ],
            "industry": ["robotics", "medical devices", "consumer electronics", "aerospace"],
            "components": ["neodymium magnet", "coreless winding", "micro encoder", "esc"],
            "related": ["sensorless control", "high kv motor", "micro propulsion"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 0.001, "max": 0.05, "typical": 0.01},
            "speed_rpm": {"min": 0, "max": 80000, "typical": 20000},
            "efficiency_percent": {"min": 70, "max": 90, "typical": 82},
            "peak_power_watts": {"min": 0.5, "max": 50, "typical": 10},
            "continuous_power_watts": {"min": 0.2, "max": 30, "typical": 5},
            "weight_grams": {"min": 1, "max": 50, "typical": 10},
            "cost_usd": {"min": 3, "max": 100, "typical": 20},
            "lifetime_hours": {"min": 5000, "max": 50000, "typical": 15000},
        },
        "interface": {
            "protocol": ["pwm"],
            "voltage_v": {"min": 3.3, "max": 12, "typical": 7.4},
            "current_a": {"min": 0.1, "max": 10, "typical": 2},
            "feedback": ["back_emf"],
        },
        "reference_products": [
            {
                "id": "faulhaber_0824",
                "name": "Faulhaber 0824 B",
                "vendor": "Faulhaber",
                "cost_usd": 45,
                "torque_nm": 0.004,
                "speed_rpm": 50000,
            },
        ],
    },
    {
        "id": "motor.brushed_dc",
        "name": "Brushed DC Motor",
        "category": "motor",
        "actuator_type": "brushed_dc",
        "description": (
            "Traditional commutated motor with carbon brushes and a wound rotor. "
            "Simple to drive (just apply voltage), low cost, and widely available. "
            "Suitable for toys, low-cost robots, and applications where brush wear "
            "is acceptable."
        ),
        "aliases": ["DC motor", "brushed motor", "permanent magnet DC motor"],
        "keywords": {
            "identity": ["brushed dc motor", "DC motor", "PMDC motor"],
            "descriptions": [
                "commutated motor", "carbon brush motor", "simple electric motor",
                "reversible motor",
            ],
            "application": [
                "toy drive", "window actuator", "seat adjuster", "hobby robot",
                "pump motor", "small conveyor",
            ],
            "industry": ["consumer products", "automotive", "robotics", "education"],
            "components": ["carbon brush", "commutator", "permanent magnet", "armature winding"],
            "related": ["h-bridge", "pwm speed control", "back-emf sensing"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 0.001, "max": 5.0, "typical": 0.1},
            "speed_rpm": {"min": 0, "max": 30000, "typical": 3000},
            "efficiency_percent": {"min": 50, "max": 80, "typical": 65},
            "peak_power_watts": {"min": 0.5, "max": 500, "typical": 20},
            "continuous_power_watts": {"min": 0.2, "max": 300, "typical": 10},
            "weight_grams": {"min": 5, "max": 3000, "typical": 100},
            "cost_usd": {"min": 0.5, "max": 100, "typical": 5},
            "lifetime_hours": {"min": 500, "max": 10000, "typical": 3000},
        },
        "interface": {
            "protocol": ["pwm", "analog"],
            "voltage_v": {"min": 1.5, "max": 48, "typical": 12},
            "current_a": {"min": 0.05, "max": 20, "typical": 2},
            "feedback": ["back_emf", "encoder_incremental"],
        },
        "reference_products": [
            {
                "id": "mabuchi_rs_775",
                "name": "Mabuchi RS-775",
                "vendor": "Mabuchi",
                "cost_usd": 8,
                "torque_nm": 0.25,
                "speed_rpm": 7400,
            },
            {
                "id": "pololu_25d_hp",
                "name": "Pololu 25D HP 12V",
                "vendor": "Pololu",
                "cost_usd": 25,
                "torque_nm": 0.3,
                "speed_rpm": 560,
            },
        ],
    },
    {
        "id": "motor.brushed_dc_gearmotor",
        "name": "Brushed DC Gearmotor",
        "category": "motor",
        "actuator_type": "brushed_dc",
        "description": (
            "Brushed DC motor with integrated gear reduction for high torque at low speed. "
            "Common in mobile robotics, door locks, and positioning systems."
        ),
        "aliases": ["gearmotor", "geared DC motor", "gear motor"],
        "keywords": {
            "identity": ["brushed dc gearmotor", "geared motor", "gearmotor"],
            "descriptions": [
                "gear-reduced motor", "high torque low speed motor", "integrated gearbox motor",
            ],
            "application": [
                "mobile robot drive", "door lock", "window lift", "antenna positioner",
                "conveyor belt", "turntable",
            ],
            "industry": ["robotics", "automotive", "building automation", "consumer products"],
            "components": ["planetary gearbox", "spur gear", "brushed motor", "encoder wheel"],
            "related": ["gear ratio", "backlash", "torque multiplication"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "torque_nm": {"min": 0.05, "max": 20.0, "typical": 2.0},
            "speed_rpm": {"min": 0, "max": 500, "typical": 100},
            "efficiency_percent": {"min": 40, "max": 70, "typical": 55},
            "peak_power_watts": {"min": 1, "max": 200, "typical": 15},
            "continuous_power_watts": {"min": 0.5, "max": 100, "typical": 8},
            "weight_grams": {"min": 20, "max": 2000, "typical": 200},
            "cost_usd": {"min": 3, "max": 150, "typical": 20},
            "lifetime_hours": {"min": 1000, "max": 10000, "typical": 5000},
        },
        "interface": {
            "protocol": ["pwm", "analog"],
            "voltage_v": {"min": 3, "max": 24, "typical": 12},
            "current_a": {"min": 0.1, "max": 10, "typical": 1.5},
            "feedback": ["encoder_incremental"],
        },
        "reference_products": [
            {
                "id": "pololu_37d_70_1",
                "name": "Pololu 37D 70:1 Gearmotor",
                "vendor": "Pololu",
                "cost_usd": 30,
                "torque_nm": 5.5,
                "speed_rpm": 150,
            },
        ],
    },
    {
        "id": "motor.stepper",
        "name": "Stepper Motor",
        "category": "motor",
        "actuator_type": "stepper",
        "description": (
            "Brushless motor that divides a full rotation into discrete steps, "
            "enabling precise open-loop position control. Widely used in 3D printers, "
            "CNC machines, pick-and-place equipment, and camera platforms."
        ),
        "aliases": ["step motor", "stepping motor"],
        "keywords": {
            "identity": ["stepper motor", "step motor", "stepping motor", "NEMA motor"],
            "descriptions": [
                "discrete step motor", "open-loop position motor", "hybrid stepper",
                "bipolar stepper", "unipolar stepper",
            ],
            "application": [
                "3D printer", "CNC axis", "pick-and-place", "camera slider",
                "telescope mount", "lab automation",
            ],
            "industry": ["manufacturing", "3D printing", "laboratory", "astronomy", "robotics"],
            "components": ["coil winding", "rotor teeth", "stepper driver", "microstepping controller"],
            "related": ["microstepping", "holding torque", "detent torque", "step angle"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "position",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "torque_nm": {"min": 0.01, "max": 15.0, "typical": 0.5},
            "speed_rpm": {"min": 0, "max": 3000, "typical": 300},
            "position_accuracy_um": {"min": 5, "max": 200, "typical": 50},
            "peak_power_watts": {"min": 2, "max": 200, "typical": 30},
            "continuous_power_watts": {"min": 1, "max": 100, "typical": 15},
            "weight_grams": {"min": 50, "max": 5000, "typical": 350},
            "cost_usd": {"min": 5, "max": 200, "typical": 15},
            "lifetime_hours": {"min": 5000, "max": 50000, "typical": 20000},
        },
        "interface": {
            "protocol": ["step_dir", "spi"],
            "voltage_v": {"min": 5, "max": 48, "typical": 24},
            "current_a": {"min": 0.2, "max": 6, "typical": 1.7},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "nema17_42byghw804",
                "name": "NEMA 17 (42BYGHW804)",
                "vendor": "StepperOnline",
                "cost_usd": 10,
                "torque_nm": 0.48,
                "speed_rpm": 600,
            },
            {
                "id": "nema23_23hs45",
                "name": "NEMA 23 (23HS45-4204S)",
                "vendor": "StepperOnline",
                "cost_usd": 25,
                "torque_nm": 3.0,
                "speed_rpm": 400,
            },
            {
                "id": "trinamic_tmc2209",
                "name": "Trinamic TMC2209 (driver)",
                "vendor": "Trinamic",
                "cost_usd": 8,
            },
        ],
    },
    {
        "id": "motor.stepper_closed_loop",
        "name": "Closed-Loop Stepper Motor",
        "category": "motor",
        "actuator_type": "stepper",
        "description": (
            "Stepper motor with integrated encoder and closed-loop driver. "
            "Eliminates missed steps, reduces heat, and enables torque control. "
            "Bridges the gap between open-loop steppers and servo motors."
        ),
        "aliases": ["servo stepper", "closed loop stepper", "hybrid servo"],
        "keywords": {
            "identity": ["closed-loop stepper", "servo stepper", "hybrid servo motor"],
            "descriptions": [
                "encoder-equipped stepper", "feedback stepper motor", "anti-stall stepper",
            ],
            "application": [
                "CNC machine", "3D printer", "packaging machine", "textile machine",
                "precision stage",
            ],
            "industry": ["manufacturing", "3D printing", "packaging", "textile"],
            "components": [
                "stepper motor", "magnetic encoder", "closed-loop driver", "current controller",
            ],
            "related": ["stall detection", "torque mode", "position recovery"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "position",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "torque_nm": {"min": 0.2, "max": 12.0, "typical": 2.0},
            "speed_rpm": {"min": 0, "max": 3000, "typical": 600},
            "position_accuracy_um": {"min": 1, "max": 50, "typical": 10},
            "peak_power_watts": {"min": 10, "max": 300, "typical": 60},
            "continuous_power_watts": {"min": 5, "max": 150, "typical": 30},
            "weight_grams": {"min": 200, "max": 5000, "typical": 600},
            "cost_usd": {"min": 30, "max": 400, "typical": 80},
            "lifetime_hours": {"min": 10000, "max": 50000, "typical": 25000},
        },
        "interface": {
            "protocol": ["step_dir", "can", "rs485"],
            "voltage_v": {"min": 12, "max": 48, "typical": 24},
            "current_a": {"min": 0.5, "max": 6, "typical": 2.5},
            "feedback": ["encoder_incremental", "encoder_absolute"],
        },
        "reference_products": [
            {
                "id": "isc04_nema17",
                "name": "Moons iSC04 NEMA 17",
                "vendor": "Moons",
                "cost_usd": 65,
                "torque_nm": 0.7,
            },
        ],
    },
    {
        "id": "motor.servo",
        "name": "Servo Motor",
        "category": "motor",
        "actuator_type": "servo",
        "description": (
            "Closed-loop motor with integrated controller, encoder, and gearbox "
            "for precise position/velocity/torque control. The standard actuator "
            "for industrial robots, cobots, and high-performance articulated systems."
        ),
        "aliases": ["servo drive", "AC servo", "servo actuator"],
        "keywords": {
            "identity": ["servo motor", "servo drive", "AC servo", "PMSM servo"],
            "descriptions": [
                "closed-loop actuator", "position-controlled motor", "torque-controlled motor",
                "precision actuator",
            ],
            "application": [
                "robot arm joint", "cobot joint", "CNC axis", "semiconductor handling",
                "packaging machine", "pick-and-place",
            ],
            "industry": ["robotics", "manufacturing", "semiconductor", "packaging", "aerospace"],
            "components": [
                "PMSM motor", "absolute encoder", "servo driver", "brake", "harmonic gear",
            ],
            "related": [
                "field-oriented control", "cascaded PID", "trajectory interpolation",
                "EtherCAT motion",
            ],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "position",
            "environment": ["indoor", "cleanroom"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "torque_nm": {"min": 0.05, "max": 500, "typical": 5.0},
            "speed_rpm": {"min": 0, "max": 10000, "typical": 3000},
            "position_accuracy_um": {"min": 1, "max": 100, "typical": 10},
            "repeatability_um": {"min": 0.5, "max": 50, "typical": 5},
            "control_rate_hz": {"min": 1000, "max": 20000, "typical": 4000},
            "efficiency_percent": {"min": 85, "max": 97, "typical": 92},
            "peak_power_watts": {"min": 50, "max": 10000, "typical": 750},
            "continuous_power_watts": {"min": 20, "max": 5000, "typical": 400},
            "weight_grams": {"min": 100, "max": 30000, "typical": 2000},
            "cost_usd": {"min": 100, "max": 10000, "typical": 500},
            "response_time_ms": {"min": 0.1, "max": 2, "typical": 0.5},
            "lifetime_hours": {"min": 20000, "max": 100000, "typical": 40000},
        },
        "interface": {
            "protocol": ["ethercat", "can", "profinet", "rs485"],
            "voltage_v": {"min": 24, "max": 400, "typical": 48},
            "current_a": {"min": 1, "max": 100, "typical": 10},
            "feedback": ["encoder_absolute", "resolver"],
        },
        "reference_products": [
            {
                "id": "maxon_ec_i_40",
                "name": "maxon EC-i 40",
                "vendor": "maxon",
                "cost_usd": 350,
                "torque_nm": 0.128,
                "speed_rpm": 15000,
                "protocol": ["pwm"],
                "feedback": ["hall_effect", "encoder_incremental"],
            },
            {
                "id": "faulhaber_2237",
                "name": "Faulhaber 2237 CXR",
                "vendor": "Faulhaber",
                "cost_usd": 420,
                "torque_nm": 0.012,
                "speed_rpm": 11000,
                "protocol": ["pwm"],
                "feedback": ["encoder_incremental"],
            },
            {
                "id": "dynamixel_xm430",
                "name": "ROBOTIS Dynamixel XM430-W350",
                "vendor": "ROBOTIS",
                "cost_usd": 260,
                "torque_nm": 4.1,
                "speed_rpm": 46,
                "protocol": ["rs485", "ttl"],
                "feedback": ["encoder_absolute"],
            },
            {
                "id": "robstride_03",
                "name": "Robstride 03 Mini",
                "vendor": "Robstride",
                "cost_usd": 180,
                "torque_nm": 12.0,
                "speed_rpm": 320,
                "protocol": ["can"],
                "feedback": ["encoder_absolute"],
            },
        ],
    },
    {
        "id": "motor.servo_miniature",
        "name": "Miniature Servo Motor",
        "category": "motor",
        "actuator_type": "servo",
        "description": (
            "Compact hobby-class servo with integrated gear train and position feedback. "
            "Low cost and easy to drive via PWM signal. Used in RC vehicles, small robots, "
            "and pan-tilt mechanisms."
        ),
        "aliases": ["hobby servo", "RC servo", "micro servo"],
        "keywords": {
            "identity": ["miniature servo", "hobby servo", "RC servo", "micro servo"],
            "descriptions": [
                "compact position actuator", "PWM-driven servo", "gear-reduced servo",
            ],
            "application": [
                "RC vehicle steering", "pan-tilt camera", "small robot arm",
                "animatronic", "model aircraft control surface",
            ],
            "industry": ["hobby", "education", "robotics", "entertainment"],
            "components": ["potentiometer", "dc motor", "gear train", "pwm controller"],
            "related": ["servo horn", "microsecond pulse", "analog servo", "digital servo"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "position",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "torque_nm": {"min": 0.05, "max": 3.0, "typical": 0.3},
            "speed_rpm": {"min": 0, "max": 100, "typical": 50},
            "peak_power_watts": {"min": 0.5, "max": 15, "typical": 3},
            "continuous_power_watts": {"min": 0.2, "max": 8, "typical": 1.5},
            "weight_grams": {"min": 3, "max": 80, "typical": 12},
            "cost_usd": {"min": 2, "max": 60, "typical": 8},
            "response_time_ms": {"min": 5, "max": 50, "typical": 15},
            "lifetime_hours": {"min": 1000, "max": 10000, "typical": 3000},
        },
        "interface": {
            "protocol": ["pwm"],
            "voltage_v": {"min": 4.8, "max": 7.4, "typical": 6.0},
            "current_a": {"min": 0.1, "max": 2, "typical": 0.5},
            "feedback": ["potentiometer"],
        },
        "reference_products": [
            {
                "id": "hitec_hs_311",
                "name": "Hitec HS-311",
                "vendor": "Hitec",
                "cost_usd": 10,
                "torque_nm": 0.3,
            },
            {
                "id": "savox_sv1250mg",
                "name": "Savox SV-1250MG",
                "vendor": "Savox",
                "cost_usd": 45,
                "torque_nm": 0.8,
            },
        ],
    },
    {
        "id": "motor.linear_actuator",
        "name": "Electric Linear Actuator",
        "category": "motor",
        "actuator_type": "linear_actuator",
        "description": (
            "Motor-driven linear motion device using a lead screw, ball screw, or belt "
            "drive to convert rotary motion to straight-line force and displacement. "
            "Used in adjustable furniture, industrial automation, and solar tracking."
        ),
        "aliases": ["electric cylinder", "linear drive", "electric ram"],
        "keywords": {
            "identity": ["electric linear actuator", "linear drive", "electric cylinder"],
            "descriptions": [
                "linear motion device", "push-pull actuator", "screw-driven actuator",
                "ball screw actuator",
            ],
            "application": [
                "adjustable desk", "hospital bed", "solar tracker", "gate opener",
                "industrial press", "agricultural equipment",
            ],
            "industry": ["furniture", "medical", "solar energy", "agriculture", "manufacturing"],
            "components": [
                "lead screw", "ball screw", "limit switch", "dc motor", "encoder",
            ],
            "related": ["stroke length", "duty cycle", "self-locking", "dynamic load"],
        },
        "classification": {
            "motion_type": "linear",
            "control_type": "position",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "force_n": {"min": 50, "max": 20000, "typical": 1500},
            "speed_mps": {"min": 0.002, "max": 0.15, "typical": 0.03},
            "stroke_mm": {"min": 25, "max": 1000, "typical": 200},
            "position_accuracy_um": {"min": 50, "max": 5000, "typical": 500},
            "peak_power_watts": {"min": 10, "max": 1000, "typical": 100},
            "continuous_power_watts": {"min": 5, "max": 500, "typical": 50},
            "weight_grams": {"min": 100, "max": 10000, "typical": 1500},
            "cost_usd": {"min": 20, "max": 2000, "typical": 150},
            "lifetime_hours": {"min": 2000, "max": 30000, "typical": 10000},
        },
        "interface": {
            "protocol": ["pwm", "analog", "rs485"],
            "voltage_v": {"min": 12, "max": 48, "typical": 24},
            "current_a": {"min": 0.5, "max": 20, "typical": 4},
            "feedback": ["potentiometer", "encoder_incremental", "hall_effect"],
        },
        "reference_products": [
            {
                "id": "tolomatic_erd",
                "name": "Tolomatic ERD Electric Cylinder",
                "vendor": "Tolomatic",
                "cost_usd": 800,
                "force_n": 8900,
                "stroke_mm": 300,
            },
            {
                "id": "festo_dgea",
                "name": "Festo DGEA Mini Actuator",
                "vendor": "Festo",
                "cost_usd": 350,
                "force_n": 800,
                "stroke_mm": 100,
            },
        ],
    },
    {
        "id": "motor.linear_actuator_precision",
        "name": "Precision Linear Stage",
        "category": "motor",
        "actuator_type": "linear_actuator",
        "description": (
            "High-accuracy linear positioner using ground ball screws or linear motors "
            "with sub-micron feedback. For semiconductor, optics, and metrology."
        ),
        "aliases": ["linear stage", "precision positioner", "linear translation stage"],
        "keywords": {
            "identity": ["precision linear stage", "linear positioner", "translation stage"],
            "descriptions": [
                "sub-micron positioner", "high accuracy stage", "nanopositioning stage",
            ],
            "application": [
                "semiconductor lithography", "optical alignment", "metrology",
                "wafer inspection", "microscopy",
            ],
            "industry": ["semiconductor", "optics", "metrology", "research"],
            "components": [
                "linear motor", "linear encoder", "air bearing", "cross roller guide",
            ],
            "related": [
                "interferometric feedback", "nanometer resolution", "planar motion",
            ],
        },
        "classification": {
            "motion_type": "linear",
            "control_type": "position",
            "environment": ["cleanroom", "indoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 5, "max": 500, "typical": 50},
            "speed_mps": {"min": 0.001, "max": 0.5, "typical": 0.1},
            "stroke_mm": {"min": 5, "max": 500, "typical": 50},
            "position_accuracy_um": {"min": 0.01, "max": 5, "typical": 0.1},
            "repeatability_um": {"min": 0.005, "max": 1, "typical": 0.05},
            "peak_power_watts": {"min": 5, "max": 200, "typical": 40},
            "continuous_power_watts": {"min": 2, "max": 100, "typical": 20},
            "weight_grams": {"min": 200, "max": 10000, "typical": 2000},
            "cost_usd": {"min": 500, "max": 50000, "typical": 5000},
        },
        "interface": {
            "protocol": ["ethercat", "analog"],
            "voltage_v": {"min": 24, "max": 48, "typical": 24},
            "current_a": {"min": 0.5, "max": 10, "typical": 2},
            "feedback": ["encoder_incremental", "encoder_absolute"],
        },
        "reference_products": [
            {
                "id": "aerotech_pro165",
                "name": "Aerotech PRO165LM",
                "vendor": "Aerotech",
                "cost_usd": 8000,
                "stroke_mm": 200,
                "position_accuracy_um": 0.1,
            },
        ],
    },
    {
        "id": "motor.voice_coil",
        "name": "Voice Coil Actuator",
        "category": "motor",
        "actuator_type": "voice_coil",
        "description": (
            "Direct-drive linear actuator using a coil in a magnetic field, "
            "producing force proportional to current with zero cogging and "
            "extremely fast response. Used for autofocus, vibration isolation, "
            "disk drive heads, and precision positioning."
        ),
        "aliases": ["VCA", "voice coil motor", "moving coil actuator"],
        "keywords": {
            "identity": ["voice coil actuator", "VCA", "moving coil actuator"],
            "descriptions": [
                "linear force actuator", "direct drive actuator", "zero cogging actuator",
                "fast response linear actuator",
            ],
            "application": [
                "autofocus mechanism", "optical image stabilization", "vibration isolation",
                "disk drive head", "valve actuation", "active damping",
            ],
            "industry": ["optics", "consumer electronics", "semiconductor", "aerospace"],
            "components": [
                "voice coil", "permanent magnet", "flexure bearing", "linear encoder",
            ],
            "related": [
                "lorentz force", "force constant", "moving mass", "bandwidth",
            ],
        },
        "classification": {
            "motion_type": "linear",
            "control_type": "force",
            "environment": ["indoor", "cleanroom"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 0.01, "max": 200, "typical": 5},
            "speed_mps": {"min": 0.01, "max": 5.0, "typical": 0.5},
            "stroke_mm": {"min": 0.1, "max": 50, "typical": 5},
            "position_accuracy_um": {"min": 0.01, "max": 50, "typical": 1},
            "peak_power_watts": {"min": 0.1, "max": 100, "typical": 10},
            "continuous_power_watts": {"min": 0.05, "max": 50, "typical": 5},
            "weight_grams": {"min": 1, "max": 500, "typical": 30},
            "cost_usd": {"min": 5, "max": 1000, "typical": 50},
            "response_time_ms": {"min": 0.01, "max": 1, "typical": 0.1},
            "lifetime_hours": {"min": 50000, "max": 500000, "typical": 100000},
        },
        "interface": {
            "protocol": ["analog", "pwm"],
            "voltage_v": {"min": 3, "max": 48, "typical": 12},
            "current_a": {"min": 0.01, "max": 10, "typical": 1},
            "feedback": ["encoder_incremental", "hall_effect"],
        },
        "reference_products": [
            {
                "id": "bij_lvcm_019",
                "name": "BEI Kimco LVCM-019-013-01",
                "vendor": "BEI Kimco",
                "cost_usd": 120,
                "force_n": 8.5,
                "stroke_mm": 6.4,
            },
        ],
    },
    {
        "id": "motor.voice_coil_rotary",
        "name": "Rotary Voice Coil Actuator",
        "category": "motor",
        "actuator_type": "voice_coil",
        "description": (
            "Limited-angle rotary actuator based on the voice coil principle. "
            "Provides fast angular positioning over a small range (typically < 60 deg). "
            "Used in scanning mirrors, fast steering mirrors, and valve control."
        ),
        "aliases": ["rotary VCA", "limited angle torque motor", "galvo motor"],
        "keywords": {
            "identity": ["rotary voice coil", "galvanometer motor", "limited angle torquer"],
            "descriptions": [
                "angular positioning actuator", "scanning mirror driver", "fast steering actuator",
            ],
            "application": [
                "laser scanning", "fast steering mirror", "optical switch",
                "lidar scanner", "barcode reader",
            ],
            "industry": ["optics", "photonics", "lidar", "semiconductor"],
            "components": [
                "coil", "permanent magnet", "flexure pivot", "position sensor",
            ],
            "related": ["galvanometer", "scan angle", "angular acceleration"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "position",
            "environment": ["indoor", "cleanroom"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 0.001, "max": 1.0, "typical": 0.05},
            "speed_rpm": {"min": 0, "max": 500, "typical": 100},
            "position_accuracy_um": {"min": 0.1, "max": 50, "typical": 2},
            "peak_power_watts": {"min": 0.5, "max": 50, "typical": 5},
            "continuous_power_watts": {"min": 0.1, "max": 20, "typical": 2},
            "weight_grams": {"min": 5, "max": 500, "typical": 50},
            "cost_usd": {"min": 50, "max": 3000, "typical": 300},
            "response_time_ms": {"min": 0.01, "max": 1, "typical": 0.1},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 5, "max": 48, "typical": 24},
            "current_a": {"min": 0.05, "max": 5, "typical": 0.5},
            "feedback": ["encoder_incremental", "hall_effect"],
        },
        "reference_products": [
            {
                "id": "cambridge_6215h",
                "name": "Cambridge Technology 6215H",
                "vendor": "Cambridge Technology",
                "cost_usd": 250,
                "torque_nm": 0.04,
            },
        ],
    },
    {
        "id": "motor.servo_high_torque",
        "name": "High-Torque Servo Actuator",
        "category": "motor",
        "actuator_type": "servo",
        "description": (
            "Industrial servo motor with harmonic or cycloidal reducer for very high "
            "torque density. Designed for humanoid and heavy-duty robot joints."
        ),
        "aliases": ["robot joint actuator", "harmonic drive servo", "quasi-direct-drive actuator"],
        "keywords": {
            "identity": ["high torque servo", "joint actuator", "quasi-direct drive"],
            "descriptions": [
                "high torque density actuator", "integrated robot joint", "compact joint module",
            ],
            "application": [
                "humanoid robot", "quadruped leg", "exoskeleton", "heavy-duty arm",
                "legged robot",
            ],
            "industry": ["robotics", "prosthetics", "defense", "research"],
            "components": [
                "harmonic drive", "cycloidal reducer", "torque sensor", "BLDC motor",
            ],
            "related": [
                "torque density", "backdrivability", "impedance control", "compliant actuation",
            ],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "torque",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 10, "max": 500, "typical": 80},
            "speed_rpm": {"min": 0, "max": 200, "typical": 30},
            "control_rate_hz": {"min": 1000, "max": 10000, "typical": 2000},
            "efficiency_percent": {"min": 60, "max": 85, "typical": 75},
            "peak_power_watts": {"min": 100, "max": 5000, "typical": 800},
            "continuous_power_watts": {"min": 50, "max": 2000, "typical": 300},
            "weight_grams": {"min": 200, "max": 10000, "typical": 1500},
            "cost_usd": {"min": 200, "max": 15000, "typical": 2000},
        },
        "interface": {
            "protocol": ["can", "ethercat", "rs485"],
            "voltage_v": {"min": 24, "max": 48, "typical": 48},
            "current_a": {"min": 2, "max": 60, "typical": 15},
            "feedback": ["encoder_absolute", "strain_gauge"],
        },
        "reference_products": [
            {
                "id": "unitree_a1",
                "name": "Unitree A1 Motor",
                "vendor": "Unitree",
                "cost_usd": 300,
                "torque_nm": 33.5,
                "speed_rpm": 225,
            },
        ],
    },
    # ======================================================================
    # HYDRAULIC category
    # ======================================================================
    {
        "id": "hydraulic.hydraulic_cylinder",
        "name": "Hydraulic Cylinder",
        "category": "hydraulic",
        "actuator_type": "hydraulic_cylinder",
        "description": (
            "Linear actuator that uses pressurized hydraulic fluid to produce "
            "high force over a defined stroke. Dominant in construction, mining, "
            "and heavy industrial equipment."
        ),
        "aliases": ["hydraulic ram", "hydraulic jack", "hydraulic piston"],
        "keywords": {
            "identity": ["hydraulic cylinder", "hydraulic ram", "hydraulic piston"],
            "descriptions": [
                "fluid power cylinder", "high force linear actuator", "double-acting cylinder",
                "single-acting cylinder",
            ],
            "application": [
                "excavator arm", "press brake", "dump truck", "crane boom",
                "injection molder", "material handling",
            ],
            "industry": ["construction", "mining", "manufacturing", "agriculture", "marine"],
            "components": [
                "piston rod", "cylinder barrel", "seals", "hydraulic oil", "cushion valve",
            ],
            "related": [
                "pascal's law", "cylinder bore", "rod diameter", "working pressure",
            ],
        },
        "classification": {
            "motion_type": "linear",
            "control_type": "force",
            "environment": ["outdoor", "indoor", "hazardous"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 1000, "max": 5000000, "typical": 100000},
            "speed_mps": {"min": 0.01, "max": 1.0, "typical": 0.1},
            "stroke_mm": {"min": 50, "max": 5000, "typical": 500},
            "peak_power_watts": {"min": 500, "max": 500000, "typical": 20000},
            "continuous_power_watts": {"min": 200, "max": 200000, "typical": 10000},
            "weight_grams": {"min": 2000, "max": 500000, "typical": 20000},
            "cost_usd": {"min": 100, "max": 50000, "typical": 2000},
            "lifetime_hours": {"min": 5000, "max": 50000, "typical": 15000},
        },
        "interface": {
            "protocol": ["analog", "can"],
            "voltage_v": {"min": 12, "max": 24, "typical": 24},
            "feedback": ["potentiometer", "encoder_incremental"],
        },
        "reference_products": [
            {
                "id": "parker_hma",
                "name": "Parker HMA Heavy-Duty Cylinder",
                "vendor": "Parker Hannifin",
                "cost_usd": 2500,
                "force_n": 200000,
                "stroke_mm": 600,
            },
        ],
    },
    {
        "id": "hydraulic.hydraulic_cylinder_compact",
        "name": "Compact Hydraulic Cylinder",
        "category": "hydraulic",
        "actuator_type": "hydraulic_cylinder",
        "description": (
            "Short-stroke, high-force hydraulic cylinder for clamping, pressing, and "
            "fixture applications in tight spaces."
        ),
        "aliases": ["compact cylinder", "pancake cylinder", "short stroke hydraulic"],
        "keywords": {
            "identity": ["compact hydraulic cylinder", "pancake cylinder", "block cylinder"],
            "descriptions": [
                "short stroke actuator", "clamping cylinder", "compact force actuator",
            ],
            "application": [
                "fixture clamping", "die pressing", "workholding", "automation clamping",
            ],
            "industry": ["manufacturing", "metalworking", "automotive assembly"],
            "components": ["piston", "seals", "proximity sensor", "cylinder block"],
            "related": ["clamping force", "bore size", "mounting flange"],
        },
        "classification": {
            "motion_type": "linear",
            "control_type": "force",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 500, "max": 500000, "typical": 30000},
            "speed_mps": {"min": 0.005, "max": 0.3, "typical": 0.05},
            "stroke_mm": {"min": 5, "max": 200, "typical": 50},
            "peak_power_watts": {"min": 100, "max": 50000, "typical": 3000},
            "weight_grams": {"min": 500, "max": 50000, "typical": 5000},
            "cost_usd": {"min": 50, "max": 5000, "typical": 500},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 12, "max": 24, "typical": 24},
            "feedback": ["potentiometer", "none"],
        },
        "reference_products": [],
    },
    {
        "id": "hydraulic.hydraulic_motor",
        "name": "Hydraulic Motor",
        "category": "hydraulic",
        "actuator_type": "hydraulic_motor",
        "description": (
            "Rotary actuator driven by pressurized hydraulic fluid. Delivers very "
            "high torque at low speed without gearing. Used in winches, augers, "
            "track drives, and marine propulsion."
        ),
        "aliases": ["hydraulic rotary motor", "orbit motor", "gear motor hydraulic"],
        "keywords": {
            "identity": ["hydraulic motor", "orbit motor", "gerotor motor", "piston motor"],
            "descriptions": [
                "fluid power motor", "high torque rotary actuator", "low speed high torque motor",
            ],
            "application": [
                "winch drive", "auger", "conveyor drive", "track drive",
                "mixer", "marine thruster",
            ],
            "industry": ["construction", "marine", "agriculture", "forestry", "mining"],
            "components": [
                "gerotor set", "axial piston", "radial piston", "hydraulic fluid", "shaft seal",
            ],
            "related": [
                "displacement", "volumetric efficiency", "pressure rating", "drain flow",
            ],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["outdoor", "indoor", "underwater"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 10, "max": 50000, "typical": 500},
            "speed_rpm": {"min": 0, "max": 5000, "typical": 200},
            "efficiency_percent": {"min": 70, "max": 95, "typical": 88},
            "peak_power_watts": {"min": 500, "max": 200000, "typical": 15000},
            "continuous_power_watts": {"min": 200, "max": 100000, "typical": 8000},
            "weight_grams": {"min": 1000, "max": 200000, "typical": 10000},
            "cost_usd": {"min": 200, "max": 30000, "typical": 2000},
            "lifetime_hours": {"min": 5000, "max": 30000, "typical": 12000},
        },
        "interface": {
            "protocol": ["analog", "can"],
            "voltage_v": {"min": 12, "max": 24, "typical": 24},
            "feedback": ["encoder_incremental", "none"],
        },
        "reference_products": [
            {
                "id": "bosch_rexroth_a4vg",
                "name": "Bosch Rexroth A4VG Variable Pump/Motor",
                "vendor": "Bosch Rexroth",
                "cost_usd": 5000,
                "torque_nm": 800,
                "speed_rpm": 3600,
            },
        ],
    },
    {
        "id": "hydraulic.hydraulic_motor_radial",
        "name": "Radial Piston Hydraulic Motor",
        "category": "hydraulic",
        "actuator_type": "hydraulic_motor",
        "description": (
            "High-displacement radial piston motor for ultra-low speed, ultra-high "
            "torque applications such as heavy winches and conveyor drives."
        ),
        "aliases": ["radial piston motor", "cam-ring motor", "low speed high torque motor"],
        "keywords": {
            "identity": ["radial piston motor", "LSHT motor", "cam-ring hydraulic motor"],
            "descriptions": [
                "ultra-high torque motor", "direct drive hydraulic motor",
                "low speed motor",
            ],
            "application": [
                "heavy winch", "conveyor drive", "tunnel boring", "mining crusher",
                "ship capstan",
            ],
            "industry": ["mining", "marine", "construction", "oil and gas"],
            "components": ["radial pistons", "cam ring", "distribution valve", "shaft"],
            "related": ["specific displacement", "starting torque", "mechanical efficiency"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "torque",
            "environment": ["outdoor", "hazardous", "underwater"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 500, "max": 100000, "typical": 10000},
            "speed_rpm": {"min": 0, "max": 200, "typical": 30},
            "efficiency_percent": {"min": 85, "max": 97, "typical": 93},
            "peak_power_watts": {"min": 5000, "max": 500000, "typical": 50000},
            "weight_grams": {"min": 10000, "max": 1000000, "typical": 80000},
            "cost_usd": {"min": 2000, "max": 100000, "typical": 15000},
        },
        "interface": {
            "protocol": ["analog", "can"],
            "voltage_v": {"min": 24, "max": 24, "typical": 24},
            "feedback": ["encoder_incremental"],
        },
        "reference_products": [
            {
                "id": "hagglunds_cbm",
                "name": "Hagglunds CBM Motor",
                "vendor": "Bosch Rexroth",
                "cost_usd": 25000,
                "torque_nm": 50000,
            },
        ],
    },
    {
        "id": "hydraulic.hydraulic_valve",
        "name": "Hydraulic Proportional Valve",
        "category": "hydraulic",
        "actuator_type": "hydraulic_valve",
        "description": (
            "Electronically controlled valve that modulates hydraulic flow and pressure "
            "proportionally to an electrical command signal. Controls the speed, force, "
            "and direction of hydraulic cylinders and motors."
        ),
        "aliases": ["proportional valve", "servo valve", "hydraulic directional valve"],
        "keywords": {
            "identity": ["hydraulic valve", "proportional valve", "servo valve", "directional valve"],
            "descriptions": [
                "flow control valve", "pressure control valve", "spool valve",
                "electrohydraulic valve",
            ],
            "application": [
                "cylinder speed control", "press control", "injection molding",
                "flight control surface", "motion platform",
            ],
            "industry": [
                "manufacturing", "aerospace", "construction", "simulation",
                "oil and gas",
            ],
            "components": ["spool", "solenoid", "LVDT", "pilot stage", "orifice"],
            "related": [
                "flow gain", "pressure gain", "null shift", "hysteresis",
            ],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "proportional",
            "environment": ["indoor", "outdoor", "hazardous"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "response_time_ms": {"min": 2, "max": 50, "typical": 10},
            "peak_power_watts": {"min": 5, "max": 200, "typical": 30},
            "continuous_power_watts": {"min": 2, "max": 100, "typical": 15},
            "weight_grams": {"min": 200, "max": 20000, "typical": 3000},
            "cost_usd": {"min": 100, "max": 20000, "typical": 1500},
            "lifetime_hours": {"min": 10000, "max": 100000, "typical": 30000},
        },
        "interface": {
            "protocol": ["analog", "can", "profinet"],
            "voltage_v": {"min": 12, "max": 24, "typical": 24},
            "current_a": {"min": 0.1, "max": 3, "typical": 0.8},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "moog_d633",
                "name": "Moog D633 Direct Drive Valve",
                "vendor": "Moog",
                "cost_usd": 3000,
                "response_time_ms": 4,
            },
            {
                "id": "bosch_4wre",
                "name": "Bosch Rexroth 4WRE",
                "vendor": "Bosch Rexroth",
                "cost_usd": 800,
                "response_time_ms": 15,
            },
        ],
    },
    {
        "id": "hydraulic.hydraulic_valve_on_off",
        "name": "Hydraulic Directional On/Off Valve",
        "category": "hydraulic",
        "actuator_type": "hydraulic_valve",
        "description": (
            "Solenoid-operated directional control valve that switches hydraulic flow "
            "between on/off states. Simple, robust, and inexpensive."
        ),
        "aliases": ["solenoid valve", "directional control valve", "DCV"],
        "keywords": {
            "identity": ["on-off hydraulic valve", "solenoid hydraulic valve", "DCV"],
            "descriptions": [
                "switching valve", "directional control valve", "bang-bang valve",
            ],
            "application": [
                "cylinder extend/retract", "motor direction", "circuit isolation",
                "safety shutoff",
            ],
            "industry": ["construction", "agriculture", "manufacturing", "mobile hydraulic"],
            "components": ["solenoid", "spool", "spring", "o-ring seals"],
            "related": ["switching time", "flow capacity", "pressure drop"],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "on_off",
            "environment": ["indoor", "outdoor", "hazardous"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "response_time_ms": {"min": 10, "max": 200, "typical": 40},
            "peak_power_watts": {"min": 5, "max": 50, "typical": 20},
            "weight_grams": {"min": 100, "max": 5000, "typical": 800},
            "cost_usd": {"min": 20, "max": 2000, "typical": 200},
            "lifetime_hours": {"min": 20000, "max": 200000, "typical": 50000},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 12, "max": 24, "typical": 24},
            "current_a": {"min": 0.2, "max": 3, "typical": 0.8},
            "feedback": ["none"],
        },
        "reference_products": [],
    },
    # ======================================================================
    # PNEUMATIC category
    # ======================================================================
    {
        "id": "pneumatic.pneumatic_cylinder",
        "name": "Pneumatic Cylinder",
        "category": "pneumatic",
        "actuator_type": "pneumatic_cylinder",
        "description": (
            "Linear actuator powered by compressed air. Fast, clean, and lightweight. "
            "Ideal for pick-and-place, clamping, sorting, and packaging lines where "
            "speed matters more than precise position control."
        ),
        "aliases": ["air cylinder", "pneumatic ram", "pneumatic piston"],
        "keywords": {
            "identity": ["pneumatic cylinder", "air cylinder", "pneumatic actuator"],
            "descriptions": [
                "compressed air actuator", "fast linear actuator", "clean actuator",
                "double-acting pneumatic",
            ],
            "application": [
                "pick-and-place", "clamping", "sorting", "packaging",
                "material handling", "part ejection",
            ],
            "industry": [
                "manufacturing", "packaging", "food processing", "automotive assembly",
                "pharmaceutical",
            ],
            "components": [
                "piston", "barrel", "air seals", "cushion", "magnetic ring",
            ],
            "related": [
                "bore size", "stroke length", "air consumption", "cylinder force",
            ],
        },
        "classification": {
            "motion_type": "linear",
            "control_type": "on_off",
            "environment": ["indoor", "cleanroom"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 10, "max": 50000, "typical": 500},
            "speed_mps": {"min": 0.05, "max": 3.0, "typical": 0.5},
            "stroke_mm": {"min": 5, "max": 2000, "typical": 100},
            "peak_power_watts": {"min": 5, "max": 2000, "typical": 100},
            "weight_grams": {"min": 20, "max": 10000, "typical": 300},
            "cost_usd": {"min": 10, "max": 1000, "typical": 60},
            "response_time_ms": {"min": 10, "max": 200, "typical": 30},
            "lifetime_hours": {"min": 10000, "max": 100000, "typical": 30000},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 12, "max": 24, "typical": 24},
            "feedback": ["hall_effect", "none"],
        },
        "reference_products": [
            {
                "id": "festo_dsbc",
                "name": "Festo DSBC Standard Cylinder",
                "vendor": "Festo",
                "cost_usd": 80,
                "force_n": 1250,
                "stroke_mm": 200,
            },
            {
                "id": "smc_cq2",
                "name": "SMC CQ2 Compact Cylinder",
                "vendor": "SMC",
                "cost_usd": 35,
                "force_n": 200,
                "stroke_mm": 50,
            },
        ],
    },
    {
        "id": "pneumatic.pneumatic_cylinder_rodless",
        "name": "Rodless Pneumatic Cylinder",
        "category": "pneumatic",
        "actuator_type": "pneumatic_cylinder",
        "description": (
            "Pneumatic actuator where the carriage rides along the cylinder body "
            "without an extending rod. Enables long strokes in compact packages."
        ),
        "aliases": ["rodless cylinder", "magnetic couple cylinder", "band cylinder"],
        "keywords": {
            "identity": ["rodless pneumatic cylinder", "band cylinder", "magnetic couple cylinder"],
            "descriptions": [
                "long stroke pneumatic", "compact travel actuator", "guided pneumatic slide",
            ],
            "application": [
                "transfer line", "gantry pick-and-place", "conveyor divert",
                "door operator", "palletizer",
            ],
            "industry": ["manufacturing", "packaging", "logistics", "automotive"],
            "components": ["magnetic piston", "sealing band", "guide rail", "carriage"],
            "related": ["installation length", "load capacity", "guide accuracy"],
        },
        "classification": {
            "motion_type": "linear",
            "control_type": "on_off",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 50, "max": 20000, "typical": 800},
            "speed_mps": {"min": 0.1, "max": 3.0, "typical": 1.0},
            "stroke_mm": {"min": 100, "max": 6000, "typical": 1000},
            "weight_grams": {"min": 200, "max": 20000, "typical": 2000},
            "cost_usd": {"min": 100, "max": 3000, "typical": 400},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 24, "max": 24, "typical": 24},
            "feedback": ["hall_effect", "none"],
        },
        "reference_products": [
            {
                "id": "festo_dgc",
                "name": "Festo DGC Rodless Cylinder",
                "vendor": "Festo",
                "cost_usd": 450,
                "stroke_mm": 1500,
            },
        ],
    },
    {
        "id": "pneumatic.pneumatic_valve",
        "name": "Pneumatic Solenoid Valve",
        "category": "pneumatic",
        "actuator_type": "pneumatic_valve",
        "description": (
            "Electrically operated valve that controls compressed air flow to pneumatic "
            "actuators. Available as 3/2-way, 5/2-way, and 5/3-way configurations."
        ),
        "aliases": ["air valve", "solenoid valve", "directional air valve"],
        "keywords": {
            "identity": ["pneumatic solenoid valve", "air valve", "directional air valve"],
            "descriptions": [
                "compressed air valve", "pilot valve", "manifold valve",
                "spool valve pneumatic",
            ],
            "application": [
                "cylinder control", "blow-off", "air logic", "gripper control",
                "valve island",
            ],
            "industry": ["manufacturing", "packaging", "food processing", "automotive"],
            "components": ["solenoid coil", "spool", "seals", "exhaust muffler"],
            "related": ["flow rate", "switching time", "Cv value", "valve island"],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "on_off",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "response_time_ms": {"min": 5, "max": 50, "typical": 15},
            "peak_power_watts": {"min": 1, "max": 10, "typical": 3},
            "continuous_power_watts": {"min": 0.5, "max": 5, "typical": 2},
            "weight_grams": {"min": 10, "max": 500, "typical": 50},
            "cost_usd": {"min": 10, "max": 500, "typical": 40},
            "lifetime_hours": {"min": 50000, "max": 500000, "typical": 100000},
        },
        "interface": {
            "protocol": ["analog", "can", "ethernet_ip"],
            "voltage_v": {"min": 12, "max": 24, "typical": 24},
            "current_a": {"min": 0.05, "max": 0.5, "typical": 0.1},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "festo_vuvg",
                "name": "Festo VUVG Solenoid Valve",
                "vendor": "Festo",
                "cost_usd": 50,
            },
            {
                "id": "smc_sy3000",
                "name": "SMC SY3000 Solenoid Valve",
                "vendor": "SMC",
                "cost_usd": 30,
            },
        ],
    },
    {
        "id": "pneumatic.pneumatic_valve_proportional",
        "name": "Pneumatic Proportional Valve",
        "category": "pneumatic",
        "actuator_type": "pneumatic_valve",
        "description": (
            "Continuously variable pneumatic valve for precise pressure or flow regulation. "
            "Enables proportional control of pneumatic cylinders and soft grippers."
        ),
        "aliases": ["proportional air valve", "pneumatic pressure regulator", "ITV"],
        "keywords": {
            "identity": ["proportional pneumatic valve", "electro-pneumatic regulator", "ITV"],
            "descriptions": [
                "variable pressure valve", "analog pressure control", "proportional flow valve",
            ],
            "application": [
                "pressure regulation", "soft gripper inflation", "force control",
                "tension control", "cushion control",
            ],
            "industry": ["manufacturing", "robotics", "medical devices", "packaging"],
            "components": [
                "piezo valve", "proportional solenoid", "pressure sensor", "controller",
            ],
            "related": [
                "pressure setpoint", "flow linearity", "hysteresis", "repeatability",
            ],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "proportional",
            "environment": ["indoor", "cleanroom"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "response_time_ms": {"min": 5, "max": 100, "typical": 20},
            "peak_power_watts": {"min": 1, "max": 20, "typical": 5},
            "weight_grams": {"min": 50, "max": 1000, "typical": 200},
            "cost_usd": {"min": 50, "max": 2000, "typical": 300},
        },
        "interface": {
            "protocol": ["analog", "rs485", "ethernet_ip"],
            "voltage_v": {"min": 12, "max": 24, "typical": 24},
            "current_a": {"min": 0.05, "max": 0.5, "typical": 0.15},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "smc_itv2050",
                "name": "SMC ITV2050 E/P Regulator",
                "vendor": "SMC",
                "cost_usd": 250,
            },
        ],
    },
    {
        "id": "pneumatic.vacuum_generator",
        "name": "Vacuum Generator",
        "category": "pneumatic",
        "actuator_type": "vacuum_generator",
        "description": (
            "Venturi-based device that converts compressed air into vacuum for "
            "suction gripping and material handling. Compact, no moving parts, "
            "and very fast response."
        ),
        "aliases": ["vacuum ejector", "venturi pump", "vacuum pump pneumatic"],
        "keywords": {
            "identity": ["vacuum generator", "vacuum ejector", "venturi pump"],
            "descriptions": [
                "suction generator", "venturi vacuum device", "inline vacuum",
                "multi-stage ejector",
            ],
            "application": [
                "suction gripping", "pick-and-place", "sheet handling",
                "label placement", "PCB handling",
            ],
            "industry": ["packaging", "electronics", "automotive", "food processing"],
            "components": [
                "venturi nozzle", "silencer", "vacuum filter", "check valve",
            ],
            "related": [
                "vacuum level", "suction flow", "evacuation time", "energy saving",
            ],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "on_off",
            "environment": ["indoor", "cleanroom"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 2, "max": 50, "typical": 10},
            "weight_grams": {"min": 10, "max": 500, "typical": 50},
            "cost_usd": {"min": 15, "max": 500, "typical": 80},
            "response_time_ms": {"min": 10, "max": 200, "typical": 50},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 24, "max": 24, "typical": 24},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "smc_zx",
                "name": "SMC ZX Vacuum Ejector",
                "vendor": "SMC",
                "cost_usd": 60,
            },
            {
                "id": "schmalz_ecbpm",
                "name": "Schmalz ECBPMi Compact Ejector",
                "vendor": "Schmalz",
                "cost_usd": 150,
            },
        ],
    },
    # ======================================================================
    # GRIPPER category
    # ======================================================================
    {
        "id": "gripper.parallel_gripper",
        "name": "Parallel Jaw Gripper",
        "category": "gripper",
        "actuator_type": "parallel_gripper",
        "description": (
            "Two-finger gripper with parallel jaw motion for grasping prismatic and "
            "cylindrical objects. The workhorse of industrial bin picking and machine "
            "tending."
        ),
        "aliases": ["2-jaw gripper", "parallel gripper", "two-finger gripper"],
        "keywords": {
            "identity": ["parallel gripper", "two-finger gripper", "2-jaw gripper"],
            "descriptions": [
                "parallel jaw end-effector", "electric gripper", "pneumatic gripper",
                "adaptive gripper",
            ],
            "application": [
                "bin picking", "machine tending", "assembly", "palletizing",
                "part transfer", "quality inspection",
            ],
            "industry": ["manufacturing", "automotive", "electronics", "logistics"],
            "components": [
                "jaw fingers", "linear guide", "force sensor", "proximity sensor",
            ],
            "related": [
                "grip force", "stroke per finger", "finger compliance", "part centering",
            ],
        },
        "classification": {
            "motion_type": "gripping",
            "control_type": "force",
            "environment": ["indoor", "cleanroom"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "force_n": {"min": 5, "max": 1000, "typical": 100},
            "stroke_mm": {"min": 5, "max": 200, "typical": 85},
            "payload_kg": {"min": 0.1, "max": 20, "typical": 2},
            "peak_power_watts": {"min": 5, "max": 100, "typical": 20},
            "weight_grams": {"min": 100, "max": 3000, "typical": 900},
            "cost_usd": {"min": 200, "max": 5000, "typical": 1500},
            "response_time_ms": {"min": 50, "max": 500, "typical": 150},
        },
        "interface": {
            "protocol": ["modbus", "ethernet_ip", "rs485"],
            "voltage_v": {"min": 12, "max": 24, "typical": 24},
            "current_a": {"min": 0.5, "max": 5, "typical": 1.5},
            "feedback": ["encoder_absolute", "strain_gauge"],
        },
        "reference_products": [
            {
                "id": "robotiq_2f85",
                "name": "Robotiq 2F-85 Adaptive Gripper",
                "vendor": "Robotiq",
                "cost_usd": 3500,
                "force_n": 235,
                "stroke_mm": 85,
                "payload_kg": 5,
            },
            {
                "id": "onrobot_rg2ft",
                "name": "OnRobot RG2-FT",
                "vendor": "OnRobot",
                "cost_usd": 4500,
                "force_n": 40,
                "stroke_mm": 110,
            },
        ],
    },
    {
        "id": "gripper.parallel_gripper_pneumatic",
        "name": "Pneumatic Parallel Gripper",
        "category": "gripper",
        "actuator_type": "parallel_gripper",
        "description": (
            "Air-driven two-finger gripper for high-speed, high-force gripping in "
            "factory automation. Faster than electric but typically on/off control only."
        ),
        "aliases": ["pneumatic 2-jaw gripper", "air gripper"],
        "keywords": {
            "identity": ["pneumatic parallel gripper", "air gripper", "pneumatic 2-finger"],
            "descriptions": [
                "compressed air gripper", "fast industrial gripper", "pneumatic end-effector",
            ],
            "application": [
                "high-speed pick-and-place", "stamping press tending", "injection mold unload",
                "assembly line",
            ],
            "industry": ["automotive", "electronics", "packaging", "manufacturing"],
            "components": ["piston", "jaw guides", "proximity sensor", "air supply"],
            "related": ["grip cycle time", "air consumption", "self-centering"],
        },
        "classification": {
            "motion_type": "gripping",
            "control_type": "on_off",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 10, "max": 3000, "typical": 300},
            "stroke_mm": {"min": 2, "max": 100, "typical": 20},
            "peak_power_watts": {"min": 2, "max": 50, "typical": 10},
            "weight_grams": {"min": 30, "max": 2000, "typical": 200},
            "cost_usd": {"min": 50, "max": 1500, "typical": 200},
            "response_time_ms": {"min": 10, "max": 100, "typical": 30},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 24, "max": 24, "typical": 24},
            "feedback": ["hall_effect", "none"],
        },
        "reference_products": [
            {
                "id": "festo_dhef",
                "name": "Festo DHEF Electric/Pneumatic Gripper",
                "vendor": "Festo",
                "cost_usd": 250,
                "force_n": 140,
                "stroke_mm": 25,
            },
        ],
    },
    {
        "id": "gripper.suction_cup",
        "name": "Suction Cup Gripper",
        "category": "gripper",
        "actuator_type": "suction_cup",
        "description": (
            "Vacuum-based gripper using suction cups to grip flat and smooth surfaces. "
            "Excellent for sheet material, boxes, glass, and PCBs."
        ),
        "aliases": ["vacuum pad", "suction gripper", "vacuum cup"],
        "keywords": {
            "identity": ["suction cup", "vacuum pad", "suction gripper"],
            "descriptions": [
                "vacuum grip end-effector", "non-contact gripping", "surface adhesion gripper",
            ],
            "application": [
                "sheet metal handling", "glass handling", "box palletizing",
                "PCB pick-and-place", "label application",
            ],
            "industry": [
                "logistics", "packaging", "electronics", "glass manufacturing",
                "automotive",
            ],
            "components": [
                "suction cup", "vacuum generator", "vacuum sensor", "quick-change adapter",
            ],
            "related": [
                "vacuum level", "cup material", "lift force", "leak rate",
            ],
        },
        "classification": {
            "motion_type": "gripping",
            "control_type": "on_off",
            "environment": ["indoor", "cleanroom"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "force_n": {"min": 1, "max": 500, "typical": 30},
            "payload_kg": {"min": 0.01, "max": 50, "typical": 3},
            "peak_power_watts": {"min": 1, "max": 50, "typical": 10},
            "weight_grams": {"min": 5, "max": 500, "typical": 30},
            "cost_usd": {"min": 2, "max": 200, "typical": 15},
            "response_time_ms": {"min": 20, "max": 300, "typical": 80},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 24, "max": 24, "typical": 24},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "schmalz_sab",
                "name": "Schmalz SAB Suction Cup",
                "vendor": "Schmalz",
                "cost_usd": 8,
            },
            {
                "id": "piab_bx",
                "name": "Piab BX Suction Cup",
                "vendor": "Piab",
                "cost_usd": 10,
            },
        ],
    },
    {
        "id": "gripper.magnetic_gripper",
        "name": "Magnetic Gripper",
        "category": "gripper",
        "actuator_type": "magnetic_gripper",
        "description": (
            "Gripper using permanent magnets or electromagnets to pick up ferromagnetic "
            "parts. Zero moving parts, handles oily or perforated surfaces that defeat "
            "vacuum grippers."
        ),
        "aliases": ["electromagnetic gripper", "magnet gripper", "magnetic end-effector"],
        "keywords": {
            "identity": ["magnetic gripper", "electromagnetic gripper", "magnet pickup"],
            "descriptions": [
                "ferromagnetic gripping", "contactless release gripper", "switchable magnet",
            ],
            "application": [
                "sheet metal stacking", "bin picking steel parts", "scrap handling",
                "automotive stamping", "steel plate handling",
            ],
            "industry": ["automotive", "steel processing", "manufacturing", "recycling"],
            "components": [
                "permanent magnet", "electromagnet coil", "release mechanism", "housing",
            ],
            "related": [
                "holding force", "residual magnetism", "air gap", "breakaway force",
            ],
        },
        "classification": {
            "motion_type": "gripping",
            "control_type": "on_off",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "force_n": {"min": 10, "max": 10000, "typical": 200},
            "payload_kg": {"min": 0.5, "max": 500, "typical": 20},
            "peak_power_watts": {"min": 5, "max": 200, "typical": 30},
            "weight_grams": {"min": 100, "max": 20000, "typical": 1500},
            "cost_usd": {"min": 50, "max": 5000, "typical": 800},
            "response_time_ms": {"min": 50, "max": 500, "typical": 100},
        },
        "interface": {
            "protocol": ["analog", "can"],
            "voltage_v": {"min": 12, "max": 48, "typical": 24},
            "current_a": {"min": 0.5, "max": 10, "typical": 2},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "schunk_emh",
                "name": "SCHUNK EMH Electromagnetic Gripper",
                "vendor": "SCHUNK",
                "cost_usd": 1200,
                "force_n": 400,
            },
        ],
    },
    {
        "id": "gripper.soft_gripper",
        "name": "Soft Gripper",
        "category": "gripper",
        "actuator_type": "soft_gripper",
        "description": (
            "Compliant gripper made from elastomeric or fabric materials that conforms "
            "to object shape. Gentle on delicate items like food, produce, and baked goods. "
            "Powered by pneumatic inflation or tendon actuation."
        ),
        "aliases": ["flexible gripper", "elastomeric gripper", "compliant gripper"],
        "keywords": {
            "identity": ["soft gripper", "flexible gripper", "elastomeric gripper"],
            "descriptions": [
                "compliant end-effector", "inflatable gripper", "shape-conforming gripper",
                "gentle gripper",
            ],
            "application": [
                "food handling", "produce picking", "bakery", "irregular object gripping",
                "bin picking delicate parts",
            ],
            "industry": ["food processing", "agriculture", "consumer goods", "e-commerce"],
            "components": [
                "silicone body", "pneumatic channel", "tendon cable", "fabric sheath",
            ],
            "related": [
                "compliance", "form closure", "force distribution", "soft robotics",
            ],
        },
        "classification": {
            "motion_type": "gripping",
            "control_type": "pressure",
            "environment": ["indoor", "cleanroom"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 0.5, "max": 50, "typical": 10},
            "payload_kg": {"min": 0.01, "max": 5, "typical": 0.5},
            "peak_power_watts": {"min": 1, "max": 30, "typical": 5},
            "weight_grams": {"min": 20, "max": 500, "typical": 100},
            "cost_usd": {"min": 50, "max": 3000, "typical": 500},
            "response_time_ms": {"min": 100, "max": 2000, "typical": 500},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 24, "max": 24, "typical": 24},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "soft_robotics_mgrip",
                "name": "Soft Robotics mGrip",
                "vendor": "Soft Robotics",
                "cost_usd": 2000,
                "payload_kg": 2.0,
            },
        ],
    },
    {
        "id": "gripper.soft_gripper_tendon",
        "name": "Tendon-Driven Soft Gripper",
        "category": "gripper",
        "actuator_type": "soft_gripper",
        "description": (
            "Soft gripper actuated by tendons (cables) pulled by electric motors. "
            "Offers better position control than pneumatic soft grippers while "
            "retaining compliance."
        ),
        "aliases": ["cable-driven gripper", "tendon gripper", "underactuated hand"],
        "keywords": {
            "identity": ["tendon-driven gripper", "cable-driven gripper", "underactuated gripper"],
            "descriptions": [
                "motor-driven soft gripper", "tendon actuated end-effector", "compliant hand",
            ],
            "application": [
                "assistive robotics", "prosthetic hand", "fruit picking",
                "delicate assembly",
            ],
            "industry": ["robotics", "prosthetics", "agriculture", "research"],
            "components": ["silicone finger", "tendon cable", "servo motor", "spool"],
            "related": ["underactuation", "tendon routing", "adaptive grasp"],
        },
        "classification": {
            "motion_type": "gripping",
            "control_type": "position",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 1, "max": 30, "typical": 8},
            "payload_kg": {"min": 0.05, "max": 3, "typical": 0.5},
            "peak_power_watts": {"min": 2, "max": 20, "typical": 5},
            "weight_grams": {"min": 50, "max": 400, "typical": 150},
            "cost_usd": {"min": 100, "max": 5000, "typical": 800},
        },
        "interface": {
            "protocol": ["pwm", "rs485"],
            "voltage_v": {"min": 5, "max": 12, "typical": 7.4},
            "feedback": ["encoder_incremental"],
        },
        "reference_products": [],
    },
    {
        "id": "gripper.multi_finger_hand",
        "name": "Multi-Finger Dexterous Hand",
        "category": "gripper",
        "actuator_type": "multi_finger_hand",
        "description": (
            "Anthropomorphic or multi-fingered robotic hand with many degrees of "
            "freedom for complex in-hand manipulation. Used in research, teleoperation, "
            "and advanced assembly."
        ),
        "aliases": ["robot hand", "dexterous manipulator", "multi-DOF hand"],
        "keywords": {
            "identity": ["multi-finger hand", "robot hand", "dexterous hand"],
            "descriptions": [
                "anthropomorphic hand", "high DOF manipulator", "in-hand manipulation device",
            ],
            "application": [
                "in-hand manipulation", "teleoperation", "humanoid robot",
                "research manipulation", "prosthetic hand",
            ],
            "industry": ["robotics", "research", "prosthetics", "defense"],
            "components": [
                "finger joints", "tendon cables", "tactile sensors", "servo motors",
                "harmonic drives",
            ],
            "related": [
                "dexterity", "grasp taxonomy", "fingertip force", "tactile sensing",
            ],
        },
        "classification": {
            "motion_type": "gripping",
            "control_type": "position",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 1, "max": 100, "typical": 20},
            "payload_kg": {"min": 0.1, "max": 5, "typical": 1},
            "control_rate_hz": {"min": 100, "max": 2000, "typical": 500},
            "peak_power_watts": {"min": 10, "max": 200, "typical": 50},
            "weight_grams": {"min": 200, "max": 3000, "typical": 800},
            "cost_usd": {"min": 1000, "max": 100000, "typical": 15000},
        },
        "interface": {
            "protocol": ["can", "ethercat", "rs485"],
            "voltage_v": {"min": 12, "max": 48, "typical": 24},
            "current_a": {"min": 1, "max": 15, "typical": 5},
            "feedback": ["encoder_absolute", "strain_gauge"],
        },
        "reference_products": [
            {
                "id": "shadow_hand",
                "name": "Shadow Dexterous Hand",
                "vendor": "Shadow Robot",
                "cost_usd": 80000,
                "payload_kg": 5,
            },
            {
                "id": "allegro_hand",
                "name": "Allegro Hand V4",
                "vendor": "Wonik Robotics",
                "cost_usd": 15000,
            },
        ],
    },
    {
        "id": "gripper.vacuum_gripper",
        "name": "Vacuum Gripper System",
        "category": "gripper",
        "actuator_type": "vacuum_gripper",
        "description": (
            "Integrated vacuum gripping system with multiple suction cups, vacuum "
            "generator, and sensing for handling flat and semi-flat workpieces. "
            "Designed as a complete end-effector for cobots and robots."
        ),
        "aliases": ["vacuum end-effector", "multi-cup vacuum gripper"],
        "keywords": {
            "identity": ["vacuum gripper", "vacuum end-effector", "multi-cup gripper"],
            "descriptions": [
                "integrated vacuum system", "multi-zone vacuum gripper", "smart vacuum gripper",
            ],
            "application": [
                "box palletizing", "carton handling", "sheet handling",
                "depalletizing", "case packing",
            ],
            "industry": ["logistics", "e-commerce", "food processing", "packaging"],
            "components": [
                "suction cups", "vacuum pump", "pressure sensor", "foam seal", "quick-change",
            ],
            "related": [
                "zone control", "leak detection", "grip verification", "tool changing",
            ],
        },
        "classification": {
            "motion_type": "gripping",
            "control_type": "on_off",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "force_n": {"min": 10, "max": 2000, "typical": 200},
            "payload_kg": {"min": 0.5, "max": 100, "typical": 15},
            "peak_power_watts": {"min": 10, "max": 500, "typical": 50},
            "weight_grams": {"min": 200, "max": 5000, "typical": 1200},
            "cost_usd": {"min": 200, "max": 8000, "typical": 2000},
            "response_time_ms": {"min": 50, "max": 500, "typical": 150},
        },
        "interface": {
            "protocol": ["modbus", "ethernet_ip"],
            "voltage_v": {"min": 24, "max": 24, "typical": 24},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "schmalz_fxcb",
                "name": "Schmalz FXCB Vacuum Gripper",
                "vendor": "Schmalz",
                "cost_usd": 3000,
                "payload_kg": 40,
            },
        ],
    },
    # ======================================================================
    # LOCOMOTION category
    # ======================================================================
    {
        "id": "locomotion.wheel_motor",
        "name": "Hub Wheel Motor",
        "category": "locomotion",
        "actuator_type": "wheel_motor",
        "description": (
            "Brushless motor integrated into a wheel hub for direct-drive "
            "differential or omnidirectional platforms. Simplifies mobile robot "
            "drivetrains by eliminating belts and gears."
        ),
        "aliases": ["hub motor", "in-wheel motor", "wheel drive motor"],
        "keywords": {
            "identity": ["hub wheel motor", "in-wheel motor", "hub drive"],
            "descriptions": [
                "direct-drive wheel", "integrated wheel motor", "electric wheel drive",
            ],
            "application": [
                "AGV drive", "AMR platform", "electric scooter", "wheelchair drive",
                "mobile robot base",
            ],
            "industry": ["robotics", "logistics", "personal mobility", "warehouse automation"],
            "components": [
                "BLDC motor", "planetary gearbox", "encoder", "tire", "controller",
            ],
            "related": [
                "wheel odometry", "differential drive", "torque vectoring",
            ],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 0.5, "max": 100, "typical": 10},
            "speed_rpm": {"min": 0, "max": 500, "typical": 100},
            "payload_kg": {"min": 5, "max": 500, "typical": 50},
            "efficiency_percent": {"min": 75, "max": 92, "typical": 85},
            "peak_power_watts": {"min": 20, "max": 3000, "typical": 250},
            "continuous_power_watts": {"min": 10, "max": 1500, "typical": 100},
            "weight_grams": {"min": 200, "max": 10000, "typical": 1500},
            "cost_usd": {"min": 30, "max": 2000, "typical": 150},
        },
        "interface": {
            "protocol": ["can", "pwm", "rs485"],
            "voltage_v": {"min": 12, "max": 48, "typical": 24},
            "current_a": {"min": 1, "max": 30, "typical": 8},
            "feedback": ["encoder_incremental", "hall_effect"],
        },
        "reference_products": [
            {
                "id": "odrive_d6374",
                "name": "ODrive D6374 Hub Motor",
                "vendor": "ODrive Robotics",
                "cost_usd": 80,
                "torque_nm": 8.3,
            },
        ],
    },
    {
        "id": "locomotion.wheel_motor_heavy",
        "name": "Heavy-Duty Wheel Motor",
        "category": "locomotion",
        "actuator_type": "wheel_motor",
        "description": (
            "High-torque wheel drive for outdoor AGVs, forklifts, and rough-terrain "
            "robots. Includes braking, IP65+ sealing, and thermal management."
        ),
        "aliases": ["heavy duty hub motor", "outdoor wheel drive", "industrial wheel motor"],
        "keywords": {
            "identity": ["heavy-duty wheel motor", "industrial hub motor", "outdoor wheel drive"],
            "descriptions": [
                "high torque wheel actuator", "sealed wheel drive", "ruggedized wheel motor",
            ],
            "application": [
                "outdoor AGV", "forklift drive", "field robot", "industrial tug",
                "rough terrain platform",
            ],
            "industry": ["logistics", "agriculture", "construction", "defense"],
            "components": [
                "BLDC motor", "planetary gearbox", "electromagnetic brake", "thermal sensor",
            ],
            "related": ["traction control", "regenerative braking", "IP rating", "thermal derating"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["outdoor", "hazardous"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "torque_nm": {"min": 20, "max": 500, "typical": 100},
            "speed_rpm": {"min": 0, "max": 200, "typical": 60},
            "payload_kg": {"min": 100, "max": 5000, "typical": 500},
            "peak_power_watts": {"min": 500, "max": 20000, "typical": 3000},
            "continuous_power_watts": {"min": 200, "max": 10000, "typical": 1500},
            "weight_grams": {"min": 5000, "max": 50000, "typical": 12000},
            "cost_usd": {"min": 500, "max": 10000, "typical": 2000},
        },
        "interface": {
            "protocol": ["can", "ethercat"],
            "voltage_v": {"min": 24, "max": 80, "typical": 48},
            "current_a": {"min": 10, "max": 200, "typical": 50},
            "feedback": ["encoder_incremental", "hall_effect"],
        },
        "reference_products": [],
    },
    {
        "id": "locomotion.propeller",
        "name": "Propeller Drive",
        "category": "locomotion",
        "actuator_type": "propeller",
        "description": (
            "Combination of brushless motor and propeller for aerial thrust generation. "
            "The primary actuator for multirotors, fixed-wing UAVs, and eVTOL aircraft."
        ),
        "aliases": ["prop drive", "rotor", "thruster"],
        "keywords": {
            "identity": ["propeller", "rotor", "prop drive", "thruster motor"],
            "descriptions": [
                "aerial thrust generator", "multirotor motor", "UAV propulsion",
                "fixed-wing motor",
            ],
            "application": [
                "multirotor drone", "fixed-wing UAV", "eVTOL", "tilt-rotor",
                "aerial photography", "delivery drone",
            ],
            "industry": ["aerospace", "drone", "defense", "delivery", "inspection"],
            "components": [
                "brushless motor", "propeller blade", "ESC", "prop adapter",
            ],
            "related": [
                "thrust coefficient", "advance ratio", "propeller pitch", "motor kv",
            ],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["outdoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 0.01, "max": 50, "typical": 1.5},
            "speed_rpm": {"min": 0, "max": 30000, "typical": 5000},
            "efficiency_percent": {"min": 50, "max": 85, "typical": 70},
            "peak_power_watts": {"min": 10, "max": 50000, "typical": 500},
            "continuous_power_watts": {"min": 5, "max": 30000, "typical": 250},
            "weight_grams": {"min": 10, "max": 10000, "typical": 200},
            "cost_usd": {"min": 10, "max": 5000, "typical": 100},
        },
        "interface": {
            "protocol": ["pwm", "can"],
            "voltage_v": {"min": 7.4, "max": 60, "typical": 22.2},
            "current_a": {"min": 1, "max": 200, "typical": 20},
            "feedback": ["back_emf"],
        },
        "reference_products": [
            {
                "id": "tmotor_u15ii",
                "name": "T-Motor U15II KV100",
                "vendor": "T-Motor",
                "cost_usd": 450,
                "peak_power_watts": 6000,
                "weight_grams": 1100,
            },
            {
                "id": "kde_14215xf",
                "name": "KDE 14215XF-185",
                "vendor": "KDE Direct",
                "cost_usd": 350,
                "peak_power_watts": 4500,
                "weight_grams": 840,
            },
        ],
    },
    {
        "id": "locomotion.propeller_micro",
        "name": "Micro Propeller Drive",
        "category": "locomotion",
        "actuator_type": "propeller",
        "description": (
            "Lightweight motor-propeller combo for micro and nano drones under 250g. "
            "Typically coreless or small BLDC motors with 2-4 inch propellers."
        ),
        "aliases": ["micro rotor", "mini prop", "nano drone motor"],
        "keywords": {
            "identity": ["micro propeller", "mini rotor", "nano drone motor"],
            "descriptions": [
                "lightweight prop drive", "micro UAV thrust", "tiny drone motor",
            ],
            "application": [
                "nano drone", "micro UAV", "indoor drone", "FPV whoop",
                "swarm drone",
            ],
            "industry": ["drone", "defense", "research", "entertainment"],
            "components": ["coreless motor", "micro prop", "micro ESC", "prop guard"],
            "related": ["gram thrust", "power loading", "motor kv", "prop pitch"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 0.001, "max": 0.1, "typical": 0.01},
            "speed_rpm": {"min": 0, "max": 50000, "typical": 20000},
            "peak_power_watts": {"min": 0.5, "max": 50, "typical": 10},
            "continuous_power_watts": {"min": 0.2, "max": 30, "typical": 5},
            "weight_grams": {"min": 1, "max": 30, "typical": 5},
            "cost_usd": {"min": 2, "max": 30, "typical": 8},
        },
        "interface": {
            "protocol": ["pwm"],
            "voltage_v": {"min": 3.7, "max": 14.8, "typical": 7.4},
            "current_a": {"min": 0.2, "max": 10, "typical": 2},
            "feedback": ["back_emf"],
        },
        "reference_products": [],
    },
    {
        "id": "locomotion.jet_thruster",
        "name": "Jet Thruster",
        "category": "locomotion",
        "actuator_type": "jet_thruster",
        "description": (
            "Miniature jet engine or electric ducted fan producing high thrust for "
            "jet-powered UAVs, model aircraft, and experimental platforms."
        ),
        "aliases": ["EDF", "electric ducted fan", "micro jet engine", "turbine thruster"],
        "keywords": {
            "identity": ["jet thruster", "EDF", "electric ducted fan", "micro turbine"],
            "descriptions": [
                "high-speed thrust actuator", "ducted fan propulsion", "jet propulsion",
            ],
            "application": [
                "jet UAV", "model jet aircraft", "high-speed drone", "cruise missile",
                "VTOL transition",
            ],
            "industry": ["aerospace", "defense", "hobby aviation", "research"],
            "components": [
                "impeller", "duct", "stator vanes", "BLDC motor", "turbine wheel",
            ],
            "related": [
                "specific thrust", "bypass ratio", "spool-up time", "exhaust velocity",
            ],
        },
        "classification": {
            "motion_type": "linear",
            "control_type": "velocity",
            "environment": ["outdoor"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "force_n": {"min": 5, "max": 5000, "typical": 50},
            "peak_power_watts": {"min": 50, "max": 100000, "typical": 2000},
            "continuous_power_watts": {"min": 20, "max": 50000, "typical": 1000},
            "weight_grams": {"min": 30, "max": 20000, "typical": 300},
            "cost_usd": {"min": 30, "max": 20000, "typical": 200},
            "response_time_ms": {"min": 50, "max": 5000, "typical": 500},
        },
        "interface": {
            "protocol": ["pwm", "can"],
            "voltage_v": {"min": 12, "max": 60, "typical": 44},
            "current_a": {"min": 2, "max": 200, "typical": 40},
            "feedback": ["back_emf"],
        },
        "reference_products": [],
    },
    {
        "id": "locomotion.leg_actuator",
        "name": "Legged Robot Joint Actuator",
        "category": "locomotion",
        "actuator_type": "leg_actuator",
        "description": (
            "High-bandwidth quasi-direct-drive or geared actuator module designed for "
            "legged robot hip, knee, and ankle joints. Emphasizes torque density, "
            "backdrivability, and impact tolerance."
        ),
        "aliases": ["leg joint motor", "walking actuator", "quadruped joint"],
        "keywords": {
            "identity": ["leg actuator", "legged robot joint", "walking robot motor"],
            "descriptions": [
                "dynamic locomotion actuator", "impact-tolerant joint", "compliant leg motor",
            ],
            "application": [
                "quadruped robot", "bipedal robot", "hexapod", "exoskeleton leg",
                "dynamic walking",
            ],
            "industry": ["robotics", "defense", "research", "prosthetics"],
            "components": [
                "BLDC motor", "planetary gearbox", "torque sensor", "encoder",
                "bearing assembly",
            ],
            "related": [
                "impedance control", "ground reaction force", "joint compliance",
                "proprioception",
            ],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "torque",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 5, "max": 200, "typical": 40},
            "speed_rpm": {"min": 0, "max": 400, "typical": 100},
            "control_rate_hz": {"min": 500, "max": 10000, "typical": 2000},
            "peak_power_watts": {"min": 50, "max": 5000, "typical": 500},
            "continuous_power_watts": {"min": 20, "max": 2000, "typical": 200},
            "weight_grams": {"min": 100, "max": 5000, "typical": 800},
            "cost_usd": {"min": 100, "max": 5000, "typical": 500},
            "response_time_ms": {"min": 0.1, "max": 2, "typical": 0.5},
        },
        "interface": {
            "protocol": ["can", "ethercat"],
            "voltage_v": {"min": 24, "max": 48, "typical": 48},
            "current_a": {"min": 2, "max": 50, "typical": 12},
            "feedback": ["encoder_absolute", "strain_gauge"],
        },
        "reference_products": [
            {
                "id": "unitree_go_m8010",
                "name": "Unitree Go M8010-6",
                "vendor": "Unitree",
                "cost_usd": 280,
                "torque_nm": 23.7,
            },
            {
                "id": "mit_mini_cheetah_motor",
                "name": "MIT Mini Cheetah Actuator",
                "vendor": "MIT (open-source design)",
                "cost_usd": 250,
                "torque_nm": 17,
            },
        ],
    },
    {
        "id": "locomotion.track_drive",
        "name": "Track Drive Motor",
        "category": "locomotion",
        "actuator_type": "track_drive",
        "description": (
            "Motor and sprocket assembly for tracked (caterpillar) platforms. "
            "Provides traction on rough terrain, stairs, and loose surfaces."
        ),
        "aliases": ["caterpillar drive", "tracked drive", "crawler motor"],
        "keywords": {
            "identity": ["track drive", "crawler motor", "caterpillar drive", "tracked vehicle motor"],
            "descriptions": [
                "tracked locomotion actuator", "skid-steer drive", "continuous track motor",
            ],
            "application": [
                "bomb disposal robot", "inspection robot", "stair climbing robot",
                "agricultural robot", "snow removal",
            ],
            "industry": ["defense", "inspection", "agriculture", "construction", "mining"],
            "components": [
                "gearmotor", "sprocket", "track belt", "tension adjuster", "idler wheel",
            ],
            "related": [
                "skid steering", "ground pressure", "drawbar pull", "track tension",
            ],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["outdoor", "hazardous"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "torque_nm": {"min": 1, "max": 500, "typical": 30},
            "speed_rpm": {"min": 0, "max": 300, "typical": 60},
            "payload_kg": {"min": 5, "max": 5000, "typical": 100},
            "peak_power_watts": {"min": 20, "max": 10000, "typical": 500},
            "continuous_power_watts": {"min": 10, "max": 5000, "typical": 200},
            "weight_grams": {"min": 500, "max": 50000, "typical": 3000},
            "cost_usd": {"min": 50, "max": 10000, "typical": 500},
        },
        "interface": {
            "protocol": ["can", "pwm"],
            "voltage_v": {"min": 12, "max": 48, "typical": 24},
            "current_a": {"min": 2, "max": 80, "typical": 15},
            "feedback": ["encoder_incremental"],
        },
        "reference_products": [],
    },
    {
        "id": "locomotion.omnidirectional_wheel",
        "name": "Omnidirectional Wheel Motor",
        "category": "locomotion",
        "actuator_type": "omnidirectional_wheel",
        "description": (
            "Drive module for mecanum or omni-wheel platforms that can move in any "
            "direction without turning. Enables holonomic motion for AGVs, service "
            "robots, and competition robots."
        ),
        "aliases": ["mecanum drive", "omni wheel", "holonomic wheel"],
        "keywords": {
            "identity": ["omnidirectional wheel", "mecanum wheel motor", "omni wheel drive"],
            "descriptions": [
                "holonomic drive module", "lateral motion actuator", "360-degree drive",
            ],
            "application": [
                "warehouse AGV", "service robot", "competition robot",
                "platform cart", "mobile manipulator base",
            ],
            "industry": ["logistics", "robotics", "warehouse automation", "healthcare"],
            "components": [
                "mecanum wheel", "BLDC motor", "encoder", "roller", "controller",
            ],
            "related": [
                "holonomic kinematics", "inverse kinematics", "roller angle", "slip ratio",
            ],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 0.5, "max": 50, "typical": 5},
            "speed_rpm": {"min": 0, "max": 300, "typical": 80},
            "payload_kg": {"min": 5, "max": 500, "typical": 50},
            "peak_power_watts": {"min": 20, "max": 2000, "typical": 150},
            "continuous_power_watts": {"min": 10, "max": 1000, "typical": 80},
            "weight_grams": {"min": 200, "max": 5000, "typical": 800},
            "cost_usd": {"min": 50, "max": 2000, "typical": 200},
        },
        "interface": {
            "protocol": ["can", "pwm", "rs485"],
            "voltage_v": {"min": 12, "max": 48, "typical": 24},
            "current_a": {"min": 1, "max": 20, "typical": 5},
            "feedback": ["encoder_incremental", "hall_effect"],
        },
        "reference_products": [],
    },
    # ======================================================================
    # FLUID category
    # ======================================================================
    {
        "id": "fluid.pump",
        "name": "Centrifugal Pump",
        "category": "fluid",
        "actuator_type": "pump",
        "description": (
            "Rotary pump that uses an impeller to move fluids by centrifugal force. "
            "Suitable for continuous high-flow, low-pressure applications like "
            "cooling loops and irrigation."
        ),
        "aliases": ["centrifugal pump", "impeller pump", "water pump"],
        "keywords": {
            "identity": ["centrifugal pump", "impeller pump", "water pump"],
            "descriptions": [
                "rotary fluid pump", "high flow pump", "continuous flow pump",
            ],
            "application": [
                "cooling system", "irrigation", "water transfer", "HVAC",
                "fountain", "aquarium",
            ],
            "industry": ["HVAC", "agriculture", "industrial", "marine", "data center"],
            "components": ["impeller", "volute casing", "shaft seal", "motor"],
            "related": ["pump curve", "head pressure", "cavitation", "NPSH"],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "velocity",
            "environment": ["indoor", "outdoor"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 5, "max": 5000, "typical": 100},
            "continuous_power_watts": {"min": 2, "max": 3000, "typical": 50},
            "weight_grams": {"min": 50, "max": 20000, "typical": 500},
            "cost_usd": {"min": 5, "max": 2000, "typical": 50},
            "lifetime_hours": {"min": 5000, "max": 100000, "typical": 30000},
        },
        "interface": {
            "protocol": ["pwm", "analog"],
            "voltage_v": {"min": 5, "max": 240, "typical": 12},
            "current_a": {"min": 0.2, "max": 20, "typical": 3},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "grundfos_magna3",
                "name": "Grundfos MAGNA3 Circulator",
                "vendor": "Grundfos",
                "cost_usd": 800,
            },
        ],
    },
    {
        "id": "fluid.pump_gear",
        "name": "Gear Pump",
        "category": "fluid",
        "actuator_type": "pump",
        "description": (
            "Positive-displacement pump using meshing gears to transfer fluid. "
            "Delivers precise, pulsation-free flow at high pressures. "
            "Common in lubrication, fuel delivery, and chemical dosing."
        ),
        "aliases": ["gear pump", "positive displacement pump", "metering pump"],
        "keywords": {
            "identity": ["gear pump", "external gear pump", "internal gear pump"],
            "descriptions": [
                "positive displacement pump", "high pressure pump", "metering pump",
            ],
            "application": [
                "lubrication system", "fuel transfer", "chemical dosing",
                "hydraulic power unit", "adhesive dispensing",
            ],
            "industry": ["manufacturing", "chemical", "oil and gas", "automotive"],
            "components": ["gear pair", "housing", "shaft seal", "relief valve"],
            "related": ["displacement", "volumetric efficiency", "viscosity range"],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "velocity",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 10, "max": 10000, "typical": 200},
            "continuous_power_watts": {"min": 5, "max": 5000, "typical": 100},
            "weight_grams": {"min": 100, "max": 30000, "typical": 2000},
            "cost_usd": {"min": 30, "max": 5000, "typical": 300},
            "lifetime_hours": {"min": 5000, "max": 50000, "typical": 20000},
        },
        "interface": {
            "protocol": ["analog", "can"],
            "voltage_v": {"min": 12, "max": 48, "typical": 24},
            "current_a": {"min": 0.5, "max": 30, "typical": 5},
            "feedback": ["none"],
        },
        "reference_products": [],
    },
    {
        "id": "fluid.peristaltic_pump",
        "name": "Peristaltic Pump",
        "category": "fluid",
        "actuator_type": "peristaltic_pump",
        "description": (
            "Pump that squeezes fluid through a flexible tube using rotating rollers. "
            "Fluid contacts only the tube, making it ideal for sterile, corrosive, "
            "or shear-sensitive applications."
        ),
        "aliases": ["roller pump", "tube pump", "hose pump"],
        "keywords": {
            "identity": ["peristaltic pump", "roller pump", "tube pump"],
            "descriptions": [
                "tube-based pump", "sterile pump", "contamination-free pump",
                "self-priming pump",
            ],
            "application": [
                "medical infusion", "laboratory dosing", "food dispensing",
                "chemical sampling", "bioprocessing",
            ],
            "industry": ["medical", "pharmaceutical", "food processing", "laboratory", "biotechnology"],
            "components": [
                "rotor", "rollers", "flexible tubing", "stepper motor", "tube cartridge",
            ],
            "related": [
                "tube life", "occlusion", "pulsation", "dosing accuracy",
            ],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "velocity",
            "environment": ["indoor", "cleanroom"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 1, "max": 500, "typical": 20},
            "continuous_power_watts": {"min": 0.5, "max": 300, "typical": 10},
            "weight_grams": {"min": 50, "max": 10000, "typical": 500},
            "cost_usd": {"min": 20, "max": 5000, "typical": 300},
            "lifetime_hours": {"min": 1000, "max": 20000, "typical": 5000},
        },
        "interface": {
            "protocol": ["step_dir", "rs485", "analog"],
            "voltage_v": {"min": 5, "max": 24, "typical": 12},
            "current_a": {"min": 0.1, "max": 5, "typical": 1},
            "feedback": ["encoder_incremental"],
        },
        "reference_products": [
            {
                "id": "watson_marlow_530",
                "name": "Watson-Marlow 530",
                "vendor": "Watson-Marlow",
                "cost_usd": 3000,
            },
        ],
    },
    {
        "id": "fluid.peristaltic_pump_micro",
        "name": "Micro Peristaltic Pump",
        "category": "fluid",
        "actuator_type": "peristaltic_pump",
        "description": (
            "Miniaturized peristaltic pump for micro-liter dosing in diagnostics, "
            "lab-on-chip, and point-of-care devices."
        ),
        "aliases": ["micro dosing pump", "miniature peristaltic", "lab pump"],
        "keywords": {
            "identity": ["micro peristaltic pump", "miniature roller pump", "micro dosing pump"],
            "descriptions": [
                "microliter pump", "precision dosing pump", "compact fluid pump",
            ],
            "application": [
                "point-of-care diagnostics", "lab-on-chip", "drug delivery",
                "sample preparation", "reagent dispensing",
            ],
            "industry": ["medical diagnostics", "biotechnology", "pharmaceutical", "research"],
            "components": ["micro motor", "silicone tubing", "mini rollers", "controller PCB"],
            "related": ["microliter accuracy", "tube degradation", "priming volume"],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "velocity",
            "environment": ["indoor", "cleanroom"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 0.1, "max": 5, "typical": 1},
            "continuous_power_watts": {"min": 0.05, "max": 3, "typical": 0.5},
            "weight_grams": {"min": 5, "max": 200, "typical": 30},
            "cost_usd": {"min": 10, "max": 500, "typical": 50},
        },
        "interface": {
            "protocol": ["pwm", "i2c"],
            "voltage_v": {"min": 3.3, "max": 12, "typical": 5},
            "current_a": {"min": 0.01, "max": 0.5, "typical": 0.1},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "knf_nf300",
                "name": "KNF NF300 Micro Pump",
                "vendor": "KNF",
                "cost_usd": 80,
            },
        ],
    },
    {
        "id": "fluid.syringe_pump",
        "name": "Syringe Pump",
        "category": "fluid",
        "actuator_type": "syringe_pump",
        "description": (
            "Precision fluid pump using a motor-driven plunger in a syringe barrel. "
            "Delivers highly accurate and pulsation-free flow. Standard in medical "
            "infusion, microfluidics, and 3D bioprinting."
        ),
        "aliases": ["infusion pump", "precision syringe dispenser"],
        "keywords": {
            "identity": ["syringe pump", "infusion pump", "syringe driver"],
            "descriptions": [
                "precision fluid dispenser", "pulsation-free pump", "linear displacement pump",
            ],
            "application": [
                "IV infusion", "microfluidics", "3D bioprinting", "electrospinning",
                "lab automation", "chromatography",
            ],
            "industry": ["medical", "research", "pharmaceutical", "biotechnology"],
            "components": [
                "syringe barrel", "plunger", "lead screw", "stepper motor", "force sensor",
            ],
            "related": [
                "flow rate accuracy", "syringe size", "infusion profile", "bolus delivery",
            ],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "velocity",
            "environment": ["indoor", "cleanroom"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "force_n": {"min": 1, "max": 500, "typical": 50},
            "speed_mps": {"min": 0.0001, "max": 0.01, "typical": 0.001},
            "peak_power_watts": {"min": 1, "max": 50, "typical": 10},
            "continuous_power_watts": {"min": 0.5, "max": 30, "typical": 5},
            "weight_grams": {"min": 100, "max": 5000, "typical": 800},
            "cost_usd": {"min": 50, "max": 10000, "typical": 1000},
            "position_accuracy_um": {"min": 0.1, "max": 50, "typical": 5},
        },
        "interface": {
            "protocol": ["rs485", "usb", "ethernet_ip"],
            "voltage_v": {"min": 12, "max": 24, "typical": 12},
            "current_a": {"min": 0.2, "max": 3, "typical": 0.5},
            "feedback": ["encoder_incremental"],
        },
        "reference_products": [
            {
                "id": "harvard_phd_ultra",
                "name": "Harvard Apparatus PHD Ultra",
                "vendor": "Harvard Apparatus",
                "cost_usd": 3500,
            },
        ],
    },
    {
        "id": "fluid.dispensing_nozzle",
        "name": "Dispensing Nozzle",
        "category": "fluid",
        "actuator_type": "dispensing_nozzle",
        "description": (
            "Precision valve or nozzle for controlled dispensing of adhesives, solder "
            "paste, conformal coatings, and other fluids. Includes jetting and needle "
            "dispensing types."
        ),
        "aliases": ["dispensing valve", "jetting valve", "dosing nozzle"],
        "keywords": {
            "identity": ["dispensing nozzle", "jetting valve", "dosing valve"],
            "descriptions": [
                "precision dispenser", "adhesive applicator", "solder paste dispenser",
                "micro-dispensing valve",
            ],
            "application": [
                "PCB assembly", "adhesive bonding", "conformal coating",
                "underfill dispensing", "potting",
            ],
            "industry": ["electronics", "semiconductor", "automotive", "medical devices"],
            "components": [
                "needle", "piezo actuator", "pneumatic piston", "heater", "fluid reservoir",
            ],
            "related": [
                "dot size", "shot weight", "viscosity range", "dispense cycle time",
            ],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "on_off",
            "environment": ["indoor", "cleanroom"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "response_time_ms": {"min": 0.1, "max": 50, "typical": 2},
            "peak_power_watts": {"min": 1, "max": 100, "typical": 10},
            "weight_grams": {"min": 20, "max": 1000, "typical": 150},
            "cost_usd": {"min": 100, "max": 10000, "typical": 1500},
            "lifetime_hours": {"min": 5000, "max": 50000, "typical": 15000},
        },
        "interface": {
            "protocol": ["analog", "rs485", "ethernet_ip"],
            "voltage_v": {"min": 12, "max": 24, "typical": 24},
            "current_a": {"min": 0.1, "max": 3, "typical": 0.5},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "nordson_efd_797pico",
                "name": "Nordson EFD 797 PICO Valve",
                "vendor": "Nordson EFD",
                "cost_usd": 5000,
            },
        ],
    },
    {
        "id": "fluid.dispensing_nozzle_inkjet",
        "name": "Inkjet Print Head",
        "category": "fluid",
        "actuator_type": "dispensing_nozzle",
        "description": (
            "Multi-nozzle piezo or thermal inkjet head for high-speed micro-droplet "
            "deposition. Used in industrial printing, 3D printing, and biofabrication."
        ),
        "aliases": ["inkjet head", "print head", "drop-on-demand head"],
        "keywords": {
            "identity": ["inkjet print head", "piezo inkjet", "drop-on-demand head"],
            "descriptions": [
                "micro-droplet dispenser", "multi-nozzle head", "digital printing head",
            ],
            "application": [
                "industrial printing", "3D material jetting", "bioprinting",
                "PCB legend printing", "textile printing",
            ],
            "industry": ["printing", "3D printing", "electronics", "biotechnology"],
            "components": [
                "piezo element", "nozzle plate", "ink channel", "waveform controller",
            ],
            "related": [
                "drop volume", "nozzle count", "jet frequency", "waveform tuning",
            ],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "on_off",
            "environment": ["indoor", "cleanroom"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "response_time_ms": {"min": 0.01, "max": 1, "typical": 0.05},
            "peak_power_watts": {"min": 5, "max": 100, "typical": 20},
            "weight_grams": {"min": 50, "max": 1000, "typical": 200},
            "cost_usd": {"min": 200, "max": 20000, "typical": 2000},
        },
        "interface": {
            "protocol": ["spi", "ethernet_ip"],
            "voltage_v": {"min": 12, "max": 42, "typical": 24},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "xaar_2002",
                "name": "Xaar 2002 GS40K",
                "vendor": "Xaar",
                "cost_usd": 5000,
            },
        ],
    },
    {
        "id": "fluid.sprayer",
        "name": "Sprayer Actuator",
        "category": "fluid",
        "actuator_type": "sprayer",
        "description": (
            "Electromechanical or pressure-driven spray nozzle for atomizing and "
            "distributing liquid over a surface. Used in agriculture, coating, "
            "disinfection, and humidification."
        ),
        "aliases": ["spray nozzle", "atomizer", "spray head"],
        "keywords": {
            "identity": ["sprayer", "spray nozzle", "atomizer", "misting nozzle"],
            "descriptions": [
                "liquid atomizer", "spray actuator", "fog generator", "mist sprayer",
            ],
            "application": [
                "crop spraying", "surface coating", "humidification", "disinfection",
                "dust suppression", "cooling mist",
            ],
            "industry": ["agriculture", "manufacturing", "HVAC", "sanitation", "fire suppression"],
            "components": [
                "nozzle orifice", "solenoid valve", "pressure regulator", "filter", "pump",
            ],
            "related": [
                "spray pattern", "droplet size", "flow rate", "spray angle",
            ],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "on_off",
            "environment": ["indoor", "outdoor"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 2, "max": 500, "typical": 30},
            "weight_grams": {"min": 10, "max": 5000, "typical": 200},
            "cost_usd": {"min": 5, "max": 2000, "typical": 50},
            "response_time_ms": {"min": 10, "max": 200, "typical": 30},
        },
        "interface": {
            "protocol": ["analog", "pwm"],
            "voltage_v": {"min": 12, "max": 48, "typical": 24},
            "current_a": {"min": 0.1, "max": 10, "typical": 1},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "teejet_airmix",
                "name": "TeeJet AirMix 11003",
                "vendor": "TeeJet Technologies",
                "cost_usd": 15,
            },
        ],
    },
    {
        "id": "fluid.sprayer_electrostatic",
        "name": "Electrostatic Sprayer",
        "category": "fluid",
        "actuator_type": "sprayer",
        "description": (
            "Sprayer that electrically charges droplets for uniform surface coverage "
            "with reduced overspray. Used in painting, coating, and agricultural spraying."
        ),
        "aliases": ["electrostatic spray gun", "charged spray nozzle"],
        "keywords": {
            "identity": ["electrostatic sprayer", "charged sprayer", "e-spray nozzle"],
            "descriptions": [
                "charged droplet sprayer", "wrap-around spray", "high-transfer spray",
            ],
            "application": [
                "automotive painting", "powder coating", "crop protection",
                "disinfection", "furniture finishing",
            ],
            "industry": ["automotive", "agriculture", "furniture", "aerospace"],
            "components": [
                "high voltage electrode", "nozzle", "charging ring", "grounding clamp",
            ],
            "related": [
                "transfer efficiency", "Faraday cage effect", "charge-to-mass ratio",
            ],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "proportional",
            "environment": ["indoor", "outdoor"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 10, "max": 2000, "typical": 100},
            "weight_grams": {"min": 200, "max": 5000, "typical": 1000},
            "cost_usd": {"min": 100, "max": 10000, "typical": 1500},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 12, "max": 24, "typical": 24},
            "feedback": ["none"],
        },
        "reference_products": [],
    },
    # ======================================================================
    # DISPLAY category
    # ======================================================================
    {
        "id": "display.led_array",
        "name": "LED Array",
        "category": "display",
        "actuator_type": "led_array",
        "description": (
            "Addressable LED array or matrix for status indication, signaling, "
            "and decorative lighting on robots and autonomous systems."
        ),
        "aliases": ["LED strip", "NeoPixel", "addressable LED", "LED panel"],
        "keywords": {
            "identity": ["LED array", "addressable LED", "NeoPixel strip", "RGB LED matrix"],
            "descriptions": [
                "light indicator", "status display", "programmable lighting",
                "decorative illumination",
            ],
            "application": [
                "robot status indicator", "drone navigation light", "AMR signal light",
                "decorative display", "alerting system",
            ],
            "industry": ["robotics", "entertainment", "signage", "automotive"],
            "components": ["LED chip", "driver IC", "PCB strip", "diffuser"],
            "related": ["WS2812", "APA102", "PWM dimming", "color temperature"],
        },
        "classification": {
            "motion_type": "display_output",
            "control_type": "on_off",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 0.1, "max": 100, "typical": 5},
            "continuous_power_watts": {"min": 0.05, "max": 60, "typical": 3},
            "weight_grams": {"min": 2, "max": 500, "typical": 30},
            "cost_usd": {"min": 1, "max": 100, "typical": 10},
            "lifetime_hours": {"min": 20000, "max": 100000, "typical": 50000},
        },
        "interface": {
            "protocol": ["spi", "i2c", "pwm"],
            "voltage_v": {"min": 3.3, "max": 12, "typical": 5},
            "current_a": {"min": 0.01, "max": 10, "typical": 0.5},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "ws2812b_strip",
                "name": "WorldSemi WS2812B Strip",
                "vendor": "WorldSemi",
                "cost_usd": 5,
            },
        ],
    },
    {
        "id": "display.led_array_high_power",
        "name": "High-Power LED Module",
        "category": "display",
        "actuator_type": "led_array",
        "description": (
            "High-intensity LED module for illumination, machine vision lighting, "
            "and LiDAR reference. Includes thermal management and driver electronics."
        ),
        "aliases": ["high power LED", "illumination LED", "machine vision light"],
        "keywords": {
            "identity": ["high power LED module", "illumination LED", "machine vision light"],
            "descriptions": [
                "high intensity light source", "structured illumination", "vision lighting",
            ],
            "application": [
                "machine vision illumination", "headlamp", "searchlight",
                "agricultural grow light", "UV curing",
            ],
            "industry": ["manufacturing", "automotive", "agriculture", "medical"],
            "components": ["LED die", "heatsink", "lens", "constant current driver"],
            "related": ["luminous flux", "CRI", "beam angle", "thermal derating"],
        },
        "classification": {
            "motion_type": "display_output",
            "control_type": "proportional",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 1, "max": 500, "typical": 30},
            "continuous_power_watts": {"min": 0.5, "max": 300, "typical": 20},
            "weight_grams": {"min": 10, "max": 2000, "typical": 100},
            "cost_usd": {"min": 5, "max": 500, "typical": 30},
            "lifetime_hours": {"min": 20000, "max": 100000, "typical": 50000},
        },
        "interface": {
            "protocol": ["pwm", "analog", "i2c"],
            "voltage_v": {"min": 5, "max": 48, "typical": 24},
            "current_a": {"min": 0.1, "max": 20, "typical": 1},
            "feedback": ["none"],
        },
        "reference_products": [],
    },
    {
        "id": "display.oled_display",
        "name": "OLED Display Module",
        "category": "display",
        "actuator_type": "oled_display",
        "description": (
            "Organic LED display module for status information, menus, and data "
            "visualization on embedded systems and robots."
        ),
        "aliases": ["OLED screen", "OLED panel", "micro display"],
        "keywords": {
            "identity": ["OLED display", "OLED screen", "OLED module"],
            "descriptions": [
                "self-emissive display", "thin film display", "embedded display",
                "organic display",
            ],
            "application": [
                "robot HMI", "drone status display", "wearable display",
                "instrument panel", "IoT display",
            ],
            "industry": ["robotics", "consumer electronics", "medical devices", "IoT"],
            "components": ["OLED panel", "driver IC", "flex cable", "cover glass"],
            "related": ["pixel density", "contrast ratio", "burn-in", "refresh rate"],
        },
        "classification": {
            "motion_type": "display_output",
            "control_type": "on_off",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 0.05, "max": 10, "typical": 0.5},
            "continuous_power_watts": {"min": 0.02, "max": 5, "typical": 0.3},
            "weight_grams": {"min": 2, "max": 200, "typical": 10},
            "cost_usd": {"min": 3, "max": 200, "typical": 15},
            "lifetime_hours": {"min": 10000, "max": 50000, "typical": 30000},
        },
        "interface": {
            "protocol": ["i2c", "spi"],
            "voltage_v": {"min": 3.3, "max": 5, "typical": 3.3},
            "current_a": {"min": 0.01, "max": 0.5, "typical": 0.05},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "solomon_ssd1306",
                "name": "SSD1306 0.96\" OLED Module",
                "vendor": "Solomon Systech",
                "cost_usd": 5,
            },
        ],
    },
    {
        "id": "display.projector",
        "name": "Micro Projector",
        "category": "display",
        "actuator_type": "projector",
        "description": (
            "Compact projector module using DLP, LCoS, or laser scanning for "
            "projecting images, structured light, or information onto surfaces."
        ),
        "aliases": ["pico projector", "DLP projector", "laser projector"],
        "keywords": {
            "identity": ["micro projector", "pico projector", "DLP projector", "laser projector"],
            "descriptions": [
                "compact image projector", "structured light source", "embedded projector",
            ],
            "application": [
                "structured light scanning", "robot HMI projection", "AR display",
                "inspection projection", "safety zone projection",
            ],
            "industry": ["robotics", "inspection", "manufacturing", "entertainment"],
            "components": ["DLP chip", "laser diode", "MEMS mirror", "lens assembly"],
            "related": ["throw ratio", "lumens", "keystone correction", "structured light"],
        },
        "classification": {
            "motion_type": "display_output",
            "control_type": "on_off",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 1, "max": 100, "typical": 15},
            "continuous_power_watts": {"min": 0.5, "max": 60, "typical": 10},
            "weight_grams": {"min": 10, "max": 500, "typical": 50},
            "cost_usd": {"min": 20, "max": 2000, "typical": 200},
            "lifetime_hours": {"min": 10000, "max": 50000, "typical": 20000},
        },
        "interface": {
            "protocol": ["spi", "usb", "i2c"],
            "voltage_v": {"min": 3.3, "max": 12, "typical": 5},
            "current_a": {"min": 0.2, "max": 5, "typical": 1},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "ti_dlp2010",
                "name": "TI DLP2010 Evaluation Module",
                "vendor": "Texas Instruments",
                "cost_usd": 300,
            },
        ],
    },
    {
        "id": "display.speaker",
        "name": "Speaker / Audio Transducer",
        "category": "display",
        "actuator_type": "speaker",
        "description": (
            "Electromechanical transducer that converts electrical signals into sound. "
            "Used for alarms, voice feedback, navigation cues, and human-robot "
            "interaction on autonomous systems."
        ),
        "aliases": ["loudspeaker", "audio transducer", "buzzer", "horn"],
        "keywords": {
            "identity": ["speaker", "loudspeaker", "audio transducer", "buzzer"],
            "descriptions": [
                "sound output device", "audio actuator", "acoustic transducer",
                "alarm sounder",
            ],
            "application": [
                "voice feedback", "alarm signal", "navigation audio", "human-robot interaction",
                "public address",
            ],
            "industry": ["robotics", "automotive", "consumer electronics", "building automation"],
            "components": ["voice coil", "diaphragm", "magnet", "amplifier", "enclosure"],
            "related": ["SPL", "frequency response", "impedance", "THD"],
        },
        "classification": {
            "motion_type": "vibration",
            "control_type": "proportional",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 0.1, "max": 200, "typical": 5},
            "continuous_power_watts": {"min": 0.05, "max": 100, "typical": 3},
            "weight_grams": {"min": 2, "max": 5000, "typical": 50},
            "cost_usd": {"min": 0.5, "max": 200, "typical": 5},
            "lifetime_hours": {"min": 10000, "max": 100000, "typical": 50000},
        },
        "interface": {
            "protocol": ["analog", "i2c", "pwm"],
            "voltage_v": {"min": 3.3, "max": 48, "typical": 5},
            "current_a": {"min": 0.01, "max": 10, "typical": 0.5},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "murata_pklcs1212e",
                "name": "Murata PKLCS1212E4001",
                "vendor": "Murata",
                "cost_usd": 2,
            },
        ],
    },
    {
        "id": "display.haptic_actuator",
        "name": "Haptic Feedback Actuator",
        "category": "display",
        "actuator_type": "haptic_actuator",
        "description": (
            "Vibrotactile or force-feedback actuator for conveying tactile information "
            "to operators. Used in teleoperation, VR controllers, wearable devices, "
            "and human-robot interfaces."
        ),
        "aliases": ["vibration motor", "haptic engine", "tactile actuator", "ERM motor"],
        "keywords": {
            "identity": ["haptic actuator", "vibration motor", "tactile feedback", "ERM", "LRA"],
            "descriptions": [
                "vibrotactile transducer", "force feedback device", "touch feedback actuator",
            ],
            "application": [
                "teleoperation feedback", "VR controller", "wearable alert",
                "automotive touchscreen", "surgical haptics",
            ],
            "industry": ["consumer electronics", "medical", "automotive", "VR/AR", "robotics"],
            "components": [
                "eccentric mass", "linear resonant actuator", "piezo element", "driver IC",
            ],
            "related": [
                "vibration frequency", "haptic waveform", "latency", "localized feedback",
            ],
        },
        "classification": {
            "motion_type": "vibration",
            "control_type": "proportional",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "force_n": {"min": 0.1, "max": 10, "typical": 1},
            "peak_power_watts": {"min": 0.05, "max": 5, "typical": 0.5},
            "continuous_power_watts": {"min": 0.02, "max": 3, "typical": 0.3},
            "weight_grams": {"min": 0.5, "max": 50, "typical": 3},
            "cost_usd": {"min": 0.5, "max": 30, "typical": 2},
            "response_time_ms": {"min": 1, "max": 20, "typical": 5},
        },
        "interface": {
            "protocol": ["pwm", "i2c"],
            "voltage_v": {"min": 1.8, "max": 5, "typical": 3.3},
            "current_a": {"min": 0.01, "max": 0.5, "typical": 0.1},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "ti_drv2605l",
                "name": "TI DRV2605L Haptic Driver + LRA",
                "vendor": "Texas Instruments",
                "cost_usd": 3,
            },
        ],
    },
    {
        "id": "display.haptic_actuator_piezo",
        "name": "Piezo Haptic Actuator",
        "category": "display",
        "actuator_type": "haptic_actuator",
        "description": (
            "Piezoelectric haptic transducer producing sharp, localized tactile "
            "feedback with very low latency. Preferred for touchscreens and "
            "precision teleoperation."
        ),
        "aliases": ["piezo haptic", "piezoelectric vibrator", "ultra-thin haptic"],
        "keywords": {
            "identity": ["piezo haptic actuator", "piezoelectric transducer", "thin haptic"],
            "descriptions": [
                "sharp tactile feedback", "low-latency haptic", "surface haptic transducer",
            ],
            "application": [
                "touchscreen feedback", "surgical instrument feedback", "stylus haptics",
                "braille display",
            ],
            "industry": ["consumer electronics", "medical", "accessibility", "VR/AR"],
            "components": ["piezo ceramic", "flex PCB", "driver IC", "adhesive layer"],
            "related": ["resonance frequency", "displacement", "waveform shaping"],
        },
        "classification": {
            "motion_type": "vibration",
            "control_type": "proportional",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "force_n": {"min": 0.01, "max": 5, "typical": 0.5},
            "peak_power_watts": {"min": 0.01, "max": 2, "typical": 0.2},
            "weight_grams": {"min": 0.2, "max": 10, "typical": 1},
            "cost_usd": {"min": 1, "max": 50, "typical": 5},
            "response_time_ms": {"min": 0.1, "max": 2, "typical": 0.5},
        },
        "interface": {
            "protocol": ["spi", "i2c"],
            "voltage_v": {"min": 3.3, "max": 200, "typical": 60},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "tdk_powerhap",
                "name": "TDK PowerHap 2.5G",
                "vendor": "TDK",
                "cost_usd": 8,
            },
        ],
    },
    # ======================================================================
    # SPECIALTY category
    # ======================================================================
    {
        "id": "specialty.laser_cutter",
        "name": "Laser Cutter Head",
        "category": "specialty",
        "actuator_type": "laser_cutter",
        "description": (
            "Focused laser beam delivery system for cutting, engraving, and marking "
            "materials. Includes CO2, fiber, and diode laser types with beam focusing "
            "optics and assist gas delivery."
        ),
        "aliases": ["laser head", "laser engraver", "laser cutting module"],
        "keywords": {
            "identity": ["laser cutter", "laser engraver", "laser cutting head"],
            "descriptions": [
                "focused beam cutter", "non-contact cutting tool", "thermal cutting head",
            ],
            "application": [
                "sheet metal cutting", "acrylic engraving", "PCB depaneling",
                "textile cutting", "wood engraving",
            ],
            "industry": ["manufacturing", "signage", "electronics", "jewelry", "prototyping"],
            "components": [
                "laser source", "focusing lens", "nozzle", "assist gas valve", "beam expander",
            ],
            "related": [
                "beam quality", "kerf width", "focal length", "cutting speed",
            ],
        },
        "classification": {
            "motion_type": "display_output",
            "control_type": "proportional",
            "environment": ["indoor"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 5, "max": 30000, "typical": 500},
            "continuous_power_watts": {"min": 2, "max": 20000, "typical": 300},
            "position_accuracy_um": {"min": 5, "max": 200, "typical": 25},
            "weight_grams": {"min": 200, "max": 20000, "typical": 3000},
            "cost_usd": {"min": 200, "max": 100000, "typical": 5000},
            "lifetime_hours": {"min": 5000, "max": 100000, "typical": 20000},
        },
        "interface": {
            "protocol": ["analog", "ethernet_ip", "rs485"],
            "voltage_v": {"min": 24, "max": 400, "typical": 48},
            "current_a": {"min": 1, "max": 100, "typical": 15},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "ipg_ylr_500",
                "name": "IPG YLR-500 Fiber Laser",
                "vendor": "IPG Photonics",
                "cost_usd": 25000,
                "peak_power_watts": 500,
            },
        ],
    },
    {
        "id": "specialty.laser_cutter_diode",
        "name": "Diode Laser Module",
        "category": "specialty",
        "actuator_type": "laser_cutter",
        "description": (
            "Compact laser diode module for desktop engraving, marking, and light "
            "cutting. Low cost and easy to integrate with CNC/3D printer frames."
        ),
        "aliases": ["laser diode cutter", "desktop laser", "laser module"],
        "keywords": {
            "identity": ["diode laser module", "desktop laser cutter", "blue laser module"],
            "descriptions": [
                "compact laser cutter", "hobby laser engraver", "low power laser tool",
            ],
            "application": [
                "desktop engraving", "leather cutting", "paper cutting",
                "marking", "hobby crafting",
            ],
            "industry": ["prototyping", "hobby", "education", "craft"],
            "components": ["laser diode", "collimating lens", "heatsink", "driver board"],
            "related": ["wavelength", "spot size", "duty cycle", "optical power"],
        },
        "classification": {
            "motion_type": "display_output",
            "control_type": "proportional",
            "environment": ["indoor"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 0.5, "max": 40, "typical": 10},
            "continuous_power_watts": {"min": 0.2, "max": 30, "typical": 5},
            "weight_grams": {"min": 20, "max": 500, "typical": 80},
            "cost_usd": {"min": 20, "max": 500, "typical": 100},
            "lifetime_hours": {"min": 5000, "max": 30000, "typical": 10000},
        },
        "interface": {
            "protocol": ["pwm", "analog"],
            "voltage_v": {"min": 5, "max": 24, "typical": 12},
            "current_a": {"min": 0.5, "max": 10, "typical": 3},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "ortur_lm3",
                "name": "Ortur Laser Module 20W",
                "vendor": "Ortur",
                "cost_usd": 120,
            },
        ],
    },
    {
        "id": "specialty.plasma_torch",
        "name": "Plasma Cutting Torch",
        "category": "specialty",
        "actuator_type": "plasma_torch",
        "description": (
            "Thermal cutting tool that uses a focused plasma arc to cut electrically "
            "conductive metals. Fast and effective on steel, stainless, and aluminum "
            "at thicknesses up to 50mm."
        ),
        "aliases": ["plasma cutter", "plasma arc torch", "CNC plasma head"],
        "keywords": {
            "identity": ["plasma torch", "plasma cutter", "plasma arc torch"],
            "descriptions": [
                "ionized gas cutting tool", "metal cutting torch", "thermal arc cutter",
            ],
            "application": [
                "steel plate cutting", "structural steel fabrication", "HVAC ductwork",
                "shipbuilding", "automotive repair",
            ],
            "industry": ["metalworking", "construction", "shipbuilding", "automotive"],
            "components": [
                "electrode", "nozzle", "swirl ring", "shield cup", "plasma gas supply",
            ],
            "related": [
                "arc voltage", "pierce height", "cutting speed", "kerf width",
            ],
        },
        "classification": {
            "motion_type": "display_output",
            "control_type": "proportional",
            "environment": ["indoor", "outdoor"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 2000, "max": 100000, "typical": 15000},
            "continuous_power_watts": {"min": 1000, "max": 80000, "typical": 10000},
            "weight_grams": {"min": 500, "max": 10000, "typical": 2000},
            "cost_usd": {"min": 500, "max": 50000, "typical": 5000},
            "lifetime_hours": {"min": 200, "max": 5000, "typical": 1000},
        },
        "interface": {
            "protocol": ["analog", "can"],
            "voltage_v": {"min": 100, "max": 400, "typical": 200},
            "current_a": {"min": 10, "max": 400, "typical": 80},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "hypertherm_powermax45",
                "name": "Hypertherm Powermax45 XP",
                "vendor": "Hypertherm",
                "cost_usd": 2500,
                "peak_power_watts": 6750,
            },
        ],
    },
    {
        "id": "specialty.plasma_torch_fine",
        "name": "High-Definition Plasma Torch",
        "category": "specialty",
        "actuator_type": "plasma_torch",
        "description": (
            "Precision plasma cutting system with fine-kerf nozzles and THC (torch height "
            "control) for CNC-quality cuts approaching laser quality on thin sheet."
        ),
        "aliases": ["HD plasma", "fine plasma", "precision plasma torch"],
        "keywords": {
            "identity": ["high-definition plasma torch", "HD plasma cutter", "fine plasma"],
            "descriptions": [
                "precision plasma cutter", "narrow kerf plasma", "CNC plasma system",
            ],
            "application": [
                "precision metal parts", "signage fabrication", "artistic metalwork",
                "thin sheet cutting",
            ],
            "industry": ["metalworking", "signage", "art fabrication", "manufacturing"],
            "components": [
                "fine-kerf nozzle", "THC controller", "electrode", "coolant system",
            ],
            "related": [
                "edge quality", "bevel angle", "dross", "pierce delay",
            ],
        },
        "classification": {
            "motion_type": "display_output",
            "control_type": "proportional",
            "environment": ["indoor"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 5000, "max": 200000, "typical": 40000},
            "continuous_power_watts": {"min": 3000, "max": 150000, "typical": 30000},
            "position_accuracy_um": {"min": 50, "max": 500, "typical": 150},
            "weight_grams": {"min": 1000, "max": 15000, "typical": 5000},
            "cost_usd": {"min": 5000, "max": 200000, "typical": 30000},
        },
        "interface": {
            "protocol": ["ethernet_ip", "analog"],
            "voltage_v": {"min": 200, "max": 600, "typical": 400},
            "current_a": {"min": 30, "max": 800, "typical": 200},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "hypertherm_hpr400xd",
                "name": "Hypertherm HPR400XD",
                "vendor": "Hypertherm",
                "cost_usd": 80000,
            },
        ],
    },
    {
        "id": "specialty.welding_torch",
        "name": "MIG/MAG Welding Torch",
        "category": "specialty",
        "actuator_type": "welding_torch",
        "description": (
            "Gas metal arc welding (GMAW) torch for robotic and manual welding. "
            "Feeds wire electrode and shielding gas while maintaining a controlled arc."
        ),
        "aliases": ["welding gun", "MIG torch", "GMAW torch", "robotic welding torch"],
        "keywords": {
            "identity": ["welding torch", "MIG torch", "MAG torch", "GMAW torch"],
            "descriptions": [
                "arc welding tool", "wire-feed welding head", "robotic weld gun",
            ],
            "application": [
                "automotive body welding", "structural fabrication", "pipe welding",
                "robotic cell welding", "repair welding",
            ],
            "industry": ["automotive", "construction", "shipbuilding", "manufacturing", "aerospace"],
            "components": [
                "contact tip", "gas nozzle", "wire liner", "wire feeder", "shielding gas supply",
            ],
            "related": [
                "wire feed speed", "arc voltage", "weld penetration", "heat input",
            ],
        },
        "classification": {
            "motion_type": "display_output",
            "control_type": "proportional",
            "environment": ["indoor"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 1000, "max": 50000, "typical": 8000},
            "continuous_power_watts": {"min": 500, "max": 30000, "typical": 5000},
            "weight_grams": {"min": 300, "max": 5000, "typical": 1500},
            "cost_usd": {"min": 100, "max": 10000, "typical": 1500},
            "lifetime_hours": {"min": 1000, "max": 20000, "typical": 5000},
        },
        "interface": {
            "protocol": ["analog", "can"],
            "voltage_v": {"min": 15, "max": 40, "typical": 25},
            "current_a": {"min": 30, "max": 500, "typical": 200},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "fronius_robacta",
                "name": "Fronius Robacta Drive",
                "vendor": "Fronius",
                "cost_usd": 3000,
            },
        ],
    },
    {
        "id": "specialty.welding_torch_tig",
        "name": "TIG Welding Torch",
        "category": "specialty",
        "actuator_type": "welding_torch",
        "description": (
            "Gas tungsten arc welding (GTAW/TIG) torch for high-quality, precise "
            "welds on thin materials and exotic alloys. Non-consumable tungsten electrode."
        ),
        "aliases": ["TIG torch", "GTAW torch", "argon welding torch"],
        "keywords": {
            "identity": ["TIG welding torch", "GTAW torch", "tungsten arc torch"],
            "descriptions": [
                "precision welding tool", "thin material welding", "non-consumable electrode welder",
            ],
            "application": [
                "aerospace welding", "pipe root pass", "stainless steel welding",
                "aluminum welding", "titanium welding",
            ],
            "industry": ["aerospace", "nuclear", "food processing", "pharmaceutical", "art"],
            "components": [
                "tungsten electrode", "ceramic cup", "collet", "gas lens", "filler wire feeder",
            ],
            "related": [
                "arc frequency", "pulse welding", "post-flow", "electrode preparation",
            ],
        },
        "classification": {
            "motion_type": "display_output",
            "control_type": "proportional",
            "environment": ["indoor"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 500, "max": 20000, "typical": 5000},
            "continuous_power_watts": {"min": 200, "max": 15000, "typical": 3000},
            "weight_grams": {"min": 200, "max": 3000, "typical": 800},
            "cost_usd": {"min": 50, "max": 5000, "typical": 500},
            "lifetime_hours": {"min": 2000, "max": 30000, "typical": 10000},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 10, "max": 35, "typical": 18},
            "current_a": {"min": 5, "max": 350, "typical": 150},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "miller_dynasty_280",
                "name": "Miller Dynasty 280 DX Torch",
                "vendor": "Miller Electric",
                "cost_usd": 400,
            },
        ],
    },
    {
        "id": "specialty.welding_torch_laser",
        "name": "Laser Welding Head",
        "category": "specialty",
        "actuator_type": "welding_torch",
        "description": (
            "Focused laser beam welding head for high-speed, low-distortion joining. "
            "Used in automotive BIW, battery tab welding, and micro-welding."
        ),
        "aliases": ["laser welder", "laser welding optic", "beam welding head"],
        "keywords": {
            "identity": ["laser welding head", "laser welder", "beam welding optic"],
            "descriptions": [
                "non-contact welding tool", "high-speed welding head", "precision laser joint",
            ],
            "application": [
                "battery tab welding", "automotive BIW", "medical device welding",
                "jewelry welding", "micro-welding",
            ],
            "industry": ["automotive", "electronics", "medical devices", "jewelry"],
            "components": [
                "fiber laser", "collimating lens", "focusing optic", "crossjet", "seam tracker",
            ],
            "related": [
                "weld depth", "beam diameter", "wobble welding", "keyhole welding",
            ],
        },
        "classification": {
            "motion_type": "display_output",
            "control_type": "proportional",
            "environment": ["indoor", "cleanroom"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 100, "max": 50000, "typical": 3000},
            "continuous_power_watts": {"min": 50, "max": 30000, "typical": 2000},
            "position_accuracy_um": {"min": 5, "max": 100, "typical": 20},
            "weight_grams": {"min": 500, "max": 15000, "typical": 3000},
            "cost_usd": {"min": 2000, "max": 200000, "typical": 20000},
        },
        "interface": {
            "protocol": ["ethernet_ip", "analog"],
            "voltage_v": {"min": 24, "max": 400, "typical": 48},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "trumpf_beo_d70",
                "name": "TRUMPF BEO D70 Welding Optic",
                "vendor": "TRUMPF",
                "cost_usd": 15000,
            },
        ],
    },
    {
        "id": "specialty.paint_sprayer",
        "name": "Robotic Paint Sprayer",
        "category": "specialty",
        "actuator_type": "paint_sprayer",
        "description": (
            "Automated paint application system for uniform coating of parts and "
            "surfaces. Includes electrostatic, airless, and HVLP spray technologies."
        ),
        "aliases": ["paint gun", "spray gun", "robotic coater"],
        "keywords": {
            "identity": ["paint sprayer", "spray gun", "robotic paint applicator"],
            "descriptions": [
                "automated coating system", "surface finishing tool", "paint application device",
            ],
            "application": [
                "automotive body painting", "furniture coating", "aerospace coating",
                "appliance painting", "wood finishing",
            ],
            "industry": ["automotive", "furniture", "aerospace", "appliance", "general manufacturing"],
            "components": [
                "spray bell", "turbine", "paint hose", "color changer", "flow meter",
            ],
            "related": [
                "transfer efficiency", "film thickness", "overspray", "pattern width",
            ],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "proportional",
            "environment": ["indoor"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 50, "max": 5000, "typical": 500},
            "continuous_power_watts": {"min": 20, "max": 3000, "typical": 300},
            "weight_grams": {"min": 300, "max": 5000, "typical": 1500},
            "cost_usd": {"min": 500, "max": 50000, "typical": 8000},
            "lifetime_hours": {"min": 2000, "max": 30000, "typical": 10000},
        },
        "interface": {
            "protocol": ["analog", "profinet", "can"],
            "voltage_v": {"min": 24, "max": 400, "typical": 24},
            "current_a": {"min": 1, "max": 20, "typical": 5},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "fanuc_p250ia",
                "name": "FANUC P-250iA Paint Robot Applicator",
                "vendor": "FANUC",
                "cost_usd": 30000,
            },
        ],
    },
    {
        "id": "specialty.paint_sprayer_powder",
        "name": "Powder Coating Gun",
        "category": "specialty",
        "actuator_type": "paint_sprayer",
        "description": (
            "Electrostatic spray gun for applying powder coating to metal parts. "
            "Powder is charged and attracted to grounded workpiece, then cured in an oven."
        ),
        "aliases": ["powder spray gun", "electrostatic powder gun", "powder applicator"],
        "keywords": {
            "identity": ["powder coating gun", "powder sprayer", "electrostatic powder applicator"],
            "descriptions": [
                "dry coating applicator", "powder paint system", "electrostatic deposition gun",
            ],
            "application": [
                "metal furniture coating", "automotive parts", "architectural aluminum",
                "appliance coating",
            ],
            "industry": ["metalworking", "furniture", "automotive", "architectural"],
            "components": [
                "corona electrode", "powder pump", "fluidized hopper", "control unit",
            ],
            "related": [
                "first-pass transfer", "Faraday penetration", "powder recovery",
                "cure temperature",
            ],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "proportional",
            "environment": ["indoor"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 20, "max": 500, "typical": 100},
            "weight_grams": {"min": 200, "max": 3000, "typical": 800},
            "cost_usd": {"min": 200, "max": 15000, "typical": 3000},
            "lifetime_hours": {"min": 5000, "max": 30000, "typical": 15000},
        },
        "interface": {
            "protocol": ["analog", "can"],
            "voltage_v": {"min": 24, "max": 24, "typical": 24},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "gema_optiflex",
                "name": "Gema OptiFlex Pro",
                "vendor": "Gema",
                "cost_usd": 5000,
            },
        ],
    },
    # ======================================================================
    # ADDITIONAL entries to reach 80+ total
    # ======================================================================
    # -- motor extras --
    {
        "id": "motor.brushless_dc_gimbal",
        "name": "Gimbal Brushless Motor",
        "category": "motor",
        "actuator_type": "brushless_dc",
        "description": (
            "Low-kv, high-pole-count BLDC motor optimized for smooth, precise "
            "camera gimbal stabilization. Direct-drive with no gearbox."
        ),
        "aliases": ["gimbal motor", "BGC motor", "stabilizer motor"],
        "keywords": {
            "identity": ["gimbal motor", "stabilizer motor", "BGC motor", "low kv BLDC"],
            "descriptions": [
                "smooth rotation motor", "direct drive gimbal", "camera stabilization motor",
            ],
            "application": [
                "camera gimbal", "antenna tracker", "sensor stabilization",
                "drone gimbal", "telescope pointing",
            ],
            "industry": ["cinematography", "drone", "defense", "astronomy"],
            "components": ["high pole count stator", "neodymium magnets", "AS5048 encoder", "BGC controller"],
            "related": ["gimbal PID", "smooth tracking", "low cogging torque"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "position",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 0.01, "max": 2.0, "typical": 0.2},
            "speed_rpm": {"min": 0, "max": 200, "typical": 30},
            "peak_power_watts": {"min": 1, "max": 50, "typical": 10},
            "continuous_power_watts": {"min": 0.5, "max": 30, "typical": 5},
            "weight_grams": {"min": 10, "max": 300, "typical": 60},
            "cost_usd": {"min": 5, "max": 100, "typical": 20},
            "response_time_ms": {"min": 0.5, "max": 5, "typical": 1},
        },
        "interface": {
            "protocol": ["pwm"],
            "voltage_v": {"min": 7.4, "max": 24, "typical": 12},
            "current_a": {"min": 0.1, "max": 3, "typical": 0.5},
            "feedback": ["encoder_absolute", "hall_effect"],
        },
        "reference_products": [
            {
                "id": "ipower_gbm5208",
                "name": "iPower GBM5208-150T",
                "vendor": "iFlight",
                "cost_usd": 25,
                "torque_nm": 0.3,
            },
        ],
    },
    {
        "id": "motor.stepper_pancake",
        "name": "Pancake Stepper Motor",
        "category": "motor",
        "actuator_type": "stepper",
        "description": (
            "Ultra-flat stepper motor for space-constrained applications. "
            "Short body length at the expense of lower torque."
        ),
        "aliases": ["flat stepper", "thin stepper", "low-profile stepper"],
        "keywords": {
            "identity": ["pancake stepper", "flat stepper", "slim stepper motor"],
            "descriptions": [
                "ultra-thin stepper", "short body stepper", "compact position motor",
            ],
            "application": [
                "optical instrument", "compact 3D printer", "turntable",
                "stage rotation", "miniature positioner",
            ],
            "industry": ["optics", "laboratory", "3D printing", "consumer electronics"],
            "components": ["thin rotor", "stator laminations", "stepper driver"],
            "related": ["holding torque", "step resolution", "form factor"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "position",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "torque_nm": {"min": 0.005, "max": 0.3, "typical": 0.05},
            "speed_rpm": {"min": 0, "max": 1500, "typical": 200},
            "peak_power_watts": {"min": 0.5, "max": 15, "typical": 3},
            "weight_grams": {"min": 20, "max": 200, "typical": 60},
            "cost_usd": {"min": 5, "max": 80, "typical": 15},
        },
        "interface": {
            "protocol": ["step_dir"],
            "voltage_v": {"min": 5, "max": 24, "typical": 12},
            "current_a": {"min": 0.1, "max": 1, "typical": 0.4},
            "feedback": ["none"],
        },
        "reference_products": [],
    },
    {
        "id": "motor.linear_actuator_tubular",
        "name": "Tubular Linear Motor",
        "category": "motor",
        "actuator_type": "linear_actuator",
        "description": (
            "Direct-drive tubular linear motor with no mechanical transmission. "
            "Provides high speed, zero backlash, and maintenance-free operation "
            "for packaging, semiconductor, and pick-and-place machines."
        ),
        "aliases": ["shaft motor", "direct-drive linear motor", "linear servo"],
        "keywords": {
            "identity": ["tubular linear motor", "shaft motor", "ironless linear motor"],
            "descriptions": [
                "direct-drive linear actuator", "zero-backlash linear drive", "cogging-free linear",
            ],
            "application": [
                "pick-and-place", "semiconductor handler", "packaging machine",
                "injection mold ejector", "textile loom",
            ],
            "industry": ["semiconductor", "packaging", "manufacturing", "textile"],
            "components": ["forcer coil", "magnetic rod", "linear encoder", "bearing guide"],
            "related": ["force constant", "continuous force", "duty cycle", "thermal time constant"],
        },
        "classification": {
            "motion_type": "linear",
            "control_type": "position",
            "environment": ["indoor", "cleanroom"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 5, "max": 2000, "typical": 200},
            "speed_mps": {"min": 0.01, "max": 5.0, "typical": 1.0},
            "stroke_mm": {"min": 10, "max": 2000, "typical": 300},
            "position_accuracy_um": {"min": 0.5, "max": 50, "typical": 5},
            "peak_power_watts": {"min": 10, "max": 2000, "typical": 200},
            "continuous_power_watts": {"min": 5, "max": 1000, "typical": 100},
            "weight_grams": {"min": 100, "max": 10000, "typical": 1000},
            "cost_usd": {"min": 300, "max": 15000, "typical": 2000},
        },
        "interface": {
            "protocol": ["ethercat", "analog"],
            "voltage_v": {"min": 24, "max": 400, "typical": 48},
            "current_a": {"min": 0.5, "max": 30, "typical": 5},
            "feedback": ["encoder_incremental", "encoder_absolute"],
        },
        "reference_products": [
            {
                "id": "linmot_ps01_37x120",
                "name": "LinMot PS01-37x120",
                "vendor": "LinMot",
                "cost_usd": 1200,
                "force_n": 255,
                "stroke_mm": 680,
            },
        ],
    },
    # -- hydraulic extra --
    {
        "id": "hydraulic.hydraulic_cylinder_servo",
        "name": "Servo Hydraulic Cylinder",
        "category": "hydraulic",
        "actuator_type": "hydraulic_cylinder",
        "description": (
            "Hydraulic cylinder with integrated servo valve and position transducer "
            "for closed-loop, high-bandwidth position/force control. Used in fatigue "
            "testing, flight simulators, and precision presses."
        ),
        "aliases": ["servo hydraulic actuator", "electrohydraulic actuator", "servo cylinder"],
        "keywords": {
            "identity": ["servo hydraulic cylinder", "electrohydraulic actuator", "servo cylinder"],
            "descriptions": [
                "closed-loop hydraulic actuator", "precision hydraulic cylinder",
                "high-bandwidth hydraulic",
            ],
            "application": [
                "fatigue testing machine", "flight simulator", "shake table",
                "precision press", "active suspension",
            ],
            "industry": ["testing", "simulation", "aerospace", "automotive R&D"],
            "components": ["servo valve", "LVDT", "cylinder", "accumulator", "controller"],
            "related": ["frequency response", "dynamic stiffness", "servo loop", "dither"],
        },
        "classification": {
            "motion_type": "linear",
            "control_type": "position",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 1000, "max": 2000000, "typical": 50000},
            "speed_mps": {"min": 0.01, "max": 2.0, "typical": 0.3},
            "stroke_mm": {"min": 25, "max": 3000, "typical": 250},
            "position_accuracy_um": {"min": 1, "max": 100, "typical": 10},
            "control_rate_hz": {"min": 100, "max": 5000, "typical": 1000},
            "peak_power_watts": {"min": 1000, "max": 500000, "typical": 30000},
            "weight_grams": {"min": 5000, "max": 500000, "typical": 30000},
            "cost_usd": {"min": 2000, "max": 200000, "typical": 20000},
        },
        "interface": {
            "protocol": ["analog", "ethercat"],
            "voltage_v": {"min": 24, "max": 24, "typical": 24},
            "feedback": ["encoder_incremental", "potentiometer"],
        },
        "reference_products": [
            {
                "id": "mts_243",
                "name": "MTS 243 Servo Hydraulic Actuator",
                "vendor": "MTS Systems",
                "cost_usd": 30000,
                "force_n": 100000,
            },
        ],
    },
    # -- pneumatic extra --
    {
        "id": "pneumatic.pneumatic_cylinder_compact",
        "name": "Compact Guided Pneumatic Cylinder",
        "category": "pneumatic",
        "actuator_type": "pneumatic_cylinder",
        "description": (
            "Short-stroke pneumatic cylinder with integrated linear guide for "
            "non-rotating piston rod motion. Ideal for pressing, punching, and "
            "clamping in tight spaces."
        ),
        "aliases": ["guided cylinder", "compact pneumatic", "short stroke air cylinder"],
        "keywords": {
            "identity": ["compact guided cylinder", "short stroke pneumatic", "mini pneumatic"],
            "descriptions": [
                "guided air cylinder", "non-rotating piston", "anti-twist pneumatic",
            ],
            "application": [
                "pressing", "light stamping", "workholding", "labeling", "component insertion",
            ],
            "industry": ["manufacturing", "packaging", "electronics assembly"],
            "components": ["twin piston", "guide rods", "end plate", "cushion"],
            "related": ["non-rotation accuracy", "guide load capacity", "cycle rate"],
        },
        "classification": {
            "motion_type": "linear",
            "control_type": "on_off",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 5, "max": 5000, "typical": 200},
            "speed_mps": {"min": 0.05, "max": 1.5, "typical": 0.3},
            "stroke_mm": {"min": 5, "max": 100, "typical": 25},
            "weight_grams": {"min": 20, "max": 2000, "typical": 150},
            "cost_usd": {"min": 20, "max": 500, "typical": 60},
            "response_time_ms": {"min": 10, "max": 100, "typical": 20},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 24, "max": 24, "typical": 24},
            "feedback": ["hall_effect", "none"],
        },
        "reference_products": [
            {
                "id": "festo_adngf",
                "name": "Festo ADNGF Compact Cylinder",
                "vendor": "Festo",
                "cost_usd": 55,
                "force_n": 314,
                "stroke_mm": 25,
            },
        ],
    },
    # -- gripper extras --
    {
        "id": "gripper.suction_cup_bellows",
        "name": "Bellows Suction Cup",
        "category": "gripper",
        "actuator_type": "suction_cup",
        "description": (
            "Multi-fold bellows suction cup that conforms to curved and uneven surfaces. "
            "Better adaptability than flat cups at the expense of lift force."
        ),
        "aliases": ["bellows cup", "multi-fold suction cup", "flexible suction pad"],
        "keywords": {
            "identity": ["bellows suction cup", "multi-fold cup", "flexible vacuum pad"],
            "descriptions": [
                "curved surface gripper", "conformable suction cup", "height-compensating cup",
            ],
            "application": [
                "curved panel handling", "bag picking", "fruit picking",
                "random bin picking", "blister pack handling",
            ],
            "industry": ["food processing", "packaging", "logistics", "automotive"],
            "components": ["bellows body", "vacuum fitting", "filter screen"],
            "related": ["level compensation", "surface conformity", "cup material durometer"],
        },
        "classification": {
            "motion_type": "gripping",
            "control_type": "on_off",
            "environment": ["indoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "force_n": {"min": 0.5, "max": 200, "typical": 15},
            "payload_kg": {"min": 0.01, "max": 20, "typical": 1.5},
            "peak_power_watts": {"min": 1, "max": 30, "typical": 5},
            "weight_grams": {"min": 3, "max": 200, "typical": 15},
            "cost_usd": {"min": 2, "max": 100, "typical": 10},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 24, "max": 24, "typical": 24},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "piab_b75",
                "name": "Piab B75 Bellows Cup",
                "vendor": "Piab",
                "cost_usd": 12,
            },
        ],
    },
    {
        "id": "gripper.magnetic_gripper_permanent",
        "name": "Permanent Magnet Gripper",
        "category": "gripper",
        "actuator_type": "magnetic_gripper",
        "description": (
            "Gripper using switchable permanent magnets (no power to hold). "
            "Actuates a mechanical linkage to redirect magnetic flux for release. "
            "Fail-safe holding with zero power consumption."
        ),
        "aliases": ["switchable magnet", "permanent magnet lifter", "no-power magnet gripper"],
        "keywords": {
            "identity": ["permanent magnet gripper", "switchable magnet", "magnetic lifter"],
            "descriptions": [
                "zero-power hold gripper", "fail-safe magnetic gripper", "flux-switching gripper",
            ],
            "application": [
                "sheet metal handling", "die change", "mold handling",
                "steel beam lifting", "press tending",
            ],
            "industry": ["metalworking", "stamping", "heavy manufacturing", "tooling"],
            "components": ["neodymium magnets", "flux switch mechanism", "housing", "actuator lever"],
            "related": ["flux path", "residual magnetism", "air gap tolerance", "breakaway force"],
        },
        "classification": {
            "motion_type": "gripping",
            "control_type": "on_off",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "force_n": {"min": 50, "max": 50000, "typical": 1000},
            "payload_kg": {"min": 1, "max": 2000, "typical": 100},
            "peak_power_watts": {"min": 0, "max": 20, "typical": 5},
            "weight_grams": {"min": 200, "max": 50000, "typical": 5000},
            "cost_usd": {"min": 100, "max": 10000, "typical": 1500},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 0, "max": 24, "typical": 24},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "eoat_mge",
                "name": "Schunk MGE Permanent Magnet Gripper",
                "vendor": "SCHUNK",
                "cost_usd": 2000,
                "force_n": 2000,
            },
        ],
    },
    # -- locomotion extras --
    {
        "id": "locomotion.propeller_large",
        "name": "Heavy-Lift Propulsion System",
        "category": "locomotion",
        "actuator_type": "propeller",
        "description": (
            "Large diameter motor-propeller assembly for heavy-lift drones, "
            "cargo UAVs, and eVTOL aircraft. Typically 24-40 inch props with "
            "coaxial or single configurations."
        ),
        "aliases": ["heavy-lift rotor", "cargo drone motor", "large prop drive"],
        "keywords": {
            "identity": ["heavy-lift propeller", "large rotor", "cargo drone propulsion"],
            "descriptions": [
                "high thrust motor", "large diameter prop", "heavy lift UAV drive",
            ],
            "application": [
                "cargo drone", "eVTOL", "agricultural spray drone",
                "heavy-lift multirotor", "aerial survey platform",
            ],
            "industry": ["aerospace", "delivery", "agriculture", "defense", "inspection"],
            "components": [
                "high-torque BLDC", "carbon fiber propeller", "ESC", "vibration damper",
            ],
            "related": [
                "thrust-to-weight", "disc loading", "blade element theory", "tip speed",
            ],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["outdoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 2, "max": 100, "typical": 20},
            "speed_rpm": {"min": 0, "max": 5000, "typical": 1500},
            "peak_power_watts": {"min": 500, "max": 50000, "typical": 5000},
            "continuous_power_watts": {"min": 200, "max": 30000, "typical": 3000},
            "weight_grams": {"min": 500, "max": 15000, "typical": 2000},
            "cost_usd": {"min": 200, "max": 10000, "typical": 1000},
        },
        "interface": {
            "protocol": ["can", "pwm"],
            "voltage_v": {"min": 22, "max": 60, "typical": 48},
            "current_a": {"min": 5, "max": 200, "typical": 50},
            "feedback": ["back_emf"],
        },
        "reference_products": [
            {
                "id": "tmotor_u15ii_kv100",
                "name": "T-Motor U15II KV100",
                "vendor": "T-Motor",
                "cost_usd": 450,
                "peak_power_watts": 6000,
            },
        ],
    },
    {
        "id": "locomotion.jet_thruster_underwater",
        "name": "Underwater Thruster",
        "category": "locomotion",
        "actuator_type": "jet_thruster",
        "description": (
            "Sealed brushless motor with propeller for underwater propulsion. "
            "Used in ROVs, AUVs, and underwater drones. Pressure-rated housings "
            "with magnetic coupling or flooded motor designs."
        ),
        "aliases": ["ROV thruster", "submarine motor", "underwater propulsion"],
        "keywords": {
            "identity": ["underwater thruster", "ROV thruster", "marine thruster"],
            "descriptions": [
                "submersible propulsion", "pressure-rated motor", "marine actuator",
            ],
            "application": [
                "ROV maneuver", "AUV propulsion", "underwater drone",
                "submarine", "diver propulsion vehicle",
            ],
            "industry": ["marine", "defense", "offshore", "research", "aquaculture"],
            "components": [
                "sealed BLDC motor", "propeller", "nozzle", "shaft seal", "potting compound",
            ],
            "related": [
                "bollard thrust", "depth rating", "flooded motor", "magnetic coupling",
            ],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["underwater"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "force_n": {"min": 1, "max": 5000, "typical": 50},
            "peak_power_watts": {"min": 10, "max": 20000, "typical": 300},
            "continuous_power_watts": {"min": 5, "max": 10000, "typical": 150},
            "weight_grams": {"min": 50, "max": 20000, "typical": 500},
            "cost_usd": {"min": 50, "max": 10000, "typical": 300},
            "lifetime_hours": {"min": 2000, "max": 30000, "typical": 10000},
        },
        "interface": {
            "protocol": ["pwm", "can", "rs485"],
            "voltage_v": {"min": 10, "max": 48, "typical": 16},
            "current_a": {"min": 1, "max": 50, "typical": 10},
            "feedback": ["back_emf", "hall_effect"],
        },
        "reference_products": [
            {
                "id": "bluerobotics_t200",
                "name": "Blue Robotics T200 Thruster",
                "vendor": "Blue Robotics",
                "cost_usd": 170,
                "force_n": 52,
            },
        ],
    },
    {
        "id": "locomotion.leg_actuator_series_elastic",
        "name": "Series Elastic Actuator",
        "category": "locomotion",
        "actuator_type": "leg_actuator",
        "description": (
            "Motor with a deliberate spring element in series for force sensing and "
            "shock absorption. Fundamental building block for compliant legged robots "
            "and exoskeletons."
        ),
        "aliases": ["SEA", "elastic actuator", "spring-loaded joint"],
        "keywords": {
            "identity": ["series elastic actuator", "SEA", "compliant joint actuator"],
            "descriptions": [
                "spring-loaded motor", "force-sensing joint", "compliant actuator",
            ],
            "application": [
                "exoskeleton joint", "bipedal walking", "rehabilitation robot",
                "human-safe robot", "prosthetic knee",
            ],
            "industry": ["robotics", "prosthetics", "rehabilitation", "defense", "research"],
            "components": [
                "BLDC motor", "torsion spring", "encoder pair", "harmonic drive",
            ],
            "related": [
                "spring stiffness", "torque transparency", "impedance control", "force bandwidth",
            ],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "torque",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": True,
        },
        "attributes": {
            "torque_nm": {"min": 5, "max": 150, "typical": 40},
            "speed_rpm": {"min": 0, "max": 200, "typical": 50},
            "control_rate_hz": {"min": 500, "max": 5000, "typical": 1000},
            "peak_power_watts": {"min": 50, "max": 3000, "typical": 400},
            "continuous_power_watts": {"min": 20, "max": 1500, "typical": 150},
            "weight_grams": {"min": 300, "max": 8000, "typical": 1500},
            "cost_usd": {"min": 500, "max": 10000, "typical": 2000},
        },
        "interface": {
            "protocol": ["can", "ethercat"],
            "voltage_v": {"min": 24, "max": 48, "typical": 48},
            "current_a": {"min": 2, "max": 30, "typical": 8},
            "feedback": ["encoder_absolute", "strain_gauge"],
        },
        "reference_products": [],
    },
    {
        "id": "locomotion.track_drive_rubber",
        "name": "Rubber Track Drive",
        "category": "locomotion",
        "actuator_type": "track_drive",
        "description": (
            "Compact rubber-track locomotion module for small to medium outdoor robots. "
            "Lightweight alternative to steel tracks with good traction on varied terrain."
        ),
        "aliases": ["rubber track module", "mini track drive", "rubber crawler"],
        "keywords": {
            "identity": ["rubber track drive", "mini crawler", "rubber track module"],
            "descriptions": [
                "lightweight tracked drive", "compact crawler", "all-terrain track",
            ],
            "application": [
                "inspection robot", "agricultural robot", "stair climbing bot",
                "search and rescue robot", "snow robot",
            ],
            "industry": ["robotics", "agriculture", "defense", "inspection"],
            "components": ["rubber belt", "drive sprocket", "idler wheel", "gearmotor", "suspension"],
            "related": ["ground clearance", "traction", "track pitch", "turning radius"],
        },
        "classification": {
            "motion_type": "rotary",
            "control_type": "velocity",
            "environment": ["outdoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "torque_nm": {"min": 0.5, "max": 100, "typical": 10},
            "speed_rpm": {"min": 0, "max": 200, "typical": 50},
            "payload_kg": {"min": 2, "max": 500, "typical": 30},
            "peak_power_watts": {"min": 10, "max": 3000, "typical": 150},
            "weight_grams": {"min": 200, "max": 20000, "typical": 2000},
            "cost_usd": {"min": 30, "max": 3000, "typical": 200},
        },
        "interface": {
            "protocol": ["pwm", "can"],
            "voltage_v": {"min": 12, "max": 48, "typical": 24},
            "current_a": {"min": 1, "max": 30, "typical": 5},
            "feedback": ["encoder_incremental"],
        },
        "reference_products": [],
    },
    # -- fluid extra --
    {
        "id": "fluid.pump_diaphragm",
        "name": "Diaphragm Pump",
        "category": "fluid",
        "actuator_type": "pump",
        "description": (
            "Positive-displacement pump using a flexible diaphragm to move fluid. "
            "Self-priming, can run dry, and handles abrasive or viscous fluids. "
            "Common in chemical dosing and air-operated applications."
        ),
        "aliases": ["membrane pump", "AODD pump", "air-operated diaphragm pump"],
        "keywords": {
            "identity": ["diaphragm pump", "membrane pump", "AODD pump"],
            "descriptions": [
                "self-priming pump", "dry-run safe pump", "pulsating pump",
            ],
            "application": [
                "chemical dosing", "wastewater transfer", "paint circulation",
                "food transfer", "mining slurry",
            ],
            "industry": ["chemical", "water treatment", "food processing", "mining", "pharmaceutical"],
            "components": ["diaphragm", "check valves", "air motor", "manifold"],
            "related": ["pulsation dampener", "stroke frequency", "displacement volume"],
        },
        "classification": {
            "motion_type": "fluid_flow",
            "control_type": "velocity",
            "environment": ["indoor", "outdoor", "hazardous"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 5, "max": 5000, "typical": 200},
            "continuous_power_watts": {"min": 2, "max": 3000, "typical": 100},
            "weight_grams": {"min": 200, "max": 50000, "typical": 3000},
            "cost_usd": {"min": 30, "max": 10000, "typical": 500},
            "lifetime_hours": {"min": 2000, "max": 30000, "typical": 10000},
        },
        "interface": {
            "protocol": ["analog"],
            "voltage_v": {"min": 12, "max": 24, "typical": 24},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "wilden_pro_flo",
                "name": "Wilden Pro-Flo AODD Pump",
                "vendor": "Wilden (PSG Dover)",
                "cost_usd": 1500,
            },
        ],
    },
    # -- display extras --
    {
        "id": "display.speaker_ultrasonic",
        "name": "Ultrasonic Speaker / Emitter",
        "category": "display",
        "actuator_type": "speaker",
        "description": (
            "Transducer emitting ultrasonic sound above 20 kHz for ranging, "
            "gesture recognition, parametric audio, and pest deterrence."
        ),
        "aliases": ["ultrasonic emitter", "parametric speaker", "ultrasonic transducer"],
        "keywords": {
            "identity": ["ultrasonic speaker", "ultrasonic emitter", "parametric speaker"],
            "descriptions": [
                "above-audible transducer", "directional audio", "ultrasonic output",
            ],
            "application": [
                "ultrasonic ranging", "gesture recognition", "directional audio",
                "pest deterrence", "proximity sensing",
            ],
            "industry": ["robotics", "automotive", "consumer electronics", "security"],
            "components": ["piezo disc", "horn", "matching layer", "driver circuit"],
            "related": ["resonance frequency", "beam angle", "SPL at 40kHz", "impedance matching"],
        },
        "classification": {
            "motion_type": "vibration",
            "control_type": "on_off",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 0.01, "max": 10, "typical": 0.5},
            "weight_grams": {"min": 0.5, "max": 50, "typical": 3},
            "cost_usd": {"min": 0.5, "max": 50, "typical": 2},
            "response_time_ms": {"min": 0.01, "max": 1, "typical": 0.1},
            "lifetime_hours": {"min": 30000, "max": 200000, "typical": 100000},
        },
        "interface": {
            "protocol": ["pwm", "analog"],
            "voltage_v": {"min": 3.3, "max": 12, "typical": 5},
            "current_a": {"min": 0.005, "max": 0.5, "typical": 0.05},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "murata_ma40s4s",
                "name": "Murata MA40S4S Ultrasonic Transducer",
                "vendor": "Murata",
                "cost_usd": 2,
            },
        ],
    },
    {
        "id": "display.projector_laser_safety",
        "name": "Laser Safety Zone Projector",
        "category": "display",
        "actuator_type": "projector",
        "description": (
            "Laser projector that casts visible safety zone markings, guidelines, "
            "and warning patterns on floors and surfaces around robots and vehicles."
        ),
        "aliases": ["safety projector", "forklift light", "zone light projector"],
        "keywords": {
            "identity": ["laser safety projector", "zone projector", "safety line projector"],
            "descriptions": [
                "floor marking projector", "dynamic safety zone", "visual warning projector",
            ],
            "application": [
                "robot safety zone", "forklift path marking", "AGV warning",
                "crane work area", "pedestrian warning",
            ],
            "industry": ["logistics", "manufacturing", "warehouse", "construction"],
            "components": ["laser diode", "diffractive optic", "housing", "mounting bracket"],
            "related": ["pattern shape", "laser class", "visibility distance", "IP rating"],
        },
        "classification": {
            "motion_type": "display_output",
            "control_type": "on_off",
            "environment": ["indoor", "outdoor"],
            "reversible": True,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 0.5, "max": 10, "typical": 3},
            "weight_grams": {"min": 50, "max": 500, "typical": 150},
            "cost_usd": {"min": 20, "max": 500, "typical": 80},
            "lifetime_hours": {"min": 10000, "max": 50000, "typical": 30000},
        },
        "interface": {
            "protocol": ["analog", "can"],
            "voltage_v": {"min": 10, "max": 48, "typical": 24},
            "current_a": {"min": 0.05, "max": 0.5, "typical": 0.15},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "banner_wls27",
                "name": "Banner WLS27 Safety Light",
                "vendor": "Banner Engineering",
                "cost_usd": 120,
            },
        ],
    },
    # -- specialty extras --
    {
        "id": "specialty.laser_cutter_co2",
        "name": "CO2 Laser Tube",
        "category": "specialty",
        "actuator_type": "laser_cutter",
        "description": (
            "Gas discharge CO2 laser for cutting and engraving non-metallic materials: "
            "wood, acrylic, fabric, leather, and paper. 10.6 um wavelength."
        ),
        "aliases": ["CO2 laser", "gas laser cutter", "infrared laser tube"],
        "keywords": {
            "identity": ["CO2 laser tube", "gas laser", "10.6 micron laser"],
            "descriptions": [
                "infrared cutting laser", "non-metal laser cutter", "sealed tube laser",
            ],
            "application": [
                "acrylic cutting", "wood engraving", "fabric cutting",
                "rubber cutting", "paper cutting",
            ],
            "industry": ["signage", "craft", "textile", "packaging", "prototyping"],
            "components": ["gas tube", "mirror", "lens", "power supply", "water cooling"],
            "related": ["beam mode", "tube life", "water cooling", "mirror alignment"],
        },
        "classification": {
            "motion_type": "display_output",
            "control_type": "proportional",
            "environment": ["indoor"],
            "reversible": False,
            "backdrivable": False,
        },
        "attributes": {
            "peak_power_watts": {"min": 20, "max": 300, "typical": 80},
            "continuous_power_watts": {"min": 10, "max": 250, "typical": 60},
            "weight_grams": {"min": 500, "max": 10000, "typical": 3000},
            "cost_usd": {"min": 50, "max": 3000, "typical": 300},
            "lifetime_hours": {"min": 1000, "max": 10000, "typical": 4000},
        },
        "interface": {
            "protocol": ["pwm", "analog"],
            "voltage_v": {"min": 20000, "max": 35000, "typical": 25000},
            "feedback": ["none"],
        },
        "reference_products": [
            {
                "id": "reci_w2",
                "name": "RECI W2 90W CO2 Laser Tube",
                "vendor": "RECI",
                "cost_usd": 200,
                "peak_power_watts": 90,
            },
        ],
    },
]


def write_actuator(actuator: dict) -> Path:
    """Write a single actuator definition to its YAML file."""
    cat = actuator["category"]
    # Use the part after the dot in the id as the filename
    filename = actuator["id"].split(".", 1)[1]
    outdir = ACTUATOR_DIR / cat
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{filename}.yaml"

    # Build ordered dict for clean YAML output
    data = {
        "id": actuator["id"],
        "name": actuator["name"],
        "category": actuator["category"],
        "actuator_type": actuator["actuator_type"],
        "description": actuator["description"],
        "aliases": actuator.get("aliases", []),
        "keywords": actuator.get("keywords", {}),
        "classification": actuator.get("classification", {}),
        "attributes": actuator.get("attributes", {}),
    }
    if "interface" in actuator:
        data["interface"] = actuator["interface"]
    data["reference_products"] = actuator.get("reference_products", [])

    with open(outpath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=100)

    return outpath


def main():
    print(f"Generating {len(ACTUATORS)} actuator YAML files...")
    categories = set()
    for act in ACTUATORS:
        path = write_actuator(act)
        categories.add(act["category"])
        print(f"  {path.relative_to(ROOT)}")

    print(f"\nDone: {len(ACTUATORS)} files across {len(categories)} categories")
    print(f"Categories: {', '.join(sorted(categories))}")

    # Validate keyword count
    low_keyword = []
    for act in ACTUATORS:
        kw = act.get("keywords", {})
        total = sum(len(v) for v in kw.values())
        if total < 15:
            low_keyword.append((act["id"], total))
    if low_keyword:
        print("\nWARNING: Actuators with < 15 keywords:")
        for aid, count in low_keyword:
            print(f"  {aid}: {count} keywords")
    else:
        print("\nAll actuators have >= 15 keywords.")


if __name__ == "__main__":
    main()
