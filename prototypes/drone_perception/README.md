# Drone Perception Pipeline

A progressive perception pipeline for drone-based object detection, tracking, 3D reasoning, and situational awareness.

📋 **[CHANGELOG](CHANGELOG.md)** | 📝 **[Latest Session Log](../../docs/sessions/2025-11-21-drone-perception-phase3-tracking-improvements.md)**

## Quick Start

### Installation

```bash
cd prototypes/drone_perception
pip install -r requirements.txt
```

### Run the Application

```bash
# Deployable pipeline (headless by default)
python app/main.py --sensor mono --video 0
python app/main.py --sensor mono --video traffic.mp4 --output json
python app/main.py --sensor stereo --reasoning --display
```

### Run Examples & Demos

```bash
# Examples (basic usage patterns)
python examples/full_pipeline.py --video 0
python examples/simple_detection.py --video your_video.mp4

# Demos (advanced sensor showcases)
python demos/stereo_pipeline.py --backend realsense --model s
python demos/reasoning_pipeline.py --camera 0 --model s --prediction-horizon 3.0
python demos/wide_angle_pipeline.py --camera 0
python demos/lidar_camera_pipeline.py --camera 0

# Advanced: GPU + specific classes + save output
python examples/full_pipeline.py \
    --video test.mp4 \
    --device cuda \
    --model s \
    --classes 0 2 7 \
    --save-video output.mp4
```

**See [QUICKSTART.md](QUICKSTART.md) for detailed instructions!**

**NEW: [Phase 3 Reasoning Documentation](docs/phase3_reasoning.md)** - Trajectory prediction, collision detection, spatial analysis, and behavior classification

## Progressive Sensor Support

This pipeline is designed to work with three levels of sensor complexity:

### Level 1: Monocular Camera
- **Input**: Video file or webcam
- **Depth**: Estimated via heuristics or MiDaS
- **Use Case**: Development, testing, recorded data
- **Status**: 🚧 In Progress

### Level 2: Stereo Camera
- **Input**: RealSense D435, OAK-D
- **Depth**: Stereo depth map
- **Use Case**: Metric tracking, velocity estimation
- **Status**: ✅ Complete

### Level 3: LiDAR + Camera
- **Input**: Livox/Velodyne + Camera
- **Depth**: 3D point cloud
- **Use Case**: Industrial deployment, high accuracy
- **Status**: 📋 Planned

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design.

```
Camera → Detection → Tracking → Scene Graph → Reasoning → Visualization
         (YOLOv8)   (ByteTrack)  (Kalman)      (Phase 3)   (3D Plot)
                                                   ↓
                                    ┌──────────────┴──────────────┐
                                    │  - Trajectory Prediction    │
                                    │  - Collision Detection      │
                                    │  - Spatial Analysis         │
                                    │  - Behavior Classification  │
                                    └─────────────────────────────┘
```

## Features

### Perception (Phase 1 & 2)
- ✅ **Object detection** with YOLOv8 (nano to xlarge models)
- ✅ **Multi-object tracking** with ByteTrack (ID persistence, re-identification)
- ✅ **3D scene graph** with position/velocity/acceleration estimation
- ✅ **Kalman filtering** for smooth state estimation (9D state per object)
- ✅ **Real-time 3D visualization** with matplotlib (position, velocity, trajectories)
- ✅ **Sensor abstraction** ready for monocular → stereo → LiDAR progression

### 3D Reasoning & Planning (Phase 3) - NEW!
- ✅ **Trajectory prediction** - Constant velocity, acceleration, and physics-based models
- ✅ **Collision detection** - Time-to-collision with 5-level risk assessment
- ✅ **Spatial analysis** - Relative positioning, proximity detection, clustering
- ✅ **Behavior classification** - Stationary, moving, turning, accelerating, approaching
- ✅ **Real-time visualization** - Predicted trajectories with color-coded risk levels

### Coming Soon
- 📋 HDF5 recording for replay
- 📋 LiDAR sensor support

## Project Structure

```
drone_perception/
├── app/                        # APPLICATION — deployable pipeline
│   └── main.py                # Configurable drone perception app
├── examples/                   # EXAMPLES — basic usage patterns
│   ├── simple_detection.py    # Minimal: camera → YOLO → display
│   └── full_pipeline.py       # Standard: camera → detect → track → 3D
├── demos/                      # DEMOS — advanced sensor showcases
│   ├── stereo_pipeline.py     # Stereo camera (RealSense/OAK-D)
│   ├── wide_angle_pipeline.py # Fisheye camera + DAC depth
│   ├── lidar_camera_pipeline.py # LiDAR + camera fusion
│   └── reasoning_pipeline.py  # 3D reasoning (trajectory, collision, behavior)
├── scripts/                    # UTILITIES — dev tools, data prep
│   ├── calibrate_lidar_camera.py
│   ├── compare_mono_vs_stereo.py
│   ├── download_test_videos.py
│   └── generate_depth_maps.py
├── tests/                      # TESTS — test runners & validation
│   ├── run_test_suite.py
│   ├── validate_stereo_accuracy.py
│   ├── run_stereo_test_suite.sh
│   └── run_toll_booth_test.sh
├── lib/                        # LIBRARY — pipeline layers (importable)
│   ├── common.py              # Shared data models (Frame, Detection, Track, etc.)
│   ├── sensors/               # Acquisition layer
│   │   ├── monocular.py       # Video/webcam
│   │   ├── stereo.py          # RealSense/OAK-D
│   │   ├── wide_angle.py      # Fisheye cameras + DAC
│   │   └── lidar.py           # LiDAR fusion
│   ├── detection/             # Detection layer
│   │   └── yolo.py            # YOLOv8 wrapper
│   ├── tracking/              # Tracking layer
│   │   ├── bytetrack.py       # ByteTrack implementation
│   │   └── kalman_filter.py   # Kalman box filter
│   ├── scene_graph/           # 3D state layer
│   │   └── manager.py         # Scene graph with Kalman filtering
│   ├── reasoning/             # Reasoning layer
│   │   ├── trajectory_predictor.py
│   │   ├── collision_detector.py
│   │   ├── spatial_analyzer.py
│   │   └── behavior_classifier.py
│   └── visualization/         # Output layer
│       └── live_view.py       # Real-time 3D plot
├── docs/                       # Documentation
├── test_data/                  # Test videos and annotations
└── requirements.txt
```

**Note:** Development session logs are maintained in `../../docs/sessions/`

## Development Status

### ✅ Phase 1: Monocular Pipeline (COMPLETE)
- [x] Project structure and architecture
- [x] Sensor abstraction layer (monocular camera)
- [x] YOLOv8 detection integration
- [x] ByteTrack multi-object tracking
- [x] 3D scene graph with Kalman filtering
- [x] Real-time 3D visualization
- [x] Full end-to-end example

### ✅ Phase 2: Multi-Sensor Support (COMPLETE)
- [x] RealSense D435 integration
- [x] OAK-D support
- [x] Wide-angle/fisheye camera support
- [x] Depth Any Camera (DAC) integration
- [x] Depth map fusion
- [x] Metric accuracy validation
- [x] Stereo pipeline example
- [x] Updated full_pipeline.py with --stereo flag

### ✅ Phase 3: 3D Reasoning & Planning (COMPLETE - Nov 2025)
- [x] Trajectory prediction (constant velocity, acceleration, physics-based)
- [x] Collision detection with risk assessment
- [x] Spatial analysis (relative positioning, proximity)
- [x] Behavior classification (stationary, moving, turning, etc.)
- [x] Real-time reasoning pipeline example
- [x] Comprehensive documentation (docs/phase3_reasoning.md)
- [x] Performance optimizations (frame skipping, reduced resolution)
- [x] Enhanced tracking (improved re-identification, object pruning)

### 📋 Phase 4: Recording & Replay
- [ ] HDF5 data recording
- [ ] Replay viewer with timeline
- [ ] Export to common formats

### 📋 Phase 5: Production Ready
- [ ] LiDAR sensor support (Livox/Velodyne)
- [ ] Multi-rate framework integration
- [ ] Performance optimization (30+ FPS on edge devices)
- [ ] Unit tests and CI/CD
- [ ] Docker deployment

## References

- Research: `../../docs/research/drone-pipeline.md`
- Multi-Rate Framework: `../multi_rate_framework/`
- ByteTrack: https://github.com/ifzhang/ByteTrack
- YOLOv8: https://github.com/ultralytics/ultralytics
