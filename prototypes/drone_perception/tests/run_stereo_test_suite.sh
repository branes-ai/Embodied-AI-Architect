#!/bin/bash
# Stereo Pipeline Test Suite
# Tests stereo pipeline with synthetic depth maps from MiDaS
# Compares results against monocular pipeline

# Set UTF-8 encoding to handle Unicode characters in Python output
export PYTHONIOENCODING=utf-8

# Ensure we're in the drone_perception root directory
cd "$(dirname "$0")/.."

# Common parameters
MODEL="s"
CLASSES="0 1 2 3 4 5 6 7 8 9"
VIDEO_DIR="test_data/videos/traffic"

echo "========================================"
echo "Stereo Pipeline Test Suite"
echo "========================================"
echo "Model: yolov8${MODEL}"
echo "Classes: ${CLASSES}"
echo ""

# Check if depth maps exist, generate if not
echo "Checking for depth maps..."
if [ ! -f "${VIDEO_DIR}/247589_tiny_depth.mp4" ]; then
    echo ""
    echo "Depth maps not found. Generating using MiDaS..."
    echo "This will take a few minutes on first run..."
    echo ""

    # Check if torch is installed
    python -c "import torch" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: PyTorch not installed!"
        echo "Install with: pip install torch torchvision"
        echo ""
        echo "Alternatively, download pre-generated depth maps from:"
        echo "  https://github.com/[your-repo]/releases/depth-maps"
        exit 1
    fi

    # Generate depth maps
    python scripts/generate_depth_maps.py --batch toll_booth

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate depth maps"
        exit 1
    fi
    echo ""
fi

echo "✓ Depth maps available"
echo ""

# Test 1: Tiny video (stereo)
echo "[1/3] Processing tiny video (stereo mode)..."
python demos/stereo_pipeline.py \
    --backend recorded \
    --rgb-video "${VIDEO_DIR}/247589_tiny.mp4" \
    --depth-video "${VIDEO_DIR}/247589_tiny_depth.mp4" \
    --model "${MODEL}" \
    --classes ${CLASSES} \
    --save-video "${VIDEO_DIR}/247589_tiny_stereo_output.mp4" \
    --no-viz-3d \
    --depth-method median \
    > "${VIDEO_DIR}/247589_tiny_stereo.log" 2>&1
echo "  ✓ Completed. Log: ${VIDEO_DIR}/247589_tiny_stereo.log"

# Test 2: Small video (stereo)
echo "[2/3] Processing small video (stereo mode)..."
python demos/stereo_pipeline.py \
    --backend recorded \
    --rgb-video "${VIDEO_DIR}/247589_small.mp4" \
    --depth-video "${VIDEO_DIR}/247589_small_depth.mp4" \
    --model "${MODEL}" \
    --classes ${CLASSES} \
    --save-video "${VIDEO_DIR}/247589_small_stereo_output.mp4" \
    --no-viz-3d \
    --depth-method median \
    > "${VIDEO_DIR}/247589_small_stereo.log" 2>&1
echo "  ✓ Completed. Log: ${VIDEO_DIR}/247589_small_stereo.log"

# Test 3: Medium video (stereo)
echo "[3/3] Processing medium video (stereo mode)..."
python demos/stereo_pipeline.py \
    --backend recorded \
    --rgb-video "${VIDEO_DIR}/247589_medium.mp4" \
    --depth-video "${VIDEO_DIR}/247589_medium_depth.mp4" \
    --model "${MODEL}" \
    --classes ${CLASSES} \
    --save-video "${VIDEO_DIR}/247589_medium_stereo_output.mp4" \
    --no-viz-3d \
    --depth-method median \
    > "${VIDEO_DIR}/247589_medium_stereo.log" 2>&1
echo "  ✓ Completed. Log: ${VIDEO_DIR}/247589_medium_stereo.log"

echo ""
echo "========================================"
echo "All stereo tests completed!"
echo "========================================"
echo ""
echo "Results:"
echo "  Stereo outputs: ${VIDEO_DIR}/*_stereo_output.mp4"
echo "  Monocular outputs: ${VIDEO_DIR}/*_output.mp4"
echo "  Logs: ${VIDEO_DIR}/*.log"
echo ""
echo "Compare stereo vs monocular:"
echo "  - Stereo should have more accurate depth"
echo "  - Stereo should have smoother 3D trajectories"
echo "  - Velocity estimates should be more consistent"
