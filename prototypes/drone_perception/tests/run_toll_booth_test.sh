#!/bin/bash
# Toll Booth Traffic Video Benchmark Suite
# Runs the full pipeline on three different video sizes

# Set UTF-8 encoding to handle Unicode characters in Python output
export PYTHONIOENCODING=utf-8

# Ensure we're in the drone_perception root directory
cd "$(dirname "$0")/.."

# Common parameters
MODEL="s"
CLASSES="0 1 2 3 4 5 6 7 8 9"
VIDEO_DIR="test_data/videos/traffic"

echo "========================================"
echo "Toll Booth Test Suite"
echo "========================================"
echo "Model: yolov8${MODEL}"
echo "Classes: ${CLASSES}"
echo ""

# Test 1: Tiny video
echo "[1/3] Processing tiny video..."
python examples/full_pipeline.py \
    --video "${VIDEO_DIR}/247589_tiny.mp4" \
    --model "${MODEL}" \
    --classes ${CLASSES} \
    --save-video "${VIDEO_DIR}/247589_tiny_output.mp4" \
    --no-viz-3d \
    > "${VIDEO_DIR}/247589_tiny.log" 2>&1
echo "  ✓ Completed. Log: ${VIDEO_DIR}/247589_tiny.log"

# Test 2: Small video
echo "[2/3] Processing small video..."
python examples/full_pipeline.py \
    --video "${VIDEO_DIR}/247589_small.mp4" \
    --model "${MODEL}" \
    --classes ${CLASSES} \
    --save-video "${VIDEO_DIR}/247589_small_output.mp4" \
    --no-viz-3d \
    > "${VIDEO_DIR}/247589_small.log" 2>&1
echo "  ✓ Completed. Log: ${VIDEO_DIR}/247589_small.log"

# Test 3: Medium video
echo "[3/3] Processing medium video..."
python examples/full_pipeline.py \
    --video "${VIDEO_DIR}/247589_medium.mp4" \
    --model "${MODEL}" \
    --classes ${CLASSES} \
    --save-video "${VIDEO_DIR}/247589_medium_output.mp4" \
    --no-viz-3d \
    > "${VIDEO_DIR}/247589_medium.log" 2>&1
echo "  ✓ Completed. Log: ${VIDEO_DIR}/247589_medium.log"

echo ""
echo "========================================"
echo "All tests completed!"
echo "========================================"
echo "Check the logs in ${VIDEO_DIR}/ for details"
