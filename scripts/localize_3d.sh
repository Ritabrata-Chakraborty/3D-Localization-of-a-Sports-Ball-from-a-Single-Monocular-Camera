#!/bin/bash
# 3D Ball Localization Pipeline

set -e

# Default parameters
FRAMES_DIR="downloads/frames/V2_1"
RESULTS_DIR="results/V2_1"
STAGE1_DIR="$RESULTS_DIR/stage1_detection"
STAGE2_DIR="$RESULTS_DIR/stage2_calibration"
STAGE3_DIR="$RESULTS_DIR/stage3_localization_3d"
BALL_DIAMETER=0.22
FRAME_RATE=30.0
MAX_VELOCITY=40.0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --frames) FRAMES_DIR="$2"; shift 2 ;;
        --results) RESULTS_DIR="$2"; shift 2 ;;
        --ball-diameter) BALL_DIAMETER="$2"; shift 2 ;;
        --frame-rate) FRAME_RATE="$2"; shift 2 ;;
        --max-velocity) MAX_VELOCITY="$2"; shift 2 ;;
        *)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --frames DIR          Frames directory"
            echo "  --results DIR         Results directory"
            echo "  --ball-diameter M     Ball diameter in meters"
            echo "  --frame-rate FPS      Frame rate"
            echo "  --max-velocity M/S    Max velocity"
            exit 1
            ;;
    esac
done

echo "3D Ball Localization Pipeline"
echo "Ball diameter: ${BALL_DIAMETER}m | Frame rate: ${FRAME_RATE} fps | Max velocity: ${MAX_VELOCITY} m/s"

# Check required files
if [ ! -f "$STAGE2_DIR/camera_calibration_smoothed_method1.csv" ]; then
    echo "Error: Camera calibration not found. Run: ./scripts/calibrate.sh calibrate && ./scripts/calibrate.sh smooth method1"
    exit 1
fi

if [ ! -f "$STAGE1_DIR/detections_smoothed_method1_diameter.csv" ]; then
    echo "Error: Detections not found. Run: ./scripts/detect.sh detect && ./scripts/detect.sh smooth 1"
    exit 1
fi

echo "Running 3D localization..."
python3 src/localize_ball_3d.py \
    --calibration "$STAGE2_DIR/camera_calibration_smoothed_method1.csv" \
    --detections "$STAGE1_DIR/detections_smoothed_method1_diameter.csv" \
    --output "$STAGE3_DIR/ball_3d_positions.csv" \
    --ball-diameter $BALL_DIAMETER \
    --frame-rate $FRAME_RATE \
    --max-velocity $MAX_VELOCITY \
    --apply-physics \
    --apply-temporal \
    --plot \
    --plot-output "$STAGE3_DIR"

echo ""
echo "✓ 3D localization complete"
echo "Output: $STAGE3_DIR/ball_3d_positions.csv"
echo "Fitted: $STAGE3_DIR/ball_3d_positions_fitted.csv"
echo "Next: ./scripts/validate_reprojection.sh"
