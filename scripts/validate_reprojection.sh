#!/bin/bash
# Reprojection Validation Script

set -e

# Default parameters
RESULTS_DIR="results/V2_1"
STAGE1_DIR="$RESULTS_DIR/stage1_detection"
STAGE2_DIR="$RESULTS_DIR/stage2_calibration"
STAGE3_DIR="$RESULTS_DIR/stage3_localization_3d"
STAGE4_DIR="$RESULTS_DIR/stage4_validation"
METHOD="bezier"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results) RESULTS_DIR="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        *)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --results DIR    Results directory"
            echo "  --method METHOD  Trajectory method (best, polynomial, bezier, exponential, mu_s)"
            exit 1
            ;;
    esac
done

echo "Reprojection Validation (method: $METHOD)"

# Check required files
if [ ! -f "$STAGE3_DIR/ball_3d_positions_fitted.csv" ]; then
    echo "Error: 3D positions not found. Run: ./scripts/localize_3d.sh"
    exit 1
fi

if [ ! -f "$STAGE2_DIR/camera_calibration_smoothed_method1.csv" ]; then
    echo "Error: Camera calibration not found"
    exit 1
fi

if [ ! -f "$STAGE1_DIR/detections_smoothed_method1_diameter.csv" ]; then
    echo "Error: Detections not found"
    exit 1
fi

echo "Running validation..."
python3 src/validate_reprojection.py \
    --positions "$STAGE3_DIR/ball_3d_positions_fitted.csv" \
    --calibration "$STAGE2_DIR/camera_calibration_smoothed_method1.csv" \
    --detections "$STAGE1_DIR/detections_smoothed_method1_diameter.csv" \
    --output "$STAGE4_DIR/reprojection_validation_${METHOD}.csv" \
    --method "$METHOD" \
    --plot \
    --plot-output "$STAGE4_DIR"

echo ""
echo "✓ Validation complete"
echo "Output: $STAGE4_DIR/reprojection_validation_${METHOD}.csv"
echo "Plots: $STAGE4_DIR/"
