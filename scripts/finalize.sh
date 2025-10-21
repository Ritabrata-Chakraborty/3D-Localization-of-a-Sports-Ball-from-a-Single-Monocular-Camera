#!/bin/bash
# Stage 5: Finalization Script
# Generates annotated video frames and comprehensive visualizations

set -e

# Default parameters
RESULTS_DIR="results/V2_2"
FRAMES_DIR="downloads/frames/V2_2"
DET_METHOD=2
CAL_METHOD=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --detection-method)
            DET_METHOD="$2"
            shift 2
            ;;
        --calibration-method)
            CAL_METHOD="$2"
            shift 2
            ;;
        --frames)
            FRAMES_DIR="$2"
            shift 2
            ;;
        --results)
            RESULTS_DIR="$2"
            shift 2
            ;;
        [1-3])
            # Positional argument: detection method
            DET_METHOD="$1"
            shift 1
            ;;
        [1-2])
            # Positional argument: calibration method
            CAL_METHOD="$1"
            shift 1
            ;;
        *)
            echo "Usage: $0 [DETECTION_METHOD [CALIBRATION_METHOD]] [OPTIONS]"
            echo ""
            echo "Positional Arguments:"
            echo "  DETECTION_METHOD <1|2|3>      Detection smoothing method (default: 2)"
            echo "  CALIBRATION_METHOD <1|2>      Calibration smoothing method (default: 1)"
            echo ""
            echo "Named Options:"
            echo "  --detection-method <1|2|3>    Detection smoothing method"
            echo "  --calibration-method <1|2>    Calibration smoothing method"
            echo "  --frames DIR                  Frames directory"
            echo "  --results DIR                 Results directory"
            echo ""
            echo "Examples:"
            echo "  $0                                        # Use defaults (det=2, cal=1)"
            echo "  $0 1 1                                    # Use det=1, cal=1"
            echo "  $0 --detection-method 1 --calibration-method 1"
            exit 1
            ;;
    esac
done

# Validate methods
if [ "$DET_METHOD" != "1" ] && [ "$DET_METHOD" != "2" ] && [ "$DET_METHOD" != "3" ]; then
    echo "Error: Invalid detection method '$DET_METHOD'. Must be 1, 2, or 3."
    exit 1
fi

if [ "$CAL_METHOD" != "1" ] && [ "$CAL_METHOD" != "2" ]; then
    echo "Error: Invalid calibration method '$CAL_METHOD'. Must be 1 or 2."
    exit 1
fi

# Set paths
STAGE1_DIR="$RESULTS_DIR/stage1_detection"
STAGE2_DIR="$RESULTS_DIR/stage2_calibration"
STAGE3_DIR="$RESULTS_DIR/stage3_localization_3d"
STAGE5_DIR="$RESULTS_DIR/stage5_finalization"

echo "Stage 5: Finalization"
echo "Detection method: $DET_METHOD | Calibration method: $CAL_METHOD"
echo ""

# Check required files
DETECTION_CSV="$STAGE1_DIR/detections_smoothed_method${DET_METHOD}_diameter.csv"
CALIBRATION_CSV="$STAGE2_DIR/camera_calibration_smoothed_method${CAL_METHOD}.csv"
CALIBRATION_RAW_CSV="$STAGE2_DIR/camera_calibration.csv"
BALL_3D_CSV="$STAGE3_DIR/ball_3d_positions.csv"
BALL_3D_FITTED_CSV="$STAGE3_DIR/ball_3d_positions_fitted.csv"

if [ ! -f "$DETECTION_CSV" ]; then
    echo "Error: Detection CSV not found: $DETECTION_CSV"
    echo "Run: ./scripts/detect.sh detect && ./scripts/detect.sh smooth $DET_METHOD"
    exit 1
fi

if [ ! -f "$CALIBRATION_CSV" ]; then
    echo "Error: Calibration CSV not found: $CALIBRATION_CSV"
    echo "Run: ./scripts/calibrate.sh calibrate && ./scripts/calibrate.sh smooth method$CAL_METHOD"
    exit 1
fi

if [ ! -f "$CALIBRATION_RAW_CSV" ]; then
    echo "Error: Raw calibration CSV not found: $CALIBRATION_RAW_CSV"
    echo "Run: ./scripts/calibrate.sh calibrate"
    exit 1
fi

if [ ! -f "$BALL_3D_CSV" ]; then
    echo "Error: 3D ball positions CSV not found: $BALL_3D_CSV"
    echo "Run: ./scripts/localize_3d.sh"
    exit 1
fi

if [ ! -f "$BALL_3D_FITTED_CSV" ]; then
    echo "Error: Fitted 3D ball positions CSV not found: $BALL_3D_FITTED_CSV"
    echo "Run: ./scripts/localize_3d.sh"
    exit 1
fi

if [ ! -d "$FRAMES_DIR" ]; then
    echo "Error: Frames directory not found: $FRAMES_DIR"
    exit 1
fi

echo "Running finalization..."
python3 src/finalize.py \
    --frames "$FRAMES_DIR" \
    --detection-csv "$DETECTION_CSV" \
    --calibration-csv "$CALIBRATION_CSV" \
    --calibration-raw-csv "$CALIBRATION_RAW_CSV" \
    --ball-3d-csv "$BALL_3D_CSV" \
    --ball-3d-fitted-csv "$BALL_3D_FITTED_CSV" \
    --output "$STAGE5_DIR" \
    --weights-kp "PnLCalib/SV_kp" \
    --weights-line "PnLCalib/SV_lines"

echo ""
echo "✓ Stage 5 complete"
echo "Output: $STAGE5_DIR"
echo "  - final_video.mp4"
echo "  - final_positions.csv"
echo "  - trajectories_3d.png"
echo "  - annotated_frames/"

