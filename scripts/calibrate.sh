#!/bin/bash
# Camera Calibration Wrapper Script
# Modes: calibrate, smooth, visualize

if [ $# -eq 0 ]; then
    echo "Camera Calibration System"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  calibrate  - Extract camera parameters"
    echo "  smooth     - Smooth camera parameters"
    echo "  visualize  - Generate visualization"
    echo ""
    echo "Smooth Options:"
    echo "  method1 - Global polynomial (default)"
    echo "  method2 - Piecewise polynomial"
    echo ""
    exit 1
fi

COMMAND=$1
SMOOTH_METHOD=${2:-method1}

# Configuration
FRAMES_DIR="./downloads/frames/V2_1"
RESULTS_DIR="results/V2_1"
STAGE2_DIR="$RESULTS_DIR/stage2_calibration"
WEIGHTS_KP="PnLCalib/SV_kp"
WEIGHTS_LINE="PnLCalib/SV_lines"
DEVICE="cuda:0"

if [ "$COMMAND" == "calibrate" ]; then
    echo "Running camera calibration..."
    python3 src/calibrate.py --mode calibrate \
        --frames "$FRAMES_DIR" \
        --results "$STAGE2_DIR" \
        --weights_kp "$WEIGHTS_KP" \
        --weights_line "$WEIGHTS_LINE" \
        --device "$DEVICE"
    
elif [ "$COMMAND" == "smooth" ]; then
    echo "Smoothing camera parameters (method: $SMOOTH_METHOD)..."
    python3 src/calibrate.py --mode smooth \
        --frames "$FRAMES_DIR" \
        --results "$STAGE2_DIR" \
        --weights_kp "$WEIGHTS_KP" \
        --weights_line "$WEIGHTS_LINE" \
        --device "$DEVICE" \
        --smooth-method "$SMOOTH_METHOD"

elif [ "$COMMAND" == "visualize" ]; then
    echo "Generating visualization..."
    python3 src/calibrate.py --mode visualize --results "$STAGE2_DIR"
    
else
    echo "Error: Invalid command '$COMMAND'"
    exit 1
fi
