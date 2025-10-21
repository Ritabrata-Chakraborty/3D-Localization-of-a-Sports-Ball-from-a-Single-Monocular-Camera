#!/bin/bash
# Ball Detection Wrapper Script
# Modes: detect, correct, smooth, visualize

if [ $# -eq 0 ]; then
    echo "Ball Detection System"
    echo ""
    echo "Usage: $0 <command> [method] [smooth-type]"
    echo ""
    echo "Commands:"
    echo "  detect                  - Run ball detection"
    echo "  correct                 - Manually correct bounding boxes"
    echo "  smooth [method] [type]  - Smooth diameter/center"
    echo "  visualize               - Generate visualization"
    echo ""
    echo "Smoothing Methods:"
    echo "  1 - Piecewise Quadratic"
    echo "  2 - Piecewise Cubic Spline (default)"
    echo "  3 - Bidirectional Exponential"
    echo ""
    echo "Smooth Types:"
    echo "  diameter - Smooth diameter only (default)"
    echo "  center   - Smooth center positions only"
    echo "  both     - Smooth both"
    echo ""
    exit 1
fi

COMMAND=$1
SMOOTH_METHOD=${2:-2}
SMOOTH_TYPE=${3:-diameter}

# Configuration
FRAMES_DIR="./downloads/frames/V2_1"
RESULTS_DIR="results/V2_1"
STAGE1_DIR="$RESULTS_DIR/stage1_detection"
STAGE2_DIR="$RESULTS_DIR/stage2_calibration"
YOLO_MODEL="YOLO/output/yolo_training/weights/best.pt"
CALIBRATION_CSV="$STAGE2_DIR/camera_calibration_smoothed_method2.csv"

if [ "$COMMAND" == "detect" ]; then
    echo "Running ball detection..."
    python3 src/detect_ball.py --mode detect --frames "$FRAMES_DIR" --output "$STAGE1_DIR" --yolo "$YOLO_MODEL"
    
elif [ "$COMMAND" == "correct" ]; then
    echo "Starting interactive correction mode..."
    python3 src/detect_ball.py --mode correct --frames "$FRAMES_DIR" --output "$STAGE1_DIR" --yolo "$YOLO_MODEL"
    
elif [ "$COMMAND" == "smooth" ]; then
    if [ "$SMOOTH_TYPE" != "diameter" ] && [ "$SMOOTH_TYPE" != "center" ] && [ "$SMOOTH_TYPE" != "both" ]; then
        echo "Error: Invalid smooth type '$SMOOTH_TYPE'"
        exit 1
    fi
    
    if [ "$SMOOTH_METHOD" != "1" ] && [ "$SMOOTH_METHOD" != "2" ] && [ "$SMOOTH_METHOD" != "3" ]; then
        echo "Error: Invalid method '$SMOOTH_METHOD'"
        exit 1
    fi
    
    echo "Smoothing with method $SMOOTH_METHOD (type: $SMOOTH_TYPE)..."
    
    if [ -f "$CALIBRATION_CSV" ]; then
        python3 src/detect_ball.py --mode smooth --frames "$FRAMES_DIR" --output "$STAGE1_DIR" \
            --yolo "$YOLO_MODEL" --calibration "$CALIBRATION_CSV" --smooth-method "$SMOOTH_METHOD" \
            --smooth-type "$SMOOTH_TYPE"
    else
        python3 src/detect_ball.py --mode smooth --frames "$FRAMES_DIR" --output "$STAGE1_DIR" \
            --yolo "$YOLO_MODEL" --smooth-method "$SMOOTH_METHOD" --smooth-type "$SMOOTH_TYPE"
    fi

elif [ "$COMMAND" == "visualize" ]; then
    echo "Generating visualization..."
    python3 src/detect_ball.py --mode visualize --output "$STAGE1_DIR"

else
    echo "Error: Invalid command '$COMMAND'"
    exit 1
fi
