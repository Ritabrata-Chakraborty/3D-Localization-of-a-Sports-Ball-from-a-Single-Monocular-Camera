#!/bin/bash
#
# Simple Ball Detection Wrapper Script
#
# MODE 1: detect  - Run YOLO + Hough Circle detection
# MODE 2: correct - Manually correct bounding boxes
# MODE 3: smooth  - Smooth ball parameters across frames
#

# Check if command argument is provided
if [ $# -eq 0 ]; then
    echo "Simple Ball Detection System"
    echo ""
    echo "Usage: $0 <command> [window_size]"
    echo ""
    echo "Commands:"
    echo "  detect   - Run ball detection (YOLO + Hough Circle)"
    echo "  correct  - Manually correct bounding boxes"
    echo "  smooth   - Smooth ball parameters (optional: window_size, default=5)"
    echo ""
    echo "Examples:"
    echo "  ./detect.sh detect      # Run detection on frames"
    echo "  ./detect.sh smooth      # Smooth with default window (5)"
    echo "  ./detect.sh smooth 7    # Smooth with window size 7"
    echo "  ./detect.sh correct     # Correct detections interactively"
    echo ""
    exit 1
fi

COMMAND=$1
WINDOW_SIZE=${2:-5}

# Configuration
FRAMES_DIR="./downloads/frames/V2_1"
RESULTS_DIR="results/V2_1"
YOLO_MODEL="YOLO/output/yolo_training/weights/best.pt"

if [ "$COMMAND" == "detect" ]; then
    echo "=========================================="
    echo "BALL DETECTION MODE"
    echo "=========================================="
    echo "- YOLO ball detection"
    echo "- Hough Circle refinement"
    echo "- Air/Ground labeling"
    echo ""
    python3 detect_ball.py --mode detect --frames "$FRAMES_DIR" --output "$RESULTS_DIR" --yolo "$YOLO_MODEL"
    
elif [ "$COMMAND" == "correct" ]; then
    echo "=========================================="
    echo "BBOX CORRECTION MODE"
    echo "=========================================="
    echo "- Interactive bbox correction"
    echo "- Press 'f' to fix current frame"
    echo "- Press 'n' for next, 'p' for previous"
    echo "- Press 'q' to quit and save"
    echo ""
    python3 detect_ball.py --mode correct --frames "$FRAMES_DIR" --output "$RESULTS_DIR"
    
elif [ "$COMMAND" == "smooth" ]; then
    echo "=========================================="
    echo "SMOOTHING MODE"
    echo "=========================================="
    echo "- Smooth ball parameters across frames"
    echo "- Window size: $WINDOW_SIZE"
    echo ""
    python3 detect_ball.py --mode smooth --frames "$FRAMES_DIR" --output "$RESULTS_DIR" --smooth-type diameter
    
else
    echo "Invalid command: $COMMAND"
    echo "Available commands: detect, correct, smooth"
    exit 1
fi

