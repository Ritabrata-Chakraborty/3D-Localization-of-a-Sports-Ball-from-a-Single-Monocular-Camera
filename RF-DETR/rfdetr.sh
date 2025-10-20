#!/bin/bash
# ============================================================
# RF-DETR End-to-End Pipeline Script
# Author: Ritabrata Chakraborty
# ============================================================

# Exit on error
set -e

# ------------- CONFIG -------------
API_KEY="zHVH9LDpqVIHlqRMn7hi"
DATASET_URL="https://app.roboflow.com/bits-pilani-ujyf0/hockey-ow95d/2"
DATASET_PATH="./Hockey-2"
MODEL_TYPE="medium"
EPOCHS=20
BATCH_SIZE=4
GRAD_ACCUM_STEPS=2
CHECKPOINT="./output/checkpoint_best_total.pth"
IMAGE_PATH="./test_images/sample.jpg"
BATCH_DIR="./Hockey-2/test"
THRESHOLD=0.3
# ----------------------------------

# Function to display usage
show_help() {
    echo "Usage: ./run_rfdetr.sh [download|train|inference-single|inference-batch]"
    echo ""
    echo "Examples:"
    echo "  ./run_rfdetr.sh download"
    echo "  ./run_rfdetr.sh train"
    echo "  ./run_rfdetr.sh inference-single"
    echo "  ./run_rfdetr.sh inference-batch"
    echo ""
}

# Check if mode provided
if [ -z "$1" ]; then
    show_help
    exit 1
fi

MODE=$1


# ---------------- MODES ----------------

case "$MODE" in
    download)
        echo "Downloading dataset from Roboflow..."
        python3 train_rfdetr.py download \
            --api-key "$API_KEY" \
            --dataset-url "$DATASET_URL" \
            --dataset-path "$DATASET_PATH"
        ;;

    train)
        echo "Training RF-DETR model..."
        python3 train_rfdetr.py train \
            --dataset-path "$DATASET_PATH" \
            --model-type "$MODEL_TYPE" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --grad-accum-steps "$GRAD_ACCUM_STEPS"
        ;;

    inference-single)
        echo "Running single image inference..."
        python3 train_rfdetr.py inference \
            --checkpoint "$CHECKPOINT" \
            --model-type "$MODEL_TYPE" \
            --image "$IMAGE_PATH" \
            --threshold "$THRESHOLD"
        ;;

    inference-batch)
        echo "Running batch inference..."
        python3 train_rfdetr.py inference \
            --checkpoint "$CHECKPOINT" \
            --model-type "$MODEL_TYPE" \
            --batch-dir "$BATCH_DIR" \
            --threshold "$THRESHOLD"
        ;;

    *)
        echo "Invalid mode: $MODE"
        show_help
        exit 1
        ;;
esac

echo ""
echo "Task complete: $MODE"
echo "============================================"
