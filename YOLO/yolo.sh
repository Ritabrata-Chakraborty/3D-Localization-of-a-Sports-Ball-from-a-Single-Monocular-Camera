#!/bin/bash
# ============================================================
# YOLO End-to-End Pipeline Script
# Based on RF-DETR pipeline but adapted for YOLO models
# ============================================================

# Exit on error
set -e

# ------------- CONFIG -------------
API_KEY="zHVH9LDpqVIHlqRMn7hi"
DATASET_URL="https://app.roboflow.com/bits-pilani-ujyf0/football-i1lvk-z5kff/1"  # Always have version number at the end
DATASET_PATH="./Football"
MODEL_TYPE="large"
EPOCHS=30
BATCH_SIZE=8
IMG_SIZE=640  # Width (height=360 is hardcoded in script for rectangular format)
DEVICE="cuda"
CHECKPOINT="./output/yolo_training/weights/best.pt"
IMAGE_PATH="./test_images/sample.jpg"
BATCH_DIR="./Football/test"
THRESHOLD=0.5
# ----------------------------------

# Function to display usage
show_help() {
    echo "Usage: ./yolo.sh [download|train|inference-single|inference-batch|validate|export]"
    echo ""
    echo "Modes:"
    echo "  download         - Download dataset from Roboflow"
    echo "  train            - Train YOLO model"
    echo "  inference-single - Run inference on single image"
    echo "  inference-batch  - Run batch inference on directory"
    echo "  validate         - Validate model on test/validation set"
    echo "  export           - Export model to ONNX format"
    echo ""
    echo "Examples:"
    echo "  ./yolo.sh download"
    echo "  ./yolo.sh train"
    echo "  ./yolo.sh inference-single"
    echo "  ./yolo.sh inference-batch"
    echo "  ./yolo.sh validate"
    echo "  ./yolo.sh export"
    echo ""
    echo "Configuration (edit script to modify):"
    echo "  Model Type: $MODEL_TYPE"
    echo "  Epochs: $EPOCHS"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Image Size: $IMG_SIZE"
    echo "  Threshold: $THRESHOLD"
    echo ""
}

# Check if mode provided
if [ -z "$1" ]; then
    show_help
    exit 1
fi

MODE=$1

# Check if Python script exists
if [ ! -f "train_yolo.py" ]; then
    echo "❌ Error: train_yolo.py not found!"
    echo "   Make sure you're in the correct directory."
    exit 1
fi

# Install dependencies if requirements file exists
if [ -f "requirements_yolo.txt" ] && [ "$MODE" != "help" ]; then
    echo "📦 Installing YOLO dependencies..."
    pip install -r requirements_yolo.txt -q
    echo "✅ Dependencies installed"
    echo ""
fi

# ---------------- MODES ----------------

case "$MODE" in
    download)
        echo "🔄 Downloading dataset from Roboflow..."
        echo "   API Key: $API_KEY"
        echo "   Dataset URL: $DATASET_URL"
        echo "   Download Path: $DATASET_PATH"
        echo ""
        python3 train_yolo.py download \
            --api-key "$API_KEY" \
            --dataset-url "$DATASET_URL" \
            --dataset-path "$DATASET_PATH"
        ;;

    train)
        echo "🚀 Training YOLO model..."
        echo "   Dataset: $DATASET_PATH"
        echo "   Model Type: $MODEL_TYPE"
        echo "   Epochs: $EPOCHS"
        echo "   Batch Size: $BATCH_SIZE"
        echo "   Image Size: $IMG_SIZE"
        echo "   Device: $DEVICE"
        echo ""
        python3 train_yolo.py train \
            --dataset-path "$DATASET_PATH" \
            --model-type "$MODEL_TYPE" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --img-size "$IMG_SIZE" \
            --device "$DEVICE"
        ;;

    resume)
        echo "🔄 Resuming YOLO training from checkpoint..."
        if [ ! -f "$CHECKPOINT" ]; then
            echo "❌ Checkpoint not found: $CHECKPOINT"
            echo "   Please train a model first or adjust CHECKPOINT path in script"
            exit 1
        fi
        echo "   Dataset: $DATASET_PATH"
        echo "   Checkpoint: $CHECKPOINT"
        echo "   Epochs: $EPOCHS"
        echo ""
        python3 train_yolo.py train \
            --dataset-path "$DATASET_PATH" \
            --model-type "$MODEL_TYPE" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --img-size "$IMG_SIZE" \
            --device "$DEVICE" \
            --resume-checkpoint "$CHECKPOINT"
        ;;

    inference-single)
        echo "🖼️  Running single image inference..."
        echo "   Image: $IMAGE_PATH"
        echo "   Model: $CHECKPOINT"
        echo "   Threshold: $THRESHOLD"
        echo ""
        if [ ! -f "$IMAGE_PATH" ]; then
            echo "⚠️  Image not found: $IMAGE_PATH"
            echo "   Using pretrained model instead of checkpoint"
            python3 train_yolo.py inference \
                --model-type "$MODEL_TYPE" \
                --image "$IMAGE_PATH" \
                --threshold "$THRESHOLD" \
                --img-size "$IMG_SIZE" \
                --device "$DEVICE"
        else
            python3 train_yolo.py inference \
                --checkpoint "$CHECKPOINT" \
                --model-type "$MODEL_TYPE" \
                --image "$IMAGE_PATH" \
                --threshold "$THRESHOLD" \
                --img-size "$IMG_SIZE" \
                --device "$DEVICE"
        fi
        ;;

    inference-batch)
        echo "📁 Running batch inference..."
        echo "   Directory: $BATCH_DIR"
        echo "   Model: $CHECKPOINT"
        echo "   Threshold: $THRESHOLD"
        echo ""
        if [ ! -d "$BATCH_DIR" ]; then
            echo "❌ Directory not found: $BATCH_DIR"
            exit 1
        fi
        
        if [ -f "$CHECKPOINT" ]; then
            python3 train_yolo.py inference \
                --checkpoint "$CHECKPOINT" \
                --model-type "$MODEL_TYPE" \
                --batch-dir "$BATCH_DIR" \
                --threshold "$THRESHOLD" \
                --img-size "$IMG_SIZE" \
                --device "$DEVICE"
        else
            echo "⚠️  Checkpoint not found: $CHECKPOINT"
            echo "   Using pretrained model instead"
            python3 train_yolo.py inference \
                --model-type "$MODEL_TYPE" \
                --batch-dir "$BATCH_DIR" \
                --threshold "$THRESHOLD" \
                --img-size "$IMG_SIZE" \
                --device "$DEVICE"
        fi
        ;;

    validate)
        echo "📊 Validating YOLO model..."
        if [ ! -f "$CHECKPOINT" ]; then
            echo "❌ Checkpoint not found: $CHECKPOINT"
            echo "   Please train a model first or adjust CHECKPOINT path in script"
            exit 1
        fi
        echo "   Dataset: $DATASET_PATH"
        echo "   Checkpoint: $CHECKPOINT"
        echo ""
        python3 train_yolo.py validate \
            --checkpoint "$CHECKPOINT" \
            --dataset-path "$DATASET_PATH" \
            --img-size "$IMG_SIZE" \
            --device "$DEVICE"
        ;;

    export)
        echo "📦 Exporting YOLO model to ONNX..."
        if [ ! -f "$CHECKPOINT" ]; then
            echo "❌ Checkpoint not found: $CHECKPOINT"
            echo "   Please train a model first or adjust CHECKPOINT path in script"
            exit 1
        fi
        echo "   Checkpoint: $CHECKPOINT"
        echo "   Format: ONNX"
        echo ""
        python3 train_yolo.py export \
            --checkpoint "$CHECKPOINT" \
            --format onnx
        ;;

    help|--help|-h)
        show_help
        ;;

    *)
        echo "❌ Invalid mode: $MODE"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "✅ Task complete: $MODE"
echo "============================================"
