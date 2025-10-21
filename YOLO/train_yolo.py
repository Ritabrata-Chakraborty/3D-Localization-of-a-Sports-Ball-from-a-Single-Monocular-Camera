import argparse
import os
import sys
from pathlib import Path

# Check and install missing packages
def check_and_install_dependencies():
    """Check and install required dependencies."""
    required_packages = {
        'tqdm': 'tqdm',
        'PIL': 'Pillow',
        'supervision': 'supervision>=0.26.1',
        'roboflow': 'roboflow',
        'charset_normalizer': 'charset-normalizer',
        'ultralytics': 'ultralytics>=8.0.0',
        'cv2': 'opencv-python',
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("✓ Dependencies installed")

check_and_install_dependencies()

from tqdm import tqdm
from PIL import Image
import supervision as sv
from roboflow import download_dataset
import cv2
import yaml

# Model mapping for YOLO models
MODEL_MAPPING = {
    "nano": "yolo11n.pt",
    "small": "yolo11s.pt", 
    "medium": "yolo11m.pt",
    "large": "yolo11l.pt",
    "extra": "yolo11x.pt"
}


def download_dataset_cli(api_key: str, dataset_url: str, dataset_path: str):
    """Download dataset from Roboflow using API key."""
    os.environ["ROBOFLOW_API_KEY"] = api_key
    dataset = download_dataset(dataset_url, "yolov8", location=dataset_path)
    print(f"✅ Dataset downloaded successfully to {dataset_path}!")
    return dataset


def convert_coco_to_yolo(dataset_path: str):
    """Convert COCO format dataset to YOLO format if needed."""
    from ultralytics.data.converter import convert_coco
    
    coco_annotation_files = list(Path(dataset_path).rglob("*_annotations.coco.json"))
    
    if coco_annotation_files:
        print("🔄 Converting COCO format to YOLO format...")
        for coco_file in coco_annotation_files:
            split_dir = coco_file.parent
            convert_coco(
                labels_dir=str(split_dir),
                save_dir=str(split_dir),
                use_segments=False,
                use_keypoints=False
            )
        print("✅ COCO to YOLO conversion complete!")


def create_yolo_yaml(dataset_path: str, class_names: list = None):
    """Create YOLO dataset configuration YAML file."""
    dataset_path = Path(dataset_path)
    
    # Try to find existing data.yaml
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        print(f"📄 Using existing data.yaml: {yaml_path}")
        return str(yaml_path)
    
    # Determine class names
    if not class_names:
        # Try to extract from COCO annotations
        import json
        coco_files = list(dataset_path.rglob("*_annotations.coco.json"))
        if coco_files:
            with open(coco_files[0], 'r') as f:
                coco_data = json.load(f)
            class_names = [cat['name'] for cat in coco_data.get('categories', [])]
        else:
            # Default for hockey dataset
            class_names = ['puck', 'player', 'stick', 'goal']
    
    # Create YAML configuration
    config = {
        'path': str(dataset_path.absolute()),
        'train': 'train/images' if (dataset_path / 'train' / 'images').exists() else 'train',
        'val': 'valid/images' if (dataset_path / 'valid' / 'images').exists() else 'valid',
        'test': 'test/images' if (dataset_path / 'test' / 'images').exists() else 'test',
        'nc': len(class_names),
        'names': class_names
    }
    
    # Save YAML file
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📄 Created data.yaml: {yaml_path}")
    print(f"   Classes ({len(class_names)}): {', '.join(class_names)}")
    return str(yaml_path)


def train_model(dataset_path: str, model_type: str, epochs: int, batch_size: int, 
                img_size: int = 640, resume_checkpoint: str = None, device: str = 'auto'):
    """Train YOLO model on downloaded dataset."""
    from ultralytics import YOLO
    
    # Get model path
    model_path = MODEL_MAPPING.get(model_type.lower())
    if not model_path:
        raise ValueError(f"Invalid model type: {model_type}. Choose from: nano, small, medium, large, extra")
    
    # Convert dataset if needed and create YAML
    convert_coco_to_yolo(dataset_path)
    yaml_path = create_yolo_yaml(dataset_path)
    
    # Load model (from checkpoint if resuming, otherwise pretrained)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"📂 Resuming training from checkpoint: {resume_checkpoint}")
        model = YOLO(resume_checkpoint)
    else:
        print(f"🆕 Initializing new {model_type} model ({model_path})")
        model = YOLO(model_path)
    
    print(f"🚀 Training {model_type} YOLO model:")
    print(f"   Dataset: {dataset_path}")
    print(f"   YAML config: {yaml_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: [640, 360] (maintaining dataset aspect ratio)")
    print(f"   Device: {device}")
    
    # Train the model with rectangular image size to match dataset (640x360)
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=[640, 360],  # Use rectangular size to match dataset
        device=device,
        project='./output',
        name='yolo_training',
        save=True,
        save_period=10,
        plots=True,
        verbose=True
    )
    
    print("✅ Training complete!")
    print(f"   Results saved to: ./output/yolo_training/")
    print(f"   Best weights: ./output/yolo_training/weights/best.pt")
    print(f"   Last weights: ./output/yolo_training/weights/last.pt")
    
    return results


def inference(checkpoint_path: str, model_type: str, image_path: str = None, 
              batch_dir: str = None, threshold: float = 0.5, img_size: int = 640,
              device: str = 'auto'):
    """Perform inference with YOLO model."""
    from ultralytics import YOLO
    
    # Handle device='auto' which causes issues
    if device == 'auto':
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Auto-detected device: {device}")
    
    # Load model
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📂 Loading model from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        # Use pretrained model
        model_path = MODEL_MAPPING.get(model_type.lower())
        if not model_path:
            raise ValueError(f"Invalid model type: {model_type}. Choose from: nano, small, medium, large, extra")
        
        print(f"🌐 Loading pretrained {model_type} YOLO model ({model_path})")
        model = YOLO(model_path)

    # Single image inference
    if image_path:
        print(f"🖼️  Running single image inference on: {image_path}")
        
        # Run prediction with rectangular image size
        results = model.predict(
            source=image_path,
            conf=threshold,
            imgsz=[640, 360],  # Use rectangular size to match dataset
            device=device,
            save=True,
            project='./output',
            name='inference_single'
        )
        
        # Convert to supervision format for consistent handling
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Load original image for annotation
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Annotate image
        box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW)
        label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW)
        
        annotated_image = box_annotator.annotate(image_rgb, detections)
        
        # Create labels
        if hasattr(model, 'names'):
            class_names = model.names
            labels = [f"{class_names[class_id]}: {conf:.2f}" 
                     for class_id, conf in zip(detections.class_id, detections.confidence)]
        else:
            labels = [f"Class {class_id}: {conf:.2f}" 
                     for class_id, conf in zip(detections.class_id, detections.confidence)]
        
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)
        
        # Save output
        output_path = str(Path(image_path).with_stem(Path(image_path).stem + '_yolo_output'))
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, annotated_image_bgr)
        print(f"✅ Output saved to: {output_path}")
        print(f"   Also saved to: ./output/inference_single/")
        
        return detections

    # Batch inference mode
    elif batch_dir:
        print(f"📁 Running batch inference on directory: {batch_dir}")
        
        # Run batch prediction with rectangular image size
        results = model.predict(
            source=batch_dir,
            conf=threshold,
            imgsz=[640, 360],  # Use rectangular size to match dataset
            device=device,
            save=True,
            project='./output',
            name='inference_batch'
        )
        
        print(f"✅ Batch inference complete: {len(results)} images processed.")
        print(f"   Results saved to: ./output/inference_batch/")
        
        # Convert all results to supervision format
        all_detections = []
        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            all_detections.append(detections)
        
        return all_detections
    
    else:
        raise ValueError("Either --image or --batch-dir must be provided for inference")


def validate_model(checkpoint_path: str, dataset_path: str, img_size: int = 640, device: str = 'auto'):
    """Validate YOLO model on test/validation set."""
    from ultralytics import YOLO
    
    # Handle device='auto' which causes issues
    if device == 'auto':
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Auto-detected device: {device}")
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"❌ Checkpoint not found: {checkpoint_path}")
    
    # Create YAML config
    yaml_path = create_yolo_yaml(dataset_path)
    
    print(f"📊 Validating model: {checkpoint_path}")
    print(f"   Dataset: {dataset_path}")
    print(f"   YAML config: {yaml_path}")
    
    model = YOLO(checkpoint_path)
    
    # Run validation with rectangular image size
    results = model.val(
        data=yaml_path,
        imgsz=[640, 360],  # Use rectangular size to match dataset
        device=device,
        project='./output',
        name='validation',
        save_json=True,
        plots=True
    )
    
    print("✅ Validation complete!")
    print(f"   Results saved to: ./output/validation/")
    
    return results


def export_model(checkpoint_path: str, format: str = 'onnx'):
    """Export YOLO model to different formats."""
    from ultralytics import YOLO
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"❌ Checkpoint not found: {checkpoint_path}")
    
    print(f"📦 Exporting model to {format.upper()} format...")
    
    model = YOLO(checkpoint_path)
    
    # Export model
    model.export(
        format=format,
        project='./output',
        name='exported_model'
    )
    
    print(f"✅ Model exported to {format.upper()} format!")
    print(f"   Exported model saved to: ./output/exported_model/")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Pipeline Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download dataset
  python train_yolo.py download --api-key YOUR_KEY --dataset-url URL --dataset-path ./Hockey-2

  # Train model
  python train_yolo.py train --dataset-path ./Hockey-2 --model-type medium --epochs 50 --batch-size 16

  # Resume training from checkpoint
  python train_yolo.py train --dataset-path ./Hockey-2 --model-type medium --epochs 50 --resume-checkpoint ./output/yolo_training/weights/last.pt

  # Single image inference
  python train_yolo.py inference --checkpoint ./output/yolo_training/weights/best.pt --model-type medium --image dog.jpg

  # Batch inference
  python train_yolo.py inference --checkpoint ./output/yolo_training/weights/best.pt --model-type medium --batch-dir ./test_images

  # Validate model
  python train_yolo.py validate --checkpoint ./output/yolo_training/weights/best.pt --dataset-path ./Hockey-2

  # Export model
  python train_yolo.py export --checkpoint ./output/yolo_training/weights/best.pt --format onnx
        """
    )

    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)

    # Download parser
    download_parser = subparsers.add_parser('download', help='Download dataset from Roboflow')
    download_parser.add_argument('--api-key', type=str, required=True,
                                 help='Roboflow API key')
    download_parser.add_argument('--dataset-url', type=str, required=True,
                                 help='Roboflow dataset URL')
    download_parser.add_argument('--dataset-path', type=str, default='./dataset',
                                 help='Path where dataset will be downloaded (default: ./dataset)')

    # Train parser
    train_parser = subparsers.add_parser('train', help='Train YOLO model')
    train_parser.add_argument('--dataset-path', type=str, default='./dataset',
                             help='Path to dataset directory (default: ./dataset)')
    train_parser.add_argument('--model-type', type=str, required=True,
                             choices=['nano', 'small', 'medium', 'large', 'extra'],
                             help='Model size: nano | small | medium | large | extra')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs (default: 50)')
    train_parser.add_argument('--batch-size', type=int, default=16,
                             help='Training batch size (default: 16)')
    train_parser.add_argument('--img-size', type=int, default=640,
                             help='Image size for training (default: 640)')
    train_parser.add_argument('--resume-checkpoint', type=str, default=None,
                             help='Path to checkpoint to resume training from')
    train_parser.add_argument('--device', type=str, default='auto',
                             help='Device to use (auto, cpu, 0, 1, etc.) (default: auto)')

    # Inference parser
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--checkpoint', type=str, default=None,
                                  help='Path to model checkpoint (optional, uses pretrained if not provided)')
    inference_parser.add_argument('--model-type', type=str, required=True,
                                  choices=['nano', 'small', 'medium', 'large', 'extra'],
                                  help='Model size: nano | small | medium | large | extra')
    
    # Inference mode: single image or batch
    inference_group = inference_parser.add_mutually_exclusive_group(required=True)
    inference_group.add_argument('--image', type=str,
                                help='Path to single image for inference')
    inference_group.add_argument('--batch-dir', type=str,
                                help='Path to directory containing images for batch inference')
    
    inference_parser.add_argument('--threshold', type=float, default=0.5,
                                 help='Detection confidence threshold (default: 0.5)')
    inference_parser.add_argument('--img-size', type=int, default=640,
                                 help='Image size for inference (default: 640)')
    inference_parser.add_argument('--device', type=str, default='auto',
                                 help='Device to use (auto, cpu, 0, 1, etc.) (default: auto)')

    # Validation parser
    validate_parser = subparsers.add_parser('validate', help='Validate model on test/validation set')
    validate_parser.add_argument('--checkpoint', type=str, required=True,
                                help='Path to model checkpoint')
    validate_parser.add_argument('--dataset-path', type=str, required=True,
                                help='Path to dataset directory')
    validate_parser.add_argument('--img-size', type=int, default=640,
                                help='Image size for validation (default: 640)')
    validate_parser.add_argument('--device', type=str, default='auto',
                                help='Device to use (auto, cpu, 0, 1, etc.) (default: auto)')

    # Export parser
    export_parser = subparsers.add_parser('export', help='Export model to different formats')
    export_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Path to model checkpoint')
    export_parser.add_argument('--format', type=str, default='onnx',
                              choices=['onnx', 'torchscript', 'coreml', 'tflite', 'pb', 'engine'],
                              help='Export format (default: onnx)')

    args = parser.parse_args()

    # Route to appropriate function
    if args.mode == "download":
        download_dataset_cli(args.api_key, args.dataset_url, args.dataset_path)

    elif args.mode == "train":
        if not os.path.exists(args.dataset_path):
            raise ValueError(f"❌ Dataset not found at {args.dataset_path}. Please run download mode first.")
        
        train_model(
            dataset_path=args.dataset_path,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            resume_checkpoint=args.resume_checkpoint,
            device=args.device
        )

    elif args.mode == "inference":
        if args.checkpoint and not os.path.exists(args.checkpoint):
            raise ValueError(f"❌ Checkpoint not found: {args.checkpoint}")
        
        if not args.checkpoint:
            print("ℹ️  No checkpoint provided, using pretrained model")
        
        inference(
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            image_path=args.image,
            batch_dir=args.batch_dir,
            threshold=args.threshold,
            img_size=args.img_size,
            device=args.device
        )

    elif args.mode == "validate":
        validate_model(
            checkpoint_path=args.checkpoint,
            dataset_path=args.dataset_path,
            img_size=args.img_size,
            device=args.device
        )

    elif args.mode == "export":
        export_model(
            checkpoint_path=args.checkpoint,
            format=args.format
        )


if __name__ == "__main__":
    main()