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
from io import BytesIO
import supervision as sv
from roboflow import download_dataset

# Model mapping for inference library
MODEL_MAPPING = {
    "nano": "rfdetr-nano",
    "small": "rfdetr-small",
    "medium": "rfdetr-medium",
    "large": "rfdetr-large"
}


def download_dataset_cli(api_key: str, dataset_url: str, dataset_path: str):
    """Download dataset from Roboflow using API key."""
    os.environ["ROBOFLOW_API_KEY"] = api_key
    dataset = download_dataset(dataset_url, "coco", location=dataset_path)
    print(f"✅ Dataset downloaded successfully to {dataset_path}!")
    return dataset


def train_model(dataset_path: str, model_type: str, epochs: int, batch_size: int, 
                grad_accum_steps: int, resume_checkpoint: str = None):
    """Train RF-DETR model on downloaded dataset."""
    from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRNano
    
    model_classes = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge
    }
    
    model_class = model_classes.get(model_type.lower())
    if not model_class:
        raise ValueError(f"Invalid model type: {model_type}. Choose from: nano, small, medium, large")
    
    # Load model (from checkpoint if resuming, otherwise new)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"📂 Resuming training from checkpoint: {resume_checkpoint}")
        model = model_class(pretrain_weights=resume_checkpoint)
    else:
        print(f"🆕 Initializing new {model_type} model")
        model = model_class()
    
    print(f"🚀 Training {model_type} model:")
    print(f"   Dataset: {dataset_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation steps: {grad_accum_steps}")
    
    model.train(dataset_dir=dataset_path, epochs=epochs, batch_size=batch_size, 
                grad_accum_steps=grad_accum_steps)
    print("✅ Training complete. Metrics saved at ./output/metrics_plot.png")


def inference(checkpoint_path: str, model_type: str, image_path: str = None, 
              batch_dir: str = None, threshold: float = 0.5):
    """Perform inference with RF-DETR model using roboflow inference library."""
    
    # Load model - use checkpoint if exists, otherwise use pretrained from roboflow
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📂 Loading model from checkpoint: {checkpoint_path}")
        try:
            from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRNano
        except ImportError:
            print("⚠️  rfdetr package not found. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rfdetr>=1.2.1"])
            from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRNano
        
        model_classes = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "large": RFDETRLarge
        }
        model_class = model_classes.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Invalid model type: {model_type}. Choose from: nano, small, medium, large")
        
        # Load checkpoint with pretrain_weights parameter
        model = model_class(pretrain_weights=checkpoint_path)
        model.optimize_for_inference()
        use_rfdetr = True
    else:
        # Use roboflow inference library with pretrained models
        try:
            from inference import get_model
        except ImportError:
            print("⚠️  inference package not found. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "inference-gpu>=0.51.7"])
            from inference import get_model
        
        model_name = MODEL_MAPPING.get(model_type.lower())
        if not model_name:
            raise ValueError(f"Invalid model type: {model_type}. Choose from: nano, small, medium, large")
        
        print(f"🌐 Loading pretrained {model_name} from Roboflow")
        model = get_model(model_name)
        use_rfdetr = False

    # Single image inference
    if image_path:
        print(f"🖼️  Running single image inference on: {image_path}")
        image = Image.open(image_path)
        
        if use_rfdetr:
            detections = model.predict(image, threshold=threshold)
        else:
            predictions = model.infer(image, confidence=threshold)[0]
            detections = sv.Detections.from_inference(predictions)

        # Annotate and save
        annotated_image = image.copy()
        annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
        
        if use_rfdetr:
            labels = [f"Class {class_id}: {conf:.2f}" 
                     for class_id, conf in zip(detections.class_id, detections.confidence)]
        else:
            labels = [f"{pred.class_name}: {pred.confidence:.2f}" 
                     for pred in predictions.predictions]
        
        annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
            annotated_image, detections, labels)

        # Save output
        output_path = str(Path(image_path).with_stem(Path(image_path).stem + '_output'))
        annotated_image.save(output_path)
        print(f"✅ Output saved to: {output_path}")
        
        sv.plot_image(annotated_image)
        return detections

    # Batch inference mode
    elif batch_dir:
        print(f"📁 Running batch inference on directory: {batch_dir}")
        
        # Get all images from directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(Path(batch_dir).glob(f'*{ext}')))
            image_files.extend(list(Path(batch_dir).glob(f'*{ext.upper()}')))
        
        if not image_files:
            print(f"❌ No images found in {batch_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        predictions = []
        output_dir = Path(batch_dir) / "predictions"
        output_dir.mkdir(exist_ok=True)
        
        for img_path in tqdm(image_files, desc="Processing images"):
            image = Image.open(img_path)
            
            # Run inference
            if use_rfdetr:
                detections = model.predict(image, threshold=threshold)
                labels = [f"Class {class_id}: {conf:.2f}" 
                         for class_id, conf in zip(detections.class_id, detections.confidence)]
            else:
                preds = model.infer(image, confidence=threshold)[0]
                detections = sv.Detections.from_inference(preds)
                labels = [f"{pred.class_name}: {pred.confidence:.2f}" 
                         for pred in preds.predictions]
            
            predictions.append(detections)
            
            # Annotate and save
            annotated_image = image.copy()
            annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
                annotated_image, detections)
            annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(
                annotated_image, detections, labels)
            
            # Save output
            output_path = output_dir / img_path.name
            annotated_image.save(output_path)

        print(f"✅ Batch inference complete: {len(predictions)} images processed.")
        print(f"   Results saved to: {output_dir}")
        return predictions
    
    else:
        raise ValueError("Either --image or --batch-dir must be provided for inference")


def main():
    parser = argparse.ArgumentParser(
        description="RF-DETR Pipeline Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download dataset
  python train_rfdetr.py download --api-key YOUR_KEY --dataset-url URL --dataset-path ./Hockey-2

  # Train model
  python train_rfdetr.py train --dataset-path ./Hockey-2 --model-type medium --epochs 30 --batch-size 4 --grad-accum-steps 2

  # Resume training from checkpoint
  python train_rfdetr.py train --dataset-path ./Hockey-2 --model-type medium --epochs 30 --resume-checkpoint ./output/checkpoint.pth

  # Single image inference
  python train_rfdetr.py inference --checkpoint ./output/checkpoint_best_regular.pth --model-type medium --image dog.jpg

  # Batch inference
  python train_rfdetr.py inference --checkpoint ./output/checkpoint_best_regular.pth --model-type medium --batch-dir ./test_images
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
    train_parser = subparsers.add_parser('train', help='Train RF-DETR model')
    train_parser.add_argument('--dataset-path', type=str, default='./dataset',
                             help='Path to dataset directory (default: ./dataset)')
    train_parser.add_argument('--model-type', type=str, required=True,
                             choices=['nano', 'small', 'medium', 'large'],
                             help='Model size: nano | small | medium | large')
    train_parser.add_argument('--epochs', type=int, default=10,
                             help='Number of training epochs (default: 10)')
    train_parser.add_argument('--batch-size', type=int, default=8,
                             help='Training batch size (default: 8)')
    train_parser.add_argument('--grad-accum-steps', type=int, default=1,
                             help='Gradient accumulation steps (default: 1)')
    train_parser.add_argument('--resume-checkpoint', type=str, default=None,
                             help='Path to checkpoint to resume training from')

    # Inference parser
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--checkpoint', type=str, default=None,
                                  help='Path to model checkpoint (optional, uses pretrained if not provided)')
    inference_parser.add_argument('--model-type', type=str, required=True,
                                  choices=['nano', 'small', 'medium', 'large'],
                                  help='Model size: nano | small | medium | large')
    
    # Inference mode: single image or batch
    inference_group = inference_parser.add_mutually_exclusive_group(required=True)
    inference_group.add_argument('--image', type=str,
                                help='Path to single image for inference')
    inference_group.add_argument('--batch-dir', type=str,
                                help='Path to directory containing images for batch inference')
    
    inference_parser.add_argument('--threshold', type=float, default=0.5,
                                 help='Detection confidence threshold (default: 0.5)')

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
            grad_accum_steps=args.grad_accum_steps,
            resume_checkpoint=args.resume_checkpoint
        )

    elif args.mode == "inference":
        if args.checkpoint and not os.path.exists(args.checkpoint):
            raise ValueError(f"❌ Checkpoint not found: {args.checkpoint}")
        
        if not args.checkpoint:
            print("ℹ️  No checkpoint provided, using pretrained model from Roboflow")
        
        inference(
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            image_path=args.image,
            batch_dir=args.batch_dir,
            threshold=args.threshold
        )


if __name__ == "__main__":
    main()
