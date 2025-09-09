"""
Object Detection Script using OWLv2

This script processes a directory of images to detect specified objects using the OWLv2 
(Open-World Localization Vision) model. It draws bounding boxes around detected objects
and saves only images with detections above a specified confidence threshold.

Features:
- Batch processing of image directories
- Customizable object detection lists
- Adjustable confidence thresholds
- Visual bounding boxes with labels and confidence scores
- Support for multiple image formats (JPG, PNG, BMP, TIFF)
- Automatic GPU detection with CPU fallback
- Optional CPU-only mode

Example Usage:
    # Basic usage - detect an object with default threshold (0.1)
    python detect_objects.py ./input_images ./output_results --objects "yello explosive capsule"

    # Detect multiple objects with custom threshold
    python detect_objects.py /path/to/images /path/to/results --objects "yello explosive capsule" "grey capsule" --threshold 0.3
    
    # Force CPU usage (useful for debugging or preserving GPU memory)
    python detect_objects.py ./input_images ./output_results --objects "explosive device" --cpu
    
    # GPU will be used automatically if available (much faster processing)

Requirements:
    pip install torch transformers pillow tqdm

For GPU support (recommended):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Date: 2024-06-15
"""

import argparse
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from tqdm import tqdm


def load_model(device):
    """Load the OWLv2 model and processor."""
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = model.to(device)
    return processor, model


def draw_bounding_boxes(image, boxes, scores, labels, threshold):
    """Draw bounding boxes on the image with labels and confidence scores."""
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score.item() >= threshold:
            # Convert box coordinates to integers
            x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
            
            # Choose color for this detection
            color = colors[i % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Prepare label text (remove "a photo of " prefix)
            clean_label = label.replace("a photo of ", "")
            label_text = f"{clean_label}: {score.item():.3f}"
            
            # Get text bounding box for background
            bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background rectangle for text
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
            
            # Draw text
            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill='white', font=font)
    
    return image


def process_image(image_path, processor, model, object_list, threshold, device):
    """Process a single image and return detection results."""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare text labels for detection
        text_labels = [[f"a photo of {obj}" for obj in object_list]]
        
        # Process image
        inputs = processor(text=text_labels, images=image, return_tensors="pt")
        
        # Move inputs to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([(image.height, image.width)])
        results = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold, text_labels=text_labels
        )
        
        result = results[0]
        boxes, scores, labels = result["boxes"], result["scores"], result["text_labels"]
        
        # Check if any detections are above threshold
        valid_detections = [score.item() >= threshold for score in scores]
        
        if any(valid_detections):
            # Draw bounding boxes on image
            image_with_boxes = draw_bounding_boxes(image.copy(), boxes, scores, labels, threshold)
            return image_with_boxes, boxes, scores, labels
        
        return None, None, None, None
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None, None, None


def main():
    parser = argparse.ArgumentParser(description="Detect objects in images using OWLv2")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory to save images with detected objects")
    parser.add_argument("--objects", nargs='+', required=True, 
                       help="List of objects to detect (e.g., --objects cat dog car)")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Detection confidence threshold (default: 0.1)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    
    # Determine device
    if args.cpu:
        device = torch.device("cpu")
        print("Using CPU (forced)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU (GPU not available)")
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory '{args.input_dir}' does not exist or is not a directory")
        return
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading OWLv2 model...")
    processor, model = load_model(device)
    print("Model loaded successfully!")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all image files
    image_files = [f for f in input_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in '{args.input_dir}'")
        return
    
    print(f"Found {len(image_files)} image files")
    print(f"Looking for objects: {', '.join(args.objects)}")
    print(f"Detection threshold: {args.threshold}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    processed_count = 0
    saved_count = 0
    
    # Process images with progress bar
    for image_file in tqdm(image_files, desc="Processing images", unit="image"):
        # Process the image
        result_image, boxes, scores, labels = process_image(
            image_file, processor, model, args.objects, args.threshold, device
        )
        
        processed_count += 1
        
        if result_image is not None:
            # Save the image with bounding boxes
            output_file = output_path / f"detected_{image_file.name}"
            result_image.save(output_file)
            saved_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count} images")
    print(f"Saved: {saved_count} images with detections")


if __name__ == "__main__":
    main()