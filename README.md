# Object Detection Script using OWLv2

A powerful Python script for detecting objects in images using the OWLv2 (Open-World Localization Vision) model. This script processes directories of images, detects specified objects, and saves images with bounding boxes drawn around detected objects.

## Features

- 🖼️ **Batch processing** of image directories
- 🎯 **Customizable object detection** - specify any objects to detect
- ⚡ **GPU acceleration** with automatic CPU fallback
- 📊 **Adjustable confidence thresholds**
- 🎨 **Visual bounding boxes** with labels and confidence scores
- 📁 **Multiple image formats** (JPG, PNG, BMP, TIFF)
- 📈 **Progress tracking** with real-time progress bar

## Requirements

### Software Dependencies
```bash
pip install torch transformers pillow tqdm
```

### For GPU Support (Recommended)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Hardware Requirements

#### GPU (Recommended)
- **VRAM**: 4-8 GB minimum
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060/4060 or better)
- **Performance**: 1-5 seconds per image

#### CPU (Fallback)
- **RAM**: 8-16 GB minimum
- **CPU**: Modern multi-core processor (Intel i5/i7 or AMD Ryzen 5/7)
- **Performance**: 30-120 seconds per image

## Installation

1. Clone or download the script
2. Install dependencies:
   ```bash
   pip install torch transformers pillow tqdm
   ```
3. For GPU support (optional but recommended):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### Basic Syntax
```bash
python detect_objects.py <input_directory> <output_directory> --objects <object1> <object2> ... [options]
```

### Examples

#### Basic Detection
```bash
python detect_objects.py ./input_images ./output_results --objects "cat" "dog"
```

#### Multiple Objects with Custom Threshold
```bash
python detect_objects.py /path/to/images /path/to/results --objects "explosive device" "weapon" "suspicious package" --threshold 0.3
```

#### Force CPU Usage
```bash
python detect_objects.py ./images ./results --objects "person" "vehicle" --cpu
```

#### Security/Safety Applications
```bash
python detect_objects.py ./security_footage ./flagged_images --objects "weapon" "knife" "gun" "explosive" --threshold 0.25
```

## Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `input_dir` | string | Yes | Directory containing input images |
| `output_dir` | string | Yes | Directory to save images with detections |
| `--objects` | list | Yes | Space-separated list of objects to detect |
| `--threshold` | float | No | Detection confidence threshold (default: 0.1) |
| `--cpu` | flag | No | Force CPU usage even if GPU is available |

## Output

The script will:

1. **Process all images** in the input directory
2. **Detect specified objects** using the OWLv2 model
3. **Draw bounding boxes** around detected objects with:
   - Colored rectangles around objects
   - Labels with object names
   - Confidence scores
4. **Save only images with detections** above the threshold to the output directory
5. **Add "detected_" prefix** to output filenames

### Example Output
```
Using GPU: NVIDIA GeForce RTX 4060
Loading OWLv2 model...
Model loaded successfully!
Found 150 image files
Looking for objects: explosive device, weapon, suspicious package
Detection threshold: 0.25
Output directory: ./flagged_images

Processing images: 100%|████████████| 150/150 [02:30<00:00, 1.2image/s]

Processing complete!
Processed: 150 images
Saved: 12 images with detections
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Performance Tips

1. **Use GPU when available** - provides 10-50x speedup
2. **Adjust threshold** - higher thresholds reduce false positives
3. **Resize large images** before processing to improve speed
4. **Use specific object names** for better detection accuracy

## Object Detection Tips

### Good Object Descriptions
- "person"
- "explosive device"
- "knife"
- "gun"
- "suspicious package"
- "vehicle"
- "bicycle"

### Less Effective Descriptions
- "thing"
- "stuff"
- "object"

## Troubleshooting

### Common Issues

**Out of Memory Error (GPU)**
```bash
# Use CPU instead
python detect_objects.py ./images ./results --objects "cat" --cpu
```

**No Images Found**
- Check that input directory exists
- Verify image file extensions are supported
- Ensure images are not in subdirectories

**Poor Detection Results**
- Try different object descriptions
- Adjust the confidence threshold
- Ensure good image quality

**Slow Processing**
- Install GPU-enabled PyTorch for faster processing
- Reduce image resolution if very large

### Model Download

On first run, the script will automatically download the OWLv2 model (~2-3 GB). This happens once and the model is cached locally.

## License

This script uses the OWLv2 model from Google Research. Please refer to the original model's license and usage terms.

## Contributing

Feel free to submit issues and enhancement requests!

## Author

Hannah White (with GitHub Copilot assistance)  
Date: 2024-06-15
