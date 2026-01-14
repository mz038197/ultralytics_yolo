# Ultralytics YOLO11 - Object Detection & Instance Segmentation

A comprehensive YOLO11-based system for object detection and instance segmentation tasks using the Ultralytics framework.

## 📋 Project Overview

This project provides a complete solution for:
- **Object Detection**: Real-time detection of objects in images and videos
- **Instance Segmentation**: Pixel-level segmentation of detected objects
- **Model Training**: Train custom YOLO11 models on your own datasets
- **Inference**: Run predictions on images, videos, or webcam feeds

## 📁 Project Structure

```
ultralytics_yolo/
├── train.py           # Model training script
├── predict.py         # Inference and prediction script
├── export.py          # Model export functionality
├── setup.bat          # Windows environment setup script
├── requirements.txt   # Python dependencies
├── yolo11n.pt         # Pre-trained YOLO11-nano model
└── .gitignore         # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Windows OS (or modify setup.bat for other systems)

### Installation (Windows)

1. **Automatic Setup (Recommended)**
   ```bash
   setup.bat
   ```
   This will:
   - Create a Python virtual environment
   - Activate the environment
   - Install all required dependencies

2. **Manual Setup**
   ```bash
   # Create virtual environment
   py -3 -m venv venv
   
   # Activate virtual environment
   venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

## 📦 Dependencies

- **ultralytics** (≥8.0.0) - YOLO detection framework
- **opencv-python** (≥4.8.0) - Image processing
- **torch** (≥2.0.0) - Deep learning framework (auto-installed)
- **torchvision** (≥0.15.0) - Computer vision utilities
- **numpy** (≥1.23.0) - Numerical computing
- **pillow** (≥10.0.0) - Image library
- **tqdm** (≥4.66.0) - Progress bars

## 🎯 Usage

### Training a Model

```bash
# Activate virtual environment first
venv\Scripts\activate

# Run training script
python train.py
```

**Features:**
- Choose between Object Detection or Instance Segmentation
- Select from multiple pre-trained models (nano, small, medium, large, xlarge)
- Customize training parameters (epochs, batch size, image size)
- Specify your dataset path (COCO format)
- Automatic model saving to `runs/` directory

**Supported Models:**
- yolo11n.pt (Nano - fastest, lowest accuracy)
- yolo11s.pt (Small - balanced)
- yolo11m.pt (Medium - higher accuracy)
- yolo11l.pt (Large - slower, high accuracy)
- yolo11x.pt (XLarge - slowest, best accuracy)

### Running Inference

```bash
# Activate virtual environment first
venv\Scripts\activate

# Run prediction script
python predict.py
```

**Input Options:**
1. **Webcam**: Real-time detection from your camera
2. **Image Folder**: Batch process images from a directory

**Detection Modes:**
1. Object Detection
2. Instance Segmentation

### Exporting Models

```bash
python export.py
```

Export trained models to various formats for deployment.

## 📊 Output

- Detection results are saved to the `runs/` directory (auto-created)
- Includes:
  - `detect/` - Detection results
  - `segment/` - Segmentation results
  - `train/` - Training logs and weights
  - Each run has subdirectories with timestamps

## 🛠️ Model Selection Guide

| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| yolo11n | ⚡⚡⚡ | ⭐ | 💾 | Real-time on edge devices |
| yolo11s | ⚡⚡ | ⭐⭐ | 💾💾 | Balanced performance |
| yolo11m | ⚡ | ⭐⭐⭐ | 💾💾💾 | High accuracy applications |
| yolo11l | - | ⭐⭐⭐⭐ | 💾💾💾💾 | Maximum accuracy |
| yolo11x | - | ⭐⭐⭐⭐⭐ | 💾💾💾💾💾 | Best accuracy (slow) |

## 📝 Dataset Format

The project expects datasets in COCO format:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## ⚙️ Configuration

### Training Parameters (Interactive)
- Model selection (nano to xlarge)
- Number of epochs
- Batch size
- Image resolution
- Dataset path
- Device selection (GPU/CPU)

### Inference Parameters (Interactive)
- Mode selection (detection or segmentation)
- Source selection (webcam or folder)
- Confidence threshold
- IOU threshold

## 📖 Resource Links

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [YOLO GitHub Repository](https://github.com/ultralytics/ultralytics)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [COCO Dataset Format](https://cocodataset.org/)

## 📌 Important Notes

- The `runs/` directory contains training outputs and predictions (not tracked in Git)
- Model files (`*.pt`) are not tracked in Git (they can be large)
- Use GPU when available for significantly faster training and inference
- For server environments without display, use `opencv-python-headless` instead

## 🤝 Troubleshooting

### GPU Not Detected
```bash
# Verify PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory Errors
- Reduce batch size
- Use a smaller model (nano or small)
- Reduce image resolution

### OpenCV Display Issues (Linux/Server)
- Use `opencv-python-headless` from requirements.txt
- Or save results to disk instead of displaying

## 📄 License

This project uses Ultralytics YOLO11, which is available under the AGPL-3.0 license.

## 👤 Author Notes

Built with the Ultralytics YOLO framework for efficient object detection and segmentation tasks.

---

**Last Updated**: January 2026
