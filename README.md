# Trans-OBB 🚀 | Oriented Object Detection with RT-DETR

<div align="center">

**Improved RT-DETR for Oriented Ship Detection in SAR Imagery**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0.1-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.2](https://img.shields.io/badge/CUDA-12.2-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GPU RTX A6000](https://img.shields.io/badge/GPU-NVIDIA%20RTX%20A6000-76b900.svg)](https://www.nvidia.com/en-us/design-visualization/rtx-a6000/)

</div>

## 📖 Introduction

Trans-OBB is an **Oriented Bounding Box (OBB)** detection project based on an improved **RT-DETR** architecture within the **Ultralytics** framework, targeting **dense ship detection in SAR imagery**.

## 🔧 Environment Setup

### Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX A6000 (48GB) |
| CUDA | 12.2 |
| Driver | 535.129.03 |

### Core Dependencies (Required)

This is the minimal set of dependencies needed for this project:

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

```bash
pip install \
    ultralytics==8.3.100 \
    ultralytics-thop>=2.0.0 \
    numpy>=1.23.0 \
    opencv-python>=4.6.0 \
    opencv-python-headless>=4.6.0 \
    pillow>=7.1.2 \
    matplotlib>=3.3.0 \
    seaborn>=0.11.0 \
    pandas>=1.1.4 \
    pyyaml>=5.3.1 \
    scipy>=1.4.1 \
    scikit-learn>=1.3.0 \
    tqdm>=4.64.0 \
    psutil \
    py-cpuinfo \
    requests>=2.23.0 \
    pycocotools>=2.0.7 \
    albumentations>=1.4.6 \
    tensorboard>=2.13.0 \
    onnx>=1.12.0 \
    onnxruntime-gpu>=1.19.0 \
    einops>=0.8.0 \
    timm>=1.0.0 \
    shapely>=2.0.0 \
    huggingface-hub>=0.20.0
```

## 📊 Dataset

### Data Format

This project uses the **DOTA format** (OBB rotated boxes) with the following directory structure:

```
dataset/
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
├── labels/
│   ├── train/          # YOLO OBB format annotations
│   └── val/            # YOLO OBB format annotations
└── labelTxt/           # Original DOTA format annotations (optional)
```

Annotations use the **YOLO OBB normalized format**: `class_id x1 y1 x2 y2 x3 y3 x4 y4` (8 normalized coordinates).

### Data Access

````
通过网盘分享的文件：DenseSARShipDataSet.rar
链接: https://pan.baidu.com/s/1p9xgsDujQJzmt0G6PkHYmg?pwd=xubv 提取码: xubv
````

## 🚀 Training

```bash
# OBB training on DenseSARShip dataset
yolo train model=ultralytics/cfg/models/rt-detr/final_model_light.yaml \
    data=ultralytics/cfg/datasets/DOTAv1.yaml \
    epochs=2000 \
    imgsz=640 \
    batch=16 \
    device=0 \
    workers=16 \
    optimizer=SGD \
    lr0=0.01 \
    momentum=0.937 \
    weight_decay=0.0005 \
    warmup_epochs=3 \
    warmup_momentum=0.8 \
    warmup_bias_lr=0.1 \
    cos_lr=True \
    box=7.5 \
    cls=0.5 \
    dfl=1.5 \
    close_mosaic=10 \
    project=runs/obb \
    name=trans-obb-exp
```

Or via Python:

```python
from ultralytics import YOLO

# Load model configuration
model = YOLO("ultralytics/cfg/models/rt-detr/final_model_light.yaml")

# Train
results = model.train(
    data="ultralytics/cfg/datasets/DOTAv1.yaml",
    epochs=2000,
    imgsz=640,
    batch=16,
    device=0,
    workers=16,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
)
```
