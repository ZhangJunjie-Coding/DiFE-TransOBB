# Trans-OBB 🚀 | 基于 RT-DETR 的旋转目标检测

<div align="center">
**基于改进 RT-DETR 的 SAR 图像舰船旋转目标检测**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0.1-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.2](https://img.shields.io/badge/CUDA-12.2-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GPU RTX A6000](https://img.shields.io/badge/GPU-NVIDIA%20RTX%20A6000-76b900.svg)](https://www.nvidia.com/en-us/design-visualization/rtx-a6000/)

</div>

## 📖 项目简介

Trans-OBB 是一个基于 **Ultralytics** 框架改进的 **RT-DETR** 旋转目标检测（Oriented Bounding Box, OBB）项目，主要针对 **SAR 图像密集舰船检测** 任务。

## 🔧 环境配置

### 硬件环境

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA RTX A6000 (48GB) |
| CUDA | 12.2 |
| Driver | 535.129.03 |

### 核心依赖（必需）

以下为此项目实际需要的最小依赖集：

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



## 📊 数据集

### 数据集格式

本项目的标注格式基于 **DOTA 格式**（OBB 旋转框），数据结构如下：

```
dataset/
├── images/
│   ├── train/          # 训练图像
│   └── val/            # 验证图像
├── labels/
│   ├── train/          # YOLO OBB 格式标注
│   └── val/            # YOLO OBB 格式标注
└── labelTxt/           # DOTA 原始格式标注（可选）
```

标注格式采用 **YOLO OBB 归一化格式**：`class_id x1 y1 x2 y2 x3 y3 x4 y4`（8 点归一化坐标）。

### 数据访问

````
通过网盘分享的文件：DenseSARShipDataSet.rar
链接: https://pan.baidu.com/s/1p9xgsDujQJzmt0G6PkHYmg?pwd=xubv 提取码: xubv
````

### 训练

```bash
# OBB 训练（使用 DenseSARShip 数据集）
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

或使用 Python：

```python
from ultralytics import YOLO

# 加载模型配置
model = YOLO("ultralytics/cfg/models/rt-detr/final_model_light.yaml")

# 训练
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
