import glob
import numpy as np
from shapely.geometry import Polygon
import os

# TODO：请改成你的真实图像尺寸
IMG_W = 640
IMG_H = 640


def poly_area_norm_to_pixel(poly):
    """
    poly: [x1,y1,x2,y2,x3,y3,x4,y4] 归一化坐标
    return: polygon 像素面积
    """
    pts = np.array(poly).reshape(-1, 2)
    pts[:, 0] *= IMG_W
    pts[:, 1] *= IMG_H
    return Polygon(pts).area


def classify_area(area):
    """
    COCO 定义
    small:  area < 32^2
    medium: 32^2 ≤ area < 96^2
    large:  area ≥ 96^2
    """
    if area < 20 * 20:
        return "small"
    elif area <45 * 45:
        return "medium"
    else:
        return "large"


small = 0
medium = 0
large = 0

# 自动读取 train 和 val
base_dir = "/home/zjj/dataset/DenseSARShipDataSet/labels/train"
label_paths = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]
base_dir = "/home/zjj/dataset/DenseSARShipDataSet/labels/val"
label_paths = label_paths + [os.path.join(base_dir, f) for f in os.listdir(base_dir)]

for file in label_paths:

    with open(file, "r") as f:
        for line in f:
            items = line.strip().split()

            # class + 8 关键点
            if len(items) != 9:
                continue

            poly = list(map(float, items[1:]))
            area = poly_area_norm_to_pixel(poly)
            cls = classify_area(area)

            if cls == "small":
                small += 1
            elif cls == "medium":
                medium += 1
            else:
                large += 1

print("======== 目标尺寸统计结果 ========")
print("Small  :", small)
print("Medium :", medium)
print("Large  :", large)
print("Total  :", small + medium + large)
