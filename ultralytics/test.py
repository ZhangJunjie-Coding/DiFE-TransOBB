import numpy as np
import cv2
import torch


def xyxyxyxy2xywhr(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation]. Rotation values are
    returned in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | torch.Tensor): Input box corners [xy1, xy2, xy3, xy4] of shape (n, 8).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted data in [cx, cy, w, h, rotation] format of shape (n, 5).
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


def xywhr2xyxyxyxy(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i: i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def polygon_to_xywha(points):
    """将四点坐标转换为 xywha 格式"""
    rect = cv2.minAreaRect(points.astype(np.float32).reshape(4, 2))
    (xc, yc), (w, h), angle = rect
    return [xc, yc, w, h, angle]


def convert_dota_to_xywha(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    results = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 9:
            continue  # 忽略错误格式
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))  # 8个点
        x = np.array([coords])
        # pts = np.array(coords, dtype=np.float32).reshape(4, 2)
        a, b, c = xyxyxyxy2xywhr(x)
        print(a)
        print(b)
        print(c)


# 读取文件
file_path = "/home/lab/zhangxm/dataset/SDFSD/labels/train/ABS001__843__.txt"
with open(file_path, 'r') as f:
    lines = f.readlines()

# 解析每行数据
xyxyxyxy_data = []
for line in lines:
    parts = list(map(float, line.strip().split()))
    class_id = parts[0]
    corners = parts[1:]  # 提取 x1 y1 x2 y2 x3 y3 x4 y4
    xyxyxyxy_data.append(corners)

# 转换为 numpy 数组 (n_objects, 8)
xyxyxyxy_array = np.array(xyxyxyxy_data, dtype=np.float32)

# 调用函数转换
xywhr_result = xyxyxyxy2xywhr(xyxyxyxy_array)

# 提取 xywh（去掉旋转角度）
xywh_result = xywhr_result[:, :4]  # 只保留 cx, cy, w, h

# 打印结果
print("原始数据（xyxyxyxy）:", xyxyxyxy_array)
print("转换后（xywhr）:", xywhr_result)
print("仅 xywh:", xywh_result)

# 保存到新文件（可选）
output_path = file_path.replace(".txt", "_xywh.txt")
np.savetxt(output_path, xywh_result, fmt='%.6f')  # 保存为6位小数
print(f"结果已保存到 {output_path}")