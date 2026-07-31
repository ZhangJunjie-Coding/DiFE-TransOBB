#
# @author ZhangJunjie
#
import math
import torch

from ultralytics.utils.metrics import batch_probiou

pred = torch.tensor([[100, 100, 40, 20, math.radians(0)]])      # 水平长方形
gt   = torch.tensor([[100, 100, 40, 20, math.radians(90)]])     # 垂直长方形

iou = batch_probiou(pred, gt)   # 如果这个结果 ≈ 1，那就错了！
print("IoU =", iou)
import cv2
import numpy as np

# 一个简单的旋转矩形
pts = np.array([[100, 100], [200, 80], [220, 180], [120, 200]], dtype=np.float32)
rect = cv2.minAreaRect(pts)
print(f"Center: {rect[0]}, Size: {rect[1]}, Angle: {rect[2]}")


def validate_probiou_calculation():
    """验证ProIoU计算的正确性"""

    print("=== ProIoU计算验证 ===")

    # 测试用例1: 相同的框
    same_box = torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.0]])
    iou_same = batch_probiou(same_box, same_box)
    print(f"相同框IoU: {iou_same.item():.4f} (期望: ~1.0)")

    # 测试用例2: 完全重叠但有旋转
    box1 = torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.0]])
    box2 = torch.tensor([[0.5, 0.5, 0.2, 0.3, math.pi / 2]])  # 90度旋转
    iou_rotated = batch_probiou(box1, box2)
    print(f"90度旋转框IoU: {iou_rotated.item():.4f} (期望: ~1.0)")

    # 测试用例3: 不重叠框
    box3 = torch.tensor([[0.2, 0.2, 0.1, 0.1, 0.0]])
    box4 = torch.tensor([[0.8, 0.8, 0.1, 0.1, 0.0]])
    iou_separate = batch_probiou(box3, box4)
    print(f"分离框IoU: {iou_separate.item():.4f} (期望: ~0.0)")

    # 如果这些测试失败，ProIoU函数肯定有问题
    if iou_same < 0.95 or iou_rotated < 0.95 or iou_separate > 0.05:
        print("❌ ProIoU计算有严重问题！")
        return False
    else:
        print("✅ ProIoU计算看起来正常")
        return True
if __name__ == "__main__":
    validate_probiou_calculation()