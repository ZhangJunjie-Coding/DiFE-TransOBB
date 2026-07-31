#
# @author ZhangJunjie
#
from ultralytics.models import RTDETR
import warnings
import torch

# 先保存原始的 torch.sqrt，方便恢复和内部调用

# # 原始 Tensor.sqrt 方法保存
# torch._original_tensor_sqrt = torch.Tensor.sqrt
#
# # 重定义 Tensor.sqrt
# def safe_tensor_sqrt(self):
#     if torch.any(self < 0):
#         print("Tensor.sqrt input has negative values:", self[self < 0])
#     return torch._original_tensor_sqrt(self.clamp(min=0))

# 替换 Tensor.sqrt 方法
# torch.Tensor.sqrt = safe_tensor_sqrt

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    model = RTDETR(model='ultralytics/cfg/models/rt-detr/final_model_light.yaml')
    # model = RTDETR(model='/home/lab/zhangxm/zjj/baseLineCode/ultralytics-main/ultralytics/runs/obb/train56/weights/best.pt')
    model.train(pretrained=False, data='ultralytics/cfg/datasets/DOTAv1.yaml',
                epochs=2000,
                batch=16,
                device=0,
                imgsz=640,
                workers=16,
                resume=False)

    # translate=0.0, scale=0.0, fliplr=0.0, mosaic=0.0, crop_fraction=1.8, close_mosaic=10)
