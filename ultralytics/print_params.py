#
# @author ZhangJunjie
#
from ultralytics.models import RTDETR
import warnings
import torch

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # model = RTDETR(model='ultralytics/cfg/models/rt-detr/rtdetr-l-Fusion-4Head.yaml')
    model = RTDETR(model='ultralytics/cfg/models/rt-detr/final_model_light.yaml')
    model.info()
    model.train(pretrained=False, data='ultralytics/cfg/datasets/DOTAv1.yaml',
                epochs=2000,
                batch=4,
                device=0,
                imgsz=640,
                workers=16,
                # optimizer='AdamW',
                # lr0=0.001,
                # momentum=0.9,  # 标准 beta1
                # resume=False
                resume=True
                )