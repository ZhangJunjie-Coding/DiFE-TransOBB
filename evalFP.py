"""
计算测试集 FP (False Positive) 和 FP/image
用法: python evalFP.py --weights /path/to/model.pt

支持: RTDETR (OBB task), YOLO (detect task)
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import torch

from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.engine.model import Model


# 模块级变量，用于从 validator 中传出 FP 数据
_fp_data = {}


class _RTDETRFPValidator(RTDETRValidator):
    """Custom validator that captures raw TP/conf stats for FP analysis."""

    def __call__(self, trainer=None, model=None):
        result = super().__call__(trainer=trainer, model=model)
        if hasattr(self, 'stats') and self.stats:
            _fp_data.clear()
            _fp_data['tp'] = [t.clone() for t in self.stats.get('tp', [])]
            _fp_data['conf'] = [c.clone() for c in self.stats.get('conf', [])]
            _fp_data['pred_cls'] = [c.clone() for c in self.stats.get('pred_cls', [])]
            _fp_data['target_cls'] = self.stats.get('target_cls', [])
        return result


class _YOLOFPValidator(DetectionValidator):
    """Custom validator for YOLO models."""

    def __call__(self, trainer=None, model=None):
        result = super().__call__(trainer=trainer, model=model)
        if hasattr(self, 'stats') and self.stats:
            _fp_data.clear()
            _fp_data['tp'] = [t.clone() for t in self.stats.get('tp', [])]
            _fp_data['conf'] = [c.clone() for c in self.stats.get('conf', [])]
            _fp_data['pred_cls'] = [c.clone() for c in self.stats.get('pred_cls', [])]
            _fp_data['target_cls'] = self.stats.get('target_cls', [])
        return result


def compute_fp(weights_path, data_yaml='ultralytics/cfg/datasets/DOTAv1.yaml',
               iou_thr=0.5, conf_thr=0.001, device='0', imgsz=640,
               batch=4, workers=8, split='test'):
    """
    计算测试集上的 FP 数量、FP/image、按类别和置信度区间统计 FP 分布。

    Args:
        weights_path: 模型权重路径 (.pt)
        data_yaml: 数据集 YAML 路径
        iou_thr: IoU 阈值
        conf_thr: 最低置信度阈值
        device: CUDA 设备号 或 cpu
        imgsz: 输入图像尺寸
        batch: 批大小
        workers: DataLoader 进程数
        split: 数据集分割 (test / val)

    Returns:
        (total_fp: int, fp_per_image: float)
    """
    global _fp_data
    _fp_data.clear()

    # ---- 加载模型 ----
    model = Model(weights_path)

    # ---- 自动选择 validator ----
    task = getattr(model, 'task', 'detect')
    ValidatorCls = _RTDETRFPValidator if task == 'obb' else _YOLOFPValidator

    # ---- 运行验证 ----
    model.val(
        validator=ValidatorCls,
        data=data_yaml,
        split=split,
        batch=batch,
        device=device,
        imgsz=imgsz,
        workers=workers,
        iou=iou_thr,
        conf=conf_thr,
        plots=False,
        save_json=False,
        save_txt=False,
        verbose=False,
    )

    tp_list = _fp_data.get('tp', [])
    if not tp_list:
        print("ERROR: No predictions found. Try lowering conf_thr (e.g., 0.001 -> 0).")
        return None, None

    # ---- 计算 FP ----
    all_tp = torch.cat(tp_list, dim=0).cpu().numpy()          # shape: (N_det, n_iou_thr)
    tp_iou0 = all_tp[:, 0].astype(bool)                        # True=TP, False=FP (IoU>=0.5)
    total_det = len(tp_iou0)
    total_tp = int(tp_iou0.sum())
    total_fp = total_det - total_tp
    num_images = len(tp_list)

    # ---- 输出汇总 ----
    model_name = weights_path.replace('\\', '/').split('/')[-1]
    print(f"\n{'='*60}")
    print(f"  FP Analysis: {model_name}")
    print(f"  IoU threshold: {iou_thr}  |  Conf threshold: {conf_thr}")
    print(f"{'='*60}")
    print(f"  {'Total images:':<24s} {num_images}")
    print(f"  {'Total detections:':<24s} {total_det}")
    print(f"  {'True Positives  (TP):':<24s} {total_tp}")
    print(f"  {'False Positives (FP):':<24s} {total_fp}")
    print(f"  {'FP per image:':<24s} {total_fp / num_images:.2f}")
    print(f"  {'TP per image:':<24s} {total_tp / num_images:.2f}")
    if total_det > 0:
        print(f"  {'FP ratio:':<24s} {total_fp / total_det * 100:.1f}%")
    print(f"{'='*60}")

    # ---- 按类别统计 FP ----
    pred_cls_list = _fp_data.get('pred_cls', [])
    if pred_cls_list and total_fp > 0:
        all_pred_cls = torch.cat(pred_cls_list, dim=0).cpu().numpy()
        fp_mask = ~tp_iou0
        fp_classes = all_pred_cls[fp_mask].astype(int)

        # 获取类别名称
        names = getattr(model, 'names', {})
        if not names:
            # 尝试从 validator.metrics 获取
            metrics = getattr(model, 'metrics', None)
            if hasattr(metrics, 'names'):
                names = metrics.names

        unique_cls = np.unique(fp_classes)
        print(f"\n{'─'*60}")
        print(f"  FP per class (IoU >= {iou_thr}):")
        print(f"  {'Class':<24s} {'FP count':<10s} {'Pct of FP'}")
        print(f"  {'─'*44}")
        for cls_id in sorted(unique_cls):
            count = int((fp_classes == cls_id).sum())
            name = names.get(cls_id, f"class_{cls_id}") if names else f"class_{cls_id}"
            print(f"  {name:<24s} {count:<10} {count / total_fp * 100:5.1f}%")
        print(f"  {'─'*44}")

    # ---- 按置信度区间统计 FP 分布 ----
    conf_list = _fp_data.get('conf', [])
    if conf_list and total_fp > 0:
        all_conf = torch.cat(conf_list, dim=0).cpu().numpy()
        fp_confs = all_conf[~tp_iou0]

        bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        print(f"\n{'─'*60}")
        print(f"  FP distribution by confidence (total FP: {len(fp_confs)}):")
        print(f"  {'Conf Range':<16s} {'FP count':<10s} {'Pct':<8s}  Histogram")
        print(f"  {'─'*52}")
        for lo, hi in bins:
            count = int(((fp_confs >= lo) & (fp_confs < hi)).sum())
            pct = count / len(fp_confs) * 100 if len(fp_confs) > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  [{lo:.1f}, {hi:.1f})  {count:<10} {pct:5.1f}%    {bar}")
        print(f"  {'─'*52}")

    return total_fp, total_fp / num_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute FP (False Positives) and FP/image for object detection models"
    )
    parser.add_argument('--weights', type=str, required=True,default='/home/zjj/baseLineCode/ultralytics-main/runs/obb/light.pt',
                        help='Path to trained model weights (.pt)')
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/DOTAv1.yaml',
                        help='Dataset YAML config path')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Confidence threshold (default: 0.001)')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device (0, 1, cpu)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--batch', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--workers', type=int, default=8,
                        help='DataLoader workers (default: 8)')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split: test or val (default: test)')
    args = parser.parse_args()

    compute_fp(
        weights_path=args.weights,
        data_yaml=args.data,
        iou_thr=args.iou,
        conf_thr=args.conf,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        split=args.split,
    )
