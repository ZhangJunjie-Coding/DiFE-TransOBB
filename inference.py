"""
RT-DETR OBB 模型推理脚本
支持对单张/多张图片进行旋转目标检测推理，后处理包含 NMS，与验证逻辑一致。
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.ops import nms_rotated
from ultralytics.utils.plotting import Annotator, colors


def normalize_angle_and_swap_wh(xywhr):
    """将角度限制在 [0, π/2)，必要时交换 w 和 h。"""
    x, y, w, h, theta = xywhr.unbind(-1)

    mask = theta < 0
    new_w = torch.where(mask, h, w)
    new_h = torch.where(mask, w, h)
    new_theta = torch.where(mask, theta + torch.pi / 2, theta)

    mask2 = new_theta >= torch.pi / 2
    final_w = torch.where(mask2, new_h, new_w)
    final_h = torch.where(mask2, new_w, new_h)
    final_theta = torch.where(mask2, new_theta - torch.pi / 2, new_theta)

    return torch.stack([x, y, final_w, final_h, final_theta], dim=-1)


def letterbox_image(im, imgsz):
    """LetterBox 预处理：square + scale_fill"""
    letterbox = LetterBox(new_shape=imgsz, auto=False, scale_fill=True)
    return letterbox(image=im)


def preprocess(img_path, imgsz, device):
    """读取并预处理单张图片"""
    im0 = cv2.imread(str(img_path))
    if im0 is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    im = letterbox_image(im0, imgsz)  # letterbox
    im = im.transpose((2, 0, 1))[::-1]  # HWC -> CHW, BGR -> RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).float() / 255.0
    im = im.unsqueeze(0).to(device)
    return im, im0


def postprocess(preds, orig_shape, imgsz, conf_thres=0.25, nms_thres=0.3):
    """
    后处理：置信度过滤 + per-class NMS + 角度归一化 + 坐标缩放

    Args:
        preds: 模型原始输出 [bs, 300, 5 + num_classes]
        orig_shape: 原始图像 (H, W)
        imgsz: 推理尺寸
        conf_thres: 置信度阈值
        nms_thres: NMS IoU 阈值

    Returns:
        torch.Tensor: shape (N, 7) = [x, y, w, h, theta, score, cls]
    """
    nd = preds.shape[-1]
    bboxes, scores = preds.split((5, nd - 5), dim=-1)  # [1, 300, 5], [1, 300, num_cls]

    # 取第一个 batch
    bbox = bboxes[0]  # (300, 5)
    score = scores[0]  # (300, num_cls)

    # 缩放 bbox 到输入尺寸的像素坐标
    bbox[..., :4] *= imgsz

    # 置信度过滤
    max_score, cls = score.max(-1)  # (300,)
    idx = max_score > conf_thres
    pred = torch.cat([bbox[idx], max_score[idx, None], cls[idx, None]], dim=-1)

    if len(pred) == 0:
        return torch.zeros((0, 7), device=preds.device)

    # 按置信度排序
    pred = pred[pred[:, 5].argsort(descending=True)]

    # Per-class NMS
    keep = []
    unique_cls = pred[..., 6].unique()
    for c in unique_cls:
        class_mask = pred[..., 6] == c
        class_boxes = pred[class_mask]
        if len(class_boxes) > 0:
            nms_indices = nms_rotated(class_boxes[:, :5], class_boxes[:, 5], threshold=nms_thres)
            keep.append(class_boxes[nms_indices])

    pred = torch.cat(keep, dim=0) if keep else torch.zeros((0, 7), device=preds.device)

    # 缩放回原始图像尺寸
    oh, ow = orig_shape[:2]
    pred[..., [0, 2]] *= ow / imgsz
    pred[..., [1, 3]] *= oh / imgsz

    # 角度归一化
    pred[..., :5] = normalize_angle_and_swap_wh(pred[..., :5])

    return pred


def draw_obb(img, obb_preds, names, save_path=None):
    """在图片上绘制旋转检测框"""
    annotator = Annotator(img.copy(), line_width=2)

    for *xywhr, conf, cls in obb_preds.tolist():
        # 构建 OBB tensor [x, y, w, h, theta]
        rbox = torch.tensor([xywhr])
        # 转换为 4 个角点
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8).tolist()[0]

        # 取颜色
        c = int(cls)
        color = colors(c, True)

        # 画四边形
        pts = np.array([
            [poly[0], poly[1]],
            [poly[2], poly[3]],
            [poly[4], poly[5]],
            [poly[6], poly[7]],
        ], dtype=np.int32)
        cv2.polylines(annotator.im, [pts], isClosed=True, color=color, thickness=2)

        # 标签
        label = f"{names[c]} {conf:.2f}"
        # 文字放在第一个顶点
        cv2.putText(annotator.im, label, (int(poly[0]), int(poly[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    result = annotator.result()
    if save_path:
        cv2.imwrite(str(save_path), result)
        print(f"结果已保存至: {save_path}")
    return result


def run_inference(model_path, source, imgsz=640, conf_thres=0.25, nms_thres=0.3,
                  device=None, save_dir=None, show=False):
    """主推理流程"""
    # 设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # 加载模型
    from ultralytics import RTDETR
    print(f"Loading model from {model_path}...")
    model = RTDETR(model=model_path)
    model = model.model.to(device)
    model.eval()
    # 兼容: 从 model.yaml 加载 weights 的情况
    if hasattr(model, 'names'):
        names = model.names
    else:
        names = {0: "ship"}  # fallback

    # 收集图片
    source = Path(source)
    if source.is_dir():
        img_paths = list(source.glob("*.jpg")) + list(source.glob("*.png")) + list(source.glob("*.bmp"))
    else:
        img_paths = [source]

    if not img_paths:
        raise FileNotFoundError(f"未找到图片: {source}")

    # 输出目录
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path("inference_results")
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"共 {len(img_paths)} 张图片，推理尺寸: {imgsz}, 置信度阈值: {conf_thres}, NMS阈值: {nms_thres}")

    for img_path in img_paths:
        print(f"\n推理: {img_path.name}")

        # 预处理
        im_tensor, im0 = preprocess(img_path, imgsz, device)

        # 推理
        with torch.no_grad():
            preds = model(im_tensor)

        # 如果是 tuple/list，取第一个
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # 后处理（含 NMS）
        results = postprocess(preds, im0.shape, imgsz, conf_thres, nms_thres)

        print(f"  检测到 {len(results)} 个目标")

        # 绘制并保存
        save_path = save_dir / f"{img_path.stem}_pred.jpg"
        draw_obb(im0, results, names, save_path)

        if show:
            cv2.imshow("result", cv2.imread(str(save_path)))
            cv2.waitKey(0)

    print(f"\n完成！结果保存在: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT-DETR OBB Inference")
    parser.add_argument("--model", type=str, required=True, default='/home/zjj/baseLineCode/ultralytics-main/runs/obb/a_best.pt',
                        help="模型路径 (.pt 或 .yaml)")
    parser.add_argument("--source", type=str, required=True, default='/home/zjj/dataset/DenseSARShipDataSet_noise_remedy/images/val',
                        help="图片路径或目录")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理尺寸 (默认: 640)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--nms", type=float, default=0.3,
                        help="NMS IoU 阈值 (默认: 0.3)")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备, 如 0/cuda/cpu")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="结果保存目录")
    parser.add_argument("--show", action="store_true",
                        help="显示推理结果")

    args = parser.parse_args()
    run_inference(
        model_path=args.model,
        source=args.source,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        nms_thres=args.nms,
        device=args.device,
        save_dir=args.save_dir,
        show=args.show,
    )
