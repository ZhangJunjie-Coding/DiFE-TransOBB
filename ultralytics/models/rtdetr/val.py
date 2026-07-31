# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
from pathlib import Path
import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, colorstr, ops
from ultralytics.utils.metrics import OBBMetrics, batch_probiou
from ultralytics.utils.plotting import output_to_rotated_target, plot_images
__all__ = ("RTDETRValidator",)  # tuple or list


class RTDETRDataset(YOLODataset):
    """
    Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.
    """

    def __init__(self, *args, data=None, **kwargs):
        """Initialize the RTDETRDataset class by inheriting from the YOLODataset class."""
        super().__init__(*args, data=data, **kwargs)

    def load_image(self, i, rect_mode=False):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """
        Build transformation pipeline for the dataset.

        Args:
            hyp (dict, optional): Hyperparameters for transformations.

        Returns:
            (Compose): Composition of transformation functions.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            transforms = Compose([])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_obb=self.use_obb,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms


class RTDETRValidator(DetectionValidator):
    """
    RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies Non-maximum suppression for
    post-processing, and updates evaluation metrics accordingly.

    Examples:
        >>> from ultralytics.models.rtdetr import RTDETRValidator
        >>> args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        >>> validator = RTDETRValidator(args=args)
        >>> validator()

    Note:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (RTDETRDataset): Dataset configured for RT-DETR validation.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            task=self.args.task,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize OBBValidator and set task to 'obb', metrics to OBBMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "obb"
        self.args.conf = 0.001  # DETR模型验证使用低阈值
        self.metrics = OBBMetrics(save_dir=self.save_dir, plot=True)

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.is_dota = isinstance(val, str) and "DOTA" in val  # check if dataset is DOTA format

    def postprocess(self, preds):
        """
        Postprocess the raw predictions from the RT-DETR model.

        Unlike standard YOLO detectors, DETR-based models use bipartite matching and self-attention
        between queries to naturally suppress duplicate detections, so NMS is not needed.
        Only confidence thresholding and sorting are applied.

        Args:
            preds (List | Tuple | torch.Tensor): Raw predictions from the model, shape (bs, 300, 5+nc).

        Returns:
            (List[torch.Tensor]): List of processed predictions for each image in batch,
                each tensor shape (N, 7) = [x, y, w, h, theta, score, cls] in imgsz coordinates.
        """
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((5, nd - 5), dim=-1)

        # DEBUG: log prediction statistics (只在第一个batch输出)
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            LOGGER.info(f"[DEBUG val] preds[0] shape={preds[0].shape}, nd={nd}")
            LOGGER.info(f"[DEBUG val] bboxes[0] (first 5 queries): {bboxes[0, :5, :].tolist()}")
            LOGGER.info(f"[DEBUG val] scores[0] (first 10 queries): {scores[0, :10, :].squeeze(-1).tolist()}")
            LOGGER.info(f"[DEBUG val] scores min={scores.min():.6f} max={scores.max():.6f} mean={scores.mean():.6f}")
            n_pass = (scores.squeeze(-1) > self.args.conf).sum(dim=1)
            LOGGER.info(f"[DEBUG val] queries passing conf>({self.args.conf}) per image: {n_pass.tolist()}")

        # Scale normalized coordinates to imgsz pixel coordinates
        bboxes[..., :4] *= self.args.imgsz
        outputs = [torch.zeros((0, 7), device=bboxes.device) for _ in range(bs)]

        for i, bbox in enumerate(bboxes):  # (300, 5)
            score, cls = scores[i].max(-1)  # (300, )
            idx = score > self.args.conf
            pred = torch.cat([bbox[idx], score[idx][..., None], cls[idx][..., None]], dim=-1)
            if len(pred) > 0:
                pred = pred[pred[:, 5].argsort(descending=True)]
                outputs[i] = pred

        return outputs

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        # iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :5], detections[:, -1:]], dim=-1))
        iou = batch_probiou(gt_bboxes, detections[:, :5])
        return self.match_predictions(detections[:, 6], gt_cls, iou)

    def _prepare_batch(self, si, batch):
        """Prepare batch data for OBB validation with proper scaling and formatting."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
            # ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)  # native-space labels
        ops.scale_boxes(imgsz, bbox, ori_shape, xywh=True)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepare predictions by scaling bounding boxes to original image dimensions."""
        predn = pred.clone()
        ops.scale_boxes(pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], xywh=True)  # native-space pred
        # ops.scale_boxes(pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True)  # native-space pred
        return predn

    def plot_predictions(self, batch, preds, ni):
        """Plot predicted bounding boxes on input images and save the result."""
        plot_images(
            batch["img"],
            *output_to_rotated_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def pred_to_json(self, predn, filename):
        """Convert YOLO predictions to COCO JSON format with rotated bounding box information."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        for i, (r, b) in enumerate(zip(rbox.tolist(), poly.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(predn[i, 5].item())],
                    "score": round(predn[i, 4].item(), 5),
                    "rbox": [round(x, 3) for x in r],
                    "poly": [round(x, 3) for x in b],
                }
            )

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates using the Results class."""
        import numpy as np

        from ultralytics.engine.results import Results

        rboxes = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)
        # xywh, r, conf, cls
        obb = torch.cat([rboxes, predn[:, 4:6]], dim=-1)
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            obb=obb,
        ).save_txt(file, save_conf=save_conf)

    def eval_json(self, stats):
        """Evaluate YOLO output in JSON format and save predictions in DOTA format."""
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # predictions
            pred_txt = self.save_dir / "predictions_txt"  # predictions
            pred_txt.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))
            # Save split results
            LOGGER.info(f"Saving predictions with DOTA format to {pred_txt}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                classname = self.names[d["category_id"] - 1].replace(" ", "-")
                p = d["poly"]

                with open(f"{pred_txt / f'Task1_{classname}'}.txt", "a", encoding="utf-8") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
            # Save merged results, this could result slightly lower map than using official merging script,
            # because of the probiou calculation.
            pred_merged_txt = self.save_dir / "predictions_merged_txt"  # predictions
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)
            LOGGER.info(f"Saving merged predictions with DOTA format to {pred_merged_txt}...")
            for d in data:
                image_id = d["image_id"].split("__")[0]
                pattern = re.compile(r"\d+___\d+")
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                bbox, score, cls = d["rbox"], d["score"], d["category_id"] - 1
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  # classes
                scores = bbox[:, 5]  # scores
                b = bbox[:, :5].clone()
                b[:, :2] += c
                # 0.3 could get results close to the ones from official merging script, even slightly better.
                i = ops.nms_rotated(b, scores, 0.3)
                bbox = bbox[i]

                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    p = [round(i, 3) for i in x[:-2]]  # poly
                    score = round(x[-2], 3)

                    with open(f"{pred_merged_txt / f'Task1_{classname}'}.txt", "a", encoding="utf-8") as f:
                        f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

        return stats
