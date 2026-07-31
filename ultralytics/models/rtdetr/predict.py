# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class RTDETRPredictor(BasePredictor):
    """
    RT-DETR (Real-Time Detection Transformer) Predictor extending the BasePredictor class for making predictions.

    This class leverages Vision Transformers to provide real-time object detection while maintaining high accuracy.
    It supports key features like efficient hybrid encoding and IoU-aware query selection.

    Attributes:
        imgsz (int): Image size for inference (must be square and scale-filled).
        args (dict): Argument overrides for the predictor.
        model (torch.nn.Module): The loaded RT-DETR model.
        batch (list): Current batch of processed inputs.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.rtdetr import RTDETRPredictor
        >>> args = dict(model="rtdetr-l.pt", source=ASSETS)
        >>> predictor = RTDETRPredictor(overrides=args)
        >>> predictor.predict_cli()
    """
    # 推理模式 下进行后处理
    def postprocess(self, preds, img, orig_imgs):
        """
        Postprocess the raw predictions from the model to generate bounding boxes and confidence scores.

        The method filters detections based on confidence and class if specified in `self.args`. It converts
        model predictions to Results objects containing properly scaled bounding boxes.

        Args:
            preds (List | Tuple): List of [predictions, extra] from the model, where predictions contain
                bounding boxes and scores.
            img (torch.Tensor): Processed input images with shape (N, 3, H, W).
            orig_imgs (List | torch.Tensor): Original, unprocessed images.

        Returns:
            (List[Results]): A list of Results objects containing the post-processed bounding boxes, confidence scores,
                and class labels.
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((5, nd - 5), dim=-1)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for bbox, score, orig_img, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):  # (300, 5)
            # bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)  # (300, 1)
            idx = max_score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]  # filter
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow  # scale x coordinates to original width
            pred[..., [1, 3]] *= oh  # scale y coordinates to original height
            pred[..., :5] = self.normalize_angle_and_swap_wh(pred[..., :5])
            results.append(Results(orig_img, path=img_path, names=self.model.names, obb=pred))

        return results

    def normalize_angle_and_swap_wh(self,xywhr):
        """
        将角度限制在 [0, π/2)，必要时交换 w 和 h。
        输入为 [N, 5] 的张量，格式为 [x, y, w, h, theta]，角度单位为弧度。
        """
        x, y, w, h, theta = xywhr.unbind(-1)

        # 如果角度小于 0，则交换 w 和 h，并加 π/2，使角度落入 [0, π/2)
        mask = theta < 0
        new_w = torch.where(mask, h, w)
        new_h = torch.where(mask, w, h)
        new_theta = torch.where(mask, theta + torch.pi / 2, theta)

        # 如果角度 >= π/2，则同样处理（因为有可能是 [0, π]）
        mask2 = new_theta >= torch.pi / 2
        final_w = torch.where(mask2, new_h, new_w)
        final_h = torch.where(mask2, new_w, new_h)
        final_theta = torch.where(mask2, new_theta - torch.pi / 2, new_theta)

        return torch.stack([x, y, final_w, final_h, final_theta], dim=-1)

    def pre_transform(self, im):
        """
        Pre-transforms the input images before feeding them into the model for inference. The input images are
        letterboxed to ensure a square aspect ratio and scale-filled. The size must be square(640) and scale_filled.

        Args:
            im (list[np.ndarray] |torch.Tensor): Input images of shape (N,3,h,w) for tensor, [(h,w,3) x N] for list.

        Returns:
            (list): List of pre-transformed images ready for model inference.
        """
        letterbox = LetterBox(self.imgsz, auto=False, scale_fill=True)
        return [letterbox(image=x) for x in im]
