# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.visualizeutils import check_coordinate_system


class HungarianMatcher(nn.Module):
    """
    A module implementing the HungarianMatcher, which is a differentiable module to solve the assignment problem in an
    end-to-end fashion.

    HungarianMatcher performs optimal assignment over the predicted and ground truth bounding boxes using a cost
    function that considers classification scores, bounding box coordinates, and optionally, mask predictions.

    Attributes:
        cost_gain (dict): Dictionary of cost coefficients: 'class', 'bbox', 'giou', 'mask', and 'dice'.
        use_fl (bool): Indicates whether to use Focal Loss for the classification cost calculation.
        with_mask (bool): Indicates whether the model makes mask predictions.
        num_sample_points (int): The number of sample points used in mask cost calculation.
        alpha (float): The alpha factor in Focal Loss calculation.
        gamma (float): The gamma factor in Focal Loss calculation.

    Methods:
        forward: Computes the assignment between predictions and ground truths for a batch.
        _cost_mask: Computes the mask cost and dice cost if masks are predicted.
    """

    def __init__(self, cost_gain=None, use_fl=True, with_mask=False, num_sample_points=12544, alpha=0.25, gamma=2.0):
        """Initialize a HungarianMatcher module for optimal assignment of predicted and ground truth bounding boxes."""
        super().__init__()
        if cost_gain is None:
            cost_gain = {"class": 1, "bbox": 5, "giou": 2, "mask": 1, "dice": 1}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None):
        """
        Forward pass for HungarianMatcher. Computes costs based on prediction and ground truth and finds the optimal
        matching between predictions and ground truth based on these costs.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (batch_size, num_queries, 5).
            pred_scores (torch.Tensor): Predicted scores with shape (batch_size, num_queries, num_classes).
            gt_cls (torch.Tensor): Ground truth classes with shape (num_gts, ).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (num_gts, 5).
            gt_groups (List[int]): List of length equal to batch size, containing the number of ground truths for
                each image.
            masks (torch.Tensor, optional): Predicted masks with shape (batch_size, num_queries, height, width).
            gt_mask (List[torch.Tensor], optional): List of ground truth masks, each with shape (num_masks, Height, Width).

        Returns:
            (List[Tuple[torch.Tensor, torch.Tensor]]): A list of size batch_size, each element is a tuple (index_i, index_j), where:
                - index_i is the tensor of indices of the selected predictions (in order)
                - index_j is the tensor of indices of the corresponding selected ground truth targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, nq, nc = pred_scores.shape

        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # (batch_size * num_queries, num_classes)
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        # (batch_size * num_queries, 5)
        pred_bboxes = pred_bboxes.detach().view(-1, 5)

        # Compute the classification cost
        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * (pred_scores ** self.gamma) * (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        # Compute the L1 cost between boxes (all 5 dims: cx, cy, w, h, angle)
        # Uses torch.cdist to include angle naturally, same as AO2-DETR
        cost_bbox = torch.cdist(pred_bboxes, gt_bboxes, p=1)  # (bs*num_queries, num_gt)

        # Compute the GIoU cost between boxes, (bs*num_queries, num_gt)
        cost_giou = 1.0 - batch_probiou(pred_bboxes, gt_bboxes)

        # Final cost matrix for Hungarian matching
        C = (
            self.cost_gain["class"] * cost_class
            + self.cost_gain["bbox"] * cost_bbox
            + self.cost_gain["giou"] * cost_giou
        )

        # Compute the mask cost and dice cost
        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)

        # Set invalid values (NaNs and infinities) to 0 (fixes ValueError: matrix contains invalid numeric entries)
        C[C.isnan() | C.isinf()] = 0.0

        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
        gt_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)  # (idx for queries, idx for gt)
        return [
            (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups[k])
            for k, (i, j) in enumerate(indices)
        ]



def get_cdn_group(
        batch, num_classes, num_queries, class_embed, num_dn=100, cls_noise_ratio=0.5, box_noise_scale=1.0,
        training=False
):
    """
    Get contrastive denoising training group with positive and negative samples from ground truths.

    Args:
        batch (dict): A dict that includes 'gt_cls' (torch.Tensor with shape (num_gts, )), 'gt_bboxes'
            (torch.Tensor with shape (num_gts, 4)), 'gt_groups' (List[int]) which is a list of batch size length
            indicating the number of gts of each image.
        num_classes (int): Number of classes.
        num_queries (int): Number of queries.
        class_embed (torch.Tensor): Embedding weights to map class labels to embedding space.
        num_dn (int, optional): Number of denoising queries.
        cls_noise_ratio (float, optional): Noise ratio for class labels.
        box_noise_scale (float, optional): Noise scale for bounding box coordinates.
        training (bool, optional): If it's in training mode.

    Returns:
        padding_cls (Optional[torch.Tensor]): The modified class embeddings for denoising.
        padding_bbox (Optional[torch.Tensor]): The modified bounding boxes for denoising.
        attn_mask (Optional[torch.Tensor]): The attention mask for denoising.
        dn_meta (Optional[Dict]): Meta information for denoising.
    """
    if (not training) or num_dn <= 0 or batch is None:
        return None, None, None, None
    gt_groups = batch["gt_groups"]
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return None, None, None, None

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    # Pad gt to max_num of a batch
    bs = len(gt_groups)
    gt_cls = batch["cls"]  # (bs*num, )
    gt_bbox = batch["bboxes"]  # bs*num, 4
    b_idx = batch["batch_idx"]

    # Each group has positive and negative queries.
    dn_cls = gt_cls.repeat(2 * num_group)  # (2*num_group*bs*num, )
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)  # 2*num_group*bs*num, 4
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # (2*num_group*bs*num, )

    # Positive and negative mask
    # (bs*num*num_group, ), the second total_num*num_group part as negative samples
    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num

    if cls_noise_ratio > 0:
        # Half of bbox prob
        mask = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        # Randomly put a new one here
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device)
        dn_cls[idx] = new_label

    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)

        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale  # 2*num_group*bs*num, 4

        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-6)  # inverse sigmoid

    num_dn = int(max_nums * 2 * num_group)  # total denoising queries
    # class_embed = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=class_embed.device)])
    dn_cls_embed = class_embed[dn_cls]  # bs*num * 2 * num_group, 256
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)

    map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)

    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])
    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
    padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    # Match query cannot see the reconstruct
    attn_mask[num_dn:, :num_dn] = True
    # Reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i: max_nums * 2 * (i + 1), max_nums * 2 * (i + 1): num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i: max_nums * 2 * (i + 1), : max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i: max_nums * 2 * (i + 1), max_nums * 2 * (i + 1): num_dn] = True
            attn_mask[max_nums * 2 * i: max_nums * 2 * (i + 1), : max_nums * 2 * i] = True
    dn_meta = {
        "dn_pos_idx": [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
        "dn_num_group": num_group,
        "dn_num_split": [num_dn, num_queries],
    }

    return (
        padding_cls.to(class_embed.device),
        padding_bbox.to(class_embed.device),
        attn_mask.to(class_embed.device),
        dn_meta,
    )


def get_cdn_group_obb(
        batch, num_classes, num_queries, class_embed, num_dn=100, cls_noise_ratio=0.5, box_noise_scale=1.0,
        training=False
):
    """
    构建用于旋转框检测（OBB）的对比去噪训练样本组（CDN），包括正负样本、注意力遮罩、扰动、batch映射等。

    Args:
        batch (dict): 包含 'cls', 'bboxes', 'gt_groups', 'batch_idx' 的数据字典
        num_classes (int): 类别数量
        num_queries (int): 匹配阶段的query数量（不包括denoising）
        class_embed (Tensor): 类别嵌入向量（nn.Embedding.weight）
        num_dn (int): 总共的去噪query数量（实际会按每组最大GT数重新分组）
        cls_noise_ratio (float): 类别扰动比例
        box_noise_scale (float): 边框扰动尺度
        training (bool): 是否为训练模式

    Returns:
        padding_cls: Tensor(bs, num_dn, dim) 带有噪声扰动的类别嵌入
        padding_bbox: Tensor(bs, num_dn, 5) 带有噪声扰动的OBB边框 (xywhr)
        attn_mask: Tensor(num_dn + num_queries, num_dn + num_queries) 注意力遮罩
        dn_meta: 字典，包括 denoising 分组信息和正样本索引
    """
    if (not training) or num_dn <= 0 or batch is None:
        return None, None, None, None

    gt_groups = batch["gt_groups"]
    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0:
        return None, None, None, None

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group

    bs = len(gt_groups)
    gt_cls = batch["cls"]
    gt_bbox = batch["bboxes"]  # xywhr
    b_idx = batch["batch_idx"]

    dn_cls = gt_cls.repeat(2 * num_group)
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)

    neg_idx = torch.arange(total_num * num_group, 2 * total_num * num_group,
                           dtype=torch.long, device=gt_bbox.device)

    # 类别噪声
    if cls_noise_ratio > 0:
        mask = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device)
        dn_cls[idx] = new_label

    # 边框噪声
    if box_noise_scale > 0:
        dn_bbox_copy = dn_bbox.clone()
        xy_noise = torch.randn_like(dn_bbox[..., :2]) * box_noise_scale * 0.05
        dn_bbox_copy[..., :2] += xy_noise

        wh_noise_ratio = torch.randn_like(dn_bbox[..., 2:4]) * box_noise_scale * 0.1
        dn_bbox_copy[..., 2:4] *= (1.0 + wh_noise_ratio)
        dn_bbox_copy[..., 2:4] = torch.clamp(dn_bbox_copy[..., 2:4], min=0.001, max=1.0)

        angle_noise = torch.randn_like(dn_bbox[..., 4:5]) * box_noise_scale * 0.1
        dn_bbox_copy[..., 4:5] += angle_noise
        dn_bbox_copy[..., 4:5] = (dn_bbox_copy[..., 4:5] + torch.pi / 2) % torch.pi - torch.pi / 2

        # 为负样本增加扰动
        rand_part = torch.rand_like(dn_bbox_copy)
        rand_part[neg_idx] += 1.0
        rand_sign = torch.randint_like(dn_bbox_copy, 0, 2) * 2.0 - 1.0
        noise_scales = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.5], device=dn_bbox.device)
        extra_noise = rand_part * rand_sign * noise_scales.unsqueeze(0) * box_noise_scale
        dn_bbox_copy += extra_noise

        dn_bbox_copy[..., :2] = torch.clamp(dn_bbox_copy[..., :2], min=0.0, max=1.0)
        dn_bbox_copy[..., 2:4] = torch.clamp(dn_bbox_copy[..., 2:4], min=0.01, max=1.0)
        dn_bbox_copy[..., 4:5] = (dn_bbox_copy[..., 4:5] + torch.pi / 2) % torch.pi - torch.pi / 2

        # 归一化编码：x y w h 用 logit，角度保持不变
        dn_bbox_transformed = dn_bbox_copy.clone()
        dn_bbox_transformed[..., :4] = torch.logit(
            torch.clamp(dn_bbox_copy[..., :4], min=1e-6, max=1 - 1e-6),
            eps=1e-6
        )
        dn_bbox = dn_bbox_transformed

    num_dn = int(max_nums * 2 * num_group)
    dn_cls_embed = class_embed[dn_cls]
    padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
    padding_bbox = torch.zeros(bs, num_dn, 5, device=gt_bbox.device)

    # ✅ 修正后的批次映射逻辑
    current_idx = 0
    for batch_i, num_gt in enumerate(gt_groups):
        if num_gt > 0:
            for group_i in range(2 * num_group):
                start_pos = group_i * max_nums
                end_pos = start_pos + num_gt

                data_start = current_idx + group_i * total_num
                data_end = data_start + num_gt

                padding_cls[batch_i, start_pos:end_pos] = dn_cls_embed[data_start:data_end]
                padding_bbox[batch_i, start_pos:end_pos] = dn_bbox[data_start:data_end]

        current_idx += num_gt

    # attention mask 构建
    tgt_size = num_dn + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    attn_mask[num_dn:, :num_dn] = True
    for i in range(num_group):
        if i == 0:
            attn_mask[max_nums * 2 * i: max_nums * 2 * (i + 1), max_nums * 2 * (i + 1): num_dn] = True
        if i == num_group - 1:
            attn_mask[max_nums * 2 * i: max_nums * 2 * (i + 1), : max_nums * i * 2] = True
        else:
            attn_mask[max_nums * 2 * i: max_nums * 2 * (i + 1), max_nums * 2 * (i + 1): num_dn] = True
            attn_mask[max_nums * 2 * i: max_nums * 2 * (i + 1), : max_nums * 2 * i] = True

    # 构造 pos_idx（每个图像中属于正样本的位置）
    # 正确生成 dn_pos_idx，确保和 gt_groups 一致（batch 维度）
    dn_pos_idx = []
    for i in range(bs):
        img_pos = []
        for g in range(num_group):
            start = g * max_nums
            end = start + gt_groups[i]
            img_pos.append(torch.arange(start, end, device='cpu'))
        dn_pos_idx.append(torch.cat(img_pos))

    dn_meta = {
        "dn_pos_idx": dn_pos_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_dn, num_queries],
    }

    return (
        padding_cls.to(class_embed.device),
        padding_bbox.to(class_embed.device),
        attn_mask.to(class_embed.device),
        dn_meta,
    )
