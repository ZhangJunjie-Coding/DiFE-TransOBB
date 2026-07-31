#
# @author ZhangJunjie
#

def visualizeImg(batch, save_dir="./visualization_output"):
    """
    可视化标注框函数

    Args:
        batch (dict): 包含以下键的字典:
            - 'bboxes': 标注框信息，格式为 [x, y, w, h, r] (xywhr格式)
            - 'batch_idx': 当前框属于第几张图的索引
            - 'im_file': 图片文件路径列表
        save_dir (str): 保存可视化结果的目录路径
    """
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os

    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取批次数据
    bboxes = batch['bboxes']  # shape: (N, 5) -> [x, y, w, h, r]
    batch_idx = batch['batch_idx']  # shape: (N,)
    im_files = batch['im_file']  # 图片路径列表

    # 确保数据类型正确
    if hasattr(bboxes, 'cpu'):
        bboxes = bboxes.cpu().numpy()
    if hasattr(batch_idx, 'cpu'):
        batch_idx = batch_idx.cpu().numpy()

    bboxes = np.array(bboxes, dtype=np.float32)
    batch_idx = np.array(batch_idx, dtype=np.int32)

    # 获取批次中唯一的图片索引
    unique_batch_idx = np.unique(batch_idx)

    # 为每张图片创建可视化
    for img_idx in unique_batch_idx:
        # 获取当前图片的标注框
        mask = batch_idx == img_idx
        current_bboxes = bboxes[mask]

        # 读取图片
        img_path = im_files[int(img_idx)]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"警告: 无法读取图片 {img_path}")
            continue

        # 获取图片尺寸
        h, w = image.shape[:2]

        # 转换BGR到RGB用于matplotlib显示
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 创建图片副本用于绘制
        vis_image = image_rgb.copy()

        # 绘制每个标注框
        for bbox in current_bboxes:
            x_center, y_center, width, height, rotation = bbox

            # 确保数据类型为float
            x_center = float(x_center)
            y_center = float(y_center)
            width = float(width)
            height = float(height)
            rotation = float(rotation)

            # 如果坐标是归一化的，转换为像素坐标
            if x_center <= 1.0 and y_center <= 1.0 and width <= 1.0 and height <= 1.0:
                x_center *= w
                y_center *= h
                width *= w
                height *= h

            # 创建旋转矩形的四个顶点
            # 先创建一个以原点为中心的矩形
            half_w, half_h = width / 2, height / 2
            corners = np.array([
                [-half_w, -half_h],  # 左上
                [half_w, -half_h],  # 右上
                [half_w, half_h],  # 右下
                [-half_w, half_h]  # 左下
            ], dtype=np.float32)

            # 应用旋转变换
            cos_r, sin_r = np.cos(rotation), np.sin(rotation)
            rotation_matrix = np.array([
                [cos_r, -sin_r],
                [sin_r, cos_r]
            ], dtype=np.float32)

            # 旋转顶点
            rotated_corners = corners @ rotation_matrix.T

            # 平移到实际位置 - 使用显式的数组操作避免+=错误
            rotated_corners = rotated_corners + np.array([x_center, y_center], dtype=np.float32)

            # 转换为整数坐标
            points = rotated_corners.astype(np.int32)

            # 确保坐标在图像范围内
            points = np.clip(points, 0, [w - 1, h - 1])

            # 绘制旋转矩形
            cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)

            # 可选：绘制中心点
            center_x, center_y = int(np.clip(x_center, 0, w - 1)), int(np.clip(y_center, 0, h - 1))
            cv2.circle(vis_image, (center_x, center_y), 3, (255, 0, 0), -1)

        # 保存图片而不是显示
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.title(f'图片 {img_idx}: {Path(img_path).name}')
        plt.axis('off')

        # 添加标注框数量信息
        num_boxes = len(current_bboxes)
        plt.text(10, 30, f'标注框数量: {num_boxes}',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                 fontsize=12)

        plt.tight_layout()

        # 生成保存文件名
        img_name = Path(img_path).stem
        save_path = save_dir / f"{img_name}_visualization.png"

        # 保存图片
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()  # 关闭图形以释放内存

        # print(f"可视化结果已保存到: {save_path}")

# 可视化增强后的图像对应的OBB标注
def visualizeAugmentImg(batch, save_dir="./visualization_output"):
    """
    可视化增强后的图像和对应的 OBB 标注框（xywhr格式）

    Args:
        batch (dict): 包含以下键的字典:
            - 'bboxes': 标注框信息，格式为 [x, y, w, h, r] (xywhr格式)
            - 'batch_idx': 当前框属于第几张图的索引
            - 'img': 图像数据，Tensor 格式 (B, 3, H, W)
        save_dir (str): 保存可视化结果的目录路径
    """
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from pathlib import Path
    import torch

    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取数据
    bboxes = batch['bboxes']
    batch_idx = batch['batch_idx']
    imgs = batch['img']  # (B, 3, H, W)


    # 转为 numpy
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().numpy()

    # 调整图片格式为 (B, H, W, 3)
    imgs = np.transpose(imgs, (0, 2, 3, 1))  # (B, 640, 640, 3)
    imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)  # 反归一化（如果原来在 [0,1]）

    # 获取每张图片对应的框
    unique_batch_idx = np.unique(batch_idx)

    for img_idx in range(len(imgs)):
        # 当前图片和框
        img = imgs[int(img_idx)]
        h, w = img.shape[:2]

        mask = batch_idx == img_idx
        current_bboxes = bboxes[mask]

        vis_image = img.copy()

        for bbox in current_bboxes:
            x_center, y_center, width, height, rotation = bbox

            # 若坐标是归一化的，则转换为像素
            if x_center <= 1.0 and y_center <= 1.0 and width <= 1.0 and height <= 1.0:
                x_center *= w
                y_center *= h
                width *= w
                height *= h

            # 计算 OBB 四个点
            half_w, half_h = width / 2, height / 2
            corners = np.array([
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h]
            ], dtype=np.float32)

            cos_r, sin_r = np.cos(rotation), np.sin(rotation)
            rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
            rotated_corners = corners @ rot_matrix.T + np.array([x_center, y_center], dtype=np.float32)
            points = np.clip(rotated_corners.astype(np.int32), 0, [w - 1, h - 1])

            cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)
            cv2.circle(vis_image, (int(x_center), int(y_center)), 3, (255, 0, 0), -1)

        # 保存图片
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f"augment image {img_idx} - bboxes number: {len(current_bboxes)}")
        plt.axis('off')


        # 使用原图文件名作为保存名（如果有）
        if 'im_file' in batch:
            img_name = Path(batch['im_file'][int(img_idx)]).stem
            save_path = save_dir / f"{img_name}_aug.png"
        else:
            save_path = save_dir / f"aug_img_{img_idx}.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def _visualize_matched_boxes(pred_bboxes, gt_bboxes, batch, gt_cls, pred_scores):
    """
    可视化已经匹配的预测框和真实框（一一对应），分别保存在两个图片中，并保存组合图片
    支持OBB格式：xywhr (x, y, w, h, r)

    Args:
        pred_bboxes: 匹配的预测框 (N, 5) - xywhr格式
        gt_bboxes: 匹配的真实框 (N, 5) - xywhr格式
        batch: 批次数据字典
        gt_cls: 匹配的真实类别 (N,)
        pred_scores: 匹配的预测分数 (N, num_classes)
    """
    import cv2
    import numpy as np
    import torch
    from pathlib import Path

    # 获取图片路径和每张图片的GT数量
    im_files = batch['im_file'] if isinstance(batch['im_file'], list) else [batch['im_file']]
    gt_groups = batch['gt_groups']

    # 确保 gt_groups 长度与 im_files 一致
    if len(gt_groups) != len(im_files):
        print(f"警告: gt_groups长度({len(gt_groups)})与图片数量({len(im_files)})不一致")
        return

    # 计算每张图片的起始索引
    gt_start_indices = [0] + list(np.cumsum(gt_groups[:-1]))
    
    # 检查总框数是否一致
    total_boxes = sum(gt_groups)
    if total_boxes != len(pred_bboxes) or total_boxes != len(gt_bboxes):
        # print(f"警告: 总框数不一致 - gt_groups总和: {total_boxes}, pred_bboxes: {len(pred_bboxes)}, gt_bboxes: {len(gt_bboxes)}")
        return

    # 清空保存目录，避免旧图片干扰
    save_dir = Path('matched_obb_visualization')
    save_dir.mkdir(exist_ok=True)

    for img_idx, (im_file, num_gts) in enumerate(zip(im_files, gt_groups)):
        if img_idx >= len(gt_start_indices):
            print(f"警告: 图片索引 {img_idx} 超出起始索引列表范围 {len(gt_start_indices)}")
            break

        # 获取当前图片的匹配框索引范围
        start_idx = gt_start_indices[img_idx]
        end_idx = start_idx + num_gts

        # 严格检查索引边界
        if start_idx >= len(pred_bboxes) or start_idx >= len(gt_bboxes) or end_idx > len(pred_bboxes) or end_idx > len(gt_bboxes):
            print(f"警告: 图片 {img_idx} 的索引范围 [{start_idx}, {end_idx}) 超出边界 pred_bboxes:{len(pred_bboxes)}, gt_bboxes:{len(gt_bboxes)}")
            continue

        # 提取当前图片的匹配框
        img_pred_boxes = pred_bboxes[start_idx:end_idx].clone().detach() if isinstance(pred_bboxes, torch.Tensor) else pred_bboxes[start_idx:end_idx].copy()
        img_gt_boxes = gt_bboxes[start_idx:end_idx].clone().detach() if isinstance(gt_bboxes, torch.Tensor) else gt_bboxes[start_idx:end_idx].copy()
        img_gt_cls = gt_cls[start_idx:end_idx].clone().detach() if isinstance(gt_cls, torch.Tensor) else gt_cls[start_idx:end_idx]
        img_pred_scores = pred_scores[start_idx:end_idx].clone().detach() if isinstance(pred_scores, torch.Tensor) else pred_scores[start_idx:end_idx]

        if len(img_pred_boxes) == 0 or len(img_gt_boxes) == 0:
            print(f"警告: 图片 {img_idx} 没有匹配的框")
            continue

        # 读取图片
        img_path = Path(im_file)
        if not img_path.exists():
            print(f"警告: 图片路径不存在 {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告: 无法读取图片 {img_path}")
            continue

        h, w = img.shape[:2]

        # 创建三个图片副本：预测框、真实框、组合图片
        img_pred = img.copy()
        img_gt = img.copy()
        img_combined = img.copy()

        # 获取模型输入尺寸 (通常是640x640)
        model_input_size = 640  # 或者从batch中获取

        # 计算缩放比例
        scale_x = w / model_input_size
        scale_y = h / model_input_size

        # 绘制匹配的框对
        for i, (pred_box, gt_box, gt_class, pred_score) in enumerate(zip(
                img_pred_boxes, img_gt_boxes, img_gt_cls, img_pred_scores
        )):

            pred_x, pred_y, pred_w, pred_h, pred_r = pred_box[:5]
            gt_x, gt_y, gt_w, gt_h, gt_r = gt_box[:5]

            # 转换为numpy数组以便计算
            if hasattr(pred_x, 'item'):  # 如果是tensor
                pred_x, pred_y, pred_w, pred_h, pred_r = [x.item() for x in [pred_x, pred_y, pred_w, pred_h, pred_r]]
                gt_x, gt_y, gt_w, gt_h, gt_r = [x.item() for x in [gt_x, gt_y, gt_w, gt_h, gt_r]]

            # 将坐标从模型输入尺寸映射到原始图像尺寸
            if pred_x <= 1.0 and pred_y <= 1.0:  # 如果是归一化坐标，先转换为模型输入尺寸的像素坐标
                pred_x *= model_input_size
                pred_y *= model_input_size
                pred_w *= model_input_size
                pred_h *= model_input_size

            if gt_x <= 1.0 and gt_y <= 1.0:
                gt_x *= model_input_size
                gt_y *= model_input_size
                gt_w *= model_input_size
                gt_h *= model_input_size

            # 映射到原始图像尺寸
            pred_x *= scale_x
            pred_y *= scale_y
            pred_w *= scale_x
            pred_h *= scale_y

            gt_x *= scale_x
            gt_y *= scale_y
            gt_w *= scale_x
            gt_h *= scale_y

            # 确保坐标在图像范围内
            pred_x = max(0, min(w - 1, pred_x))
            pred_y = max(0, min(h - 1, pred_y))
            gt_x = max(0, min(w - 1, gt_x))
            gt_y = max(0, min(h - 1, gt_y))
            from ultralytics.utils.ops import xywhr2xyxyxyxy

            # 正确使用 xywhr2xyxyxyxy 函数 - 传入完整的5元素数组
            pred_obb = np.array([pred_x, pred_y, pred_w, pred_h, pred_r])
            gt_obb = np.array([gt_x, gt_y, gt_w, gt_h, gt_r])

            # 计算旋转矩形的四个角点
            pred_corners = xywhr2xyxyxyxy(pred_obb)  # 返回 (4, 2) 的角点坐标
            gt_corners = xywhr2xyxyxyxy(gt_obb)  # 返回 (4, 2) 的角点坐标

            # 转换为整数坐标并确保在图像范围内
            pred_corners = pred_corners.astype(np.int32)
            gt_corners = gt_corners.astype(np.int32)
            pred_corners = np.clip(pred_corners, [0, 0], [w - 1, h - 1])
            gt_corners = np.clip(gt_corners, [0, 0], [w - 1, h - 1])

            # 在预测框图片上绘制预测框 (红色)
            cv2.polylines(img_pred, [pred_corners], True, (0, 0, 255), 2)

            # 在真实框图片上绘制真实框 (绿色)
            cv2.polylines(img_gt, [gt_corners], True, (0, 255, 0), 2)

            # 在组合图片上绘制预测框和真实框
            cv2.polylines(img_combined, [pred_corners], True, (0, 0, 255), 2)  # 红色预测框
            cv2.polylines(img_combined, [gt_corners], True, (0, 255, 0), 2)  # 绿色真实框

            # 添加标签
            gt_class_id = int(gt_class.item()) if hasattr(gt_class, 'item') else int(gt_class)
            pred_class_id = int(torch.argmax(pred_score))
            pred_conf = float(torch.max(pred_score))

            # 计算标签位置（使用第一个角点）
            pred_label_pos = (max(10, pred_corners[0][0]), max(20, pred_corners[0][1] - 10))
            gt_label_pos = (max(10, gt_corners[0][0]), max(20, gt_corners[0][1] - 10))

            # 预测框标签
            cv2.putText(img_pred, f'Pred:{pred_class_id}({pred_conf:.2f})', pred_label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 真实框标签
            cv2.putText(img_gt, f'GT:{gt_class_id}', gt_label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 组合图片标签
            cv2.putText(img_combined, f'Pred:{pred_class_id}({pred_conf:.2f})', pred_label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img_combined, f'GT:{gt_class_id}', (gt_label_pos[0], gt_label_pos[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 添加匹配序号和中心点
            pred_center = (int(pred_x), int(pred_y))
            gt_center = (int(gt_x), int(gt_y))

            cv2.circle(img_pred, pred_center, 3, (0, 0, 255), -1)
            cv2.putText(img_pred, f'{i}', (pred_center[0] - 5, pred_center[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.circle(img_gt, gt_center, 3, (0, 255, 0), -1)
            cv2.putText(img_gt, f'{i}', (gt_center[0] - 5, gt_center[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # 在组合图片上绘制连接线和序号
            cv2.line(img_combined, pred_center, gt_center, (255, 0, 0), 2)  # 蓝色连接线
            cv2.circle(img_combined, pred_center, 3, (0, 0, 255), -1)
            cv2.circle(img_combined, gt_center, 3, (0, 255, 0), -1)

            # 在连接线中点添加匹配序号
            mid_point = ((pred_center[0] + gt_center[0]) // 2, (pred_center[1] + gt_center[1]) // 2)
            cv2.circle(img_combined, mid_point, 3, (255, 0, 0), -1)
            cv2.putText(img_combined, f'{i}', (mid_point[0] - 5, mid_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # 可选：在中心点绘制旋转角度指示线
            angle_length = min(pred_w, pred_h) / 4
            angle_end_x = int(pred_x + angle_length * np.cos(pred_r))
            angle_end_y = int(pred_y + angle_length * np.sin(pred_r))
            cv2.arrowedLine(img_pred, pred_center, (angle_end_x, angle_end_y), (255, 255, 0), 2)
            cv2.arrowedLine(img_combined, pred_center, (angle_end_x, angle_end_y), (255, 255, 0), 2)

            # GT旋转角度指示线
            gt_angle_end_x = int(gt_x + angle_length * np.cos(gt_r))
            gt_angle_end_y = int(gt_y + angle_length * np.sin(gt_r))
            cv2.arrowedLine(img_gt, gt_center, (gt_angle_end_x, gt_angle_end_y), (255, 255, 0), 2)
            cv2.arrowedLine(img_combined, gt_center, (gt_angle_end_x, gt_angle_end_y), (255, 255, 0), 2)

        # 保存可视化结果
        save_dir = Path('matched_obb_visualization')
        save_dir.mkdir(exist_ok=True)

        # 分别保存预测框、真实框和组合图片
        pred_save_path = save_dir / f'{img_path.stem}_pred.jpg'
        gt_save_path = save_dir / f'{img_path.stem}_gt.jpg'
        combined_save_path = save_dir / f'{img_path.stem}_combined.jpg'

        cv2.imwrite(str(pred_save_path), img_pred)
        cv2.imwrite(str(gt_save_path), img_gt)
        cv2.imwrite(str(combined_save_path), img_combined)

        # print(f"已保存图片 {img_idx} ({img_path.name}) 的可视化结果")
        # 打印匹配详情，包含旋转角度信息
        for i, (pred_box, gt_box, gt_class, pred_score) in enumerate(zip(
                img_pred_boxes, img_gt_boxes, img_gt_cls, img_pred_scores
        )):
            gt_class_id = int(gt_class) if hasattr(gt_class, 'item') else int(gt_class)
            pred_class_id = int(torch.argmax(pred_score))
            pred_conf = float(torch.max(pred_score))

            # 获取旋转角度（转换为度数）
            pred_angle = float(pred_box[4]) * 180 / np.pi if hasattr(pred_box[4], 'item') else pred_box[4] * 180 / np.pi
            gt_angle = float(gt_box[4]) * 180 / np.pi if hasattr(gt_box[4], 'item') else gt_box[4] * 180 / np.pi

            # print(
            #     f"  匹配 {i}: GT类别{gt_class_id}(角度{gt_angle:.1f}°) <-> 预测类别{pred_class_id}(置信度{pred_conf:.3f}, 角度{pred_angle:.1f}°)")


# 检查坐标系统的调试代码
def check_coordinate_system(batch):
    """检查坐标系统是否为归一化坐标"""

    bboxes = batch["bboxes"]  # shape: [N, 5] for OBB

    print("=== 坐标系统分析 ===")
    print(f"bbox形状: {bboxes.shape}")
    print(f"bbox数据类型: {bboxes.dtype}")

    # 检查各个维度的数值范围
    for i, dim_name in enumerate(['x', 'y', 'w', 'h', 'r']):
        values = bboxes[:, i]
        print(f"{dim_name}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")

    # 判断是否为归一化坐标
    x_normalized = (bboxes[:, 0].min() >= 0) and (bboxes[:, 0].max() <= 1)
    y_normalized = (bboxes[:, 1].min() >= 0) and (bboxes[:, 1].max() <= 1)
    w_normalized = (bboxes[:, 2].min() >= 0) and (bboxes[:, 2].max() <= 1)
    h_normalized = (bboxes[:, 3].min() >= 0) and (bboxes[:, 3].max() <= 1)

    print("\n=== 归一化判断 ===")
    print(f"x坐标归一化: {x_normalized}")
    print(f"y坐标归一化: {y_normalized}")
    print(f"宽度归一化: {w_normalized}")
    print(f"高度归一化: {h_normalized}")

    # 角度范围检查
    angles = bboxes[:, 4]
    angle_in_radians = (angles.min() >= -3.14159) and (angles.max() <= 3.14159)
    angle_in_degrees = (angles.min() >= -180) and (angles.max() <= 180)

    print(f"角度范围: min={angles.min():.4f}, max={angles.max():.4f}")
    print(f"可能是弧度制: {angle_in_radians}")
    print(f"可能是角度制: {angle_in_degrees}")

    # 总体判断
    likely_normalized = x_normalized and y_normalized and w_normalized and h_normalized
    print(f"\n=== 结论 ===")
    print(f"坐标很可能是归一化的: {likely_normalized}")

    if not likely_normalized:
        print("建议检查:")
        print("1. 数据预处理流程")
        print("2. 图像尺寸信息")
        print("3. 数据集格式说明")

    return likely_normalized

# 使用示例
# is_normalized = check_coordinate_system(batch)


import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics.utils.ops import xywhr2xyxyxyxy
isSave = False
def visualize_val(preds, batch, save_dir='runs/val/obb_vis', model_input_size=640,isSave=True):
    if(not isSave):
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    gt_boxes_all = batch['bboxes']   # (M, 5)
    gt_cls_all = batch['cls']        # (M,)
    batch_idx_all = batch['batch_idx']  # (M,)
    im_files = batch['im_file']      # len = batch_size

    if isinstance(im_files, str):  # 单张图像兼容性
        im_files = [im_files]

    device = preds[0].device if isinstance(preds[0], torch.Tensor) else 'cpu'

    for img_id, im_file in enumerate(im_files):
        pred = preds[img_id]
        pred = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.array(pred)

        # 当前图像对应的 GT 框
        mask = (batch_idx_all == img_id)
        gt_boxes = gt_boxes_all[mask]
        gt_cls = gt_cls_all[mask]
        gt_boxes = gt_boxes.cpu().numpy() if torch.is_tensor(gt_boxes) else gt_boxes
        gt_cls = gt_cls.cpu().numpy() if torch.is_tensor(gt_cls) else gt_cls

        # 读取图像
        img = cv2.imread(str(im_file))
        if img is None:
            print(f"无法读取图像: {im_file}")
            continue

        h, w = img.shape[:2]
        scale_x = w / model_input_size
        scale_y = h / model_input_size

        img_pred = img.copy()
        img_gt = img.copy()
        img_comb = img.copy()

        # --- 绘制预测框 ---
        for box in pred:
            x, y, w_, h_, angle, score, cls = box
            if x <= 1.0:  # 归一化
                x *= model_input_size
                y *= model_input_size
                w_ *= model_input_size
                h_ *= model_input_size
            x *= scale_x; y *= scale_y
            w_ *= scale_x; h_ *= scale_y

            corners = xywhr2xyxyxyxy(np.array([x, y, w_, h_, angle])).astype(np.int32)
            corners = np.clip(corners, [0, 0], [img.shape[1]-1, img.shape[0]-1])
            cv2.polylines(img_pred, [corners], True, (0, 0, 255), 2)
            cv2.polylines(img_comb, [corners], True, (0, 0, 255), 2)
            pos = tuple(corners[0])
            cv2.putText(img_pred, f'{int(cls)}({score:.2f})', pos, 0, 0.5, (0, 0, 255), 1)
            cv2.putText(img_comb, f'{int(cls)}({score:.2f})', pos, 0, 0.5, (0, 0, 255), 1)

        # --- 绘制 GT 框 ---
        for j, (x, y, w_, h_, angle) in enumerate(gt_boxes):
            if x <= 1.0:
                x *= model_input_size
                y *= model_input_size
                w_ *= model_input_size
                h_ *= model_input_size
            x *= scale_x; y *= scale_y
            w_ *= scale_x; h_ *= scale_y

            corners = xywhr2xyxyxyxy(np.array([x, y, w_, h_, angle])).astype(np.int32)
            corners = np.clip(corners, [0, 0], [img.shape[1]-1, img.shape[0]-1])
            cv2.polylines(img_gt, [corners], True, (0, 255, 0), 2)
            cv2.polylines(img_comb, [corners], True, (0, 255, 0), 2)
            pos = tuple(corners[0])
            cv2.putText(img_gt, f'{int(gt_cls[j])}', pos, 0, 0.5, (0, 255, 0), 1)
            cv2.putText(img_comb, f'{int(gt_cls[j])}', pos, 0, 0.5, (0, 255, 0), 1)

        # 保存图像
        stem = Path(im_file).stem
        cv2.imwrite(str(save_dir / f'{stem}_pred.jpg'), img_pred)
        cv2.imwrite(str(save_dir / f'{stem}_gt.jpg'), img_gt)
        cv2.imwrite(str(save_dir / f'{stem}_combined.jpg'), img_comb)
