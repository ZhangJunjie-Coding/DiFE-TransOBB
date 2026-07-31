#
# @author ZhangJunjie
#
#
# @author ZhangJunjie
# 改进版：支持单张图片可视化和批量处理
#
import os
import cv2
import numpy as np
import argparse
from pathlib import Path


def get_class_info(class_id):
    """获取类别名称和颜色"""
    class_names = {
        0: 'plane', 1: 'ship', 2: 'storage-tank', 3: 'baseball-diamond',
        4: 'tennis-court', 5: 'basketball-court', 6: 'ground-track-field',
        7: 'harbor', 8: 'bridge', 9: 'large-vehicle', 10: 'small-vehicle',
        11: 'helicopter', 12: 'roundabout', 13: 'soccer-ball-field', 14: 'swimming-pool'
    }

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 255), (128, 128, 0),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 128), (255, 192, 203)
    ]

    class_name = class_names.get(class_id, f'class_{class_id}')
    color = colors[class_id % len(colors)]
    return class_name, color


def draw_obb(image, obb_list, class_ids=None, show_labels=False, thickness=2):
    """绘制OBB框"""
    for i, obb in enumerate(obb_list):
        class_id = class_ids[i] if class_ids and i < len(class_ids) else 0
        class_name, color = get_class_info(class_id)

        # 绘制多边形
        pts = np.array(obb, dtype=np.float32).reshape((4, 2))
        cv2.polylines(image, [pts.astype(np.int32)], isClosed=True, color=color, thickness=thickness)

        if show_labels:
            # 计算中心点并添加标签
            center = np.mean(pts, axis=0).astype(int)
            label_text = f"{class_name}({class_id})"

            # 绘制标签背景
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image,
                          (center[0] - text_width // 2 - 5, center[1] - text_height // 2 - 5),
                          (center[0] + text_width // 2 + 5, center[1] + text_height // 2 + 5),
                          color, -1)

            # 绘制文字
            cv2.putText(image, label_text,
                        (center[0] - text_width // 2, center[1] + text_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def load_labels(label_path, img_width, img_height):
    """加载标签文件"""
    obbs = []
    class_ids = []

    if not os.path.exists(label_path):
        print(f"警告: 标签文件不存在: {label_path}")
        return obbs, class_ids

    with open(label_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 9:
                print(f"警告: 第{line_num}行格式错误: {line.strip()}")
                continue

            try:
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                coords = np.array(coords).reshape(4, 2)
                coords[:, 0] *= img_width
                coords[:, 1] *= img_height
                obbs.append(coords.flatten().tolist())
                class_ids.append(class_id)
            except ValueError as e:
                print(f"警告: 第{line_num}行解析错误: {e}")
                continue

    return obbs, class_ids


def find_label_path(image_path, label_dir=None):
    """根据图片路径查找对应的标签文件"""
    image_path = Path(image_path)

    # 如果没有指定label目录，则在同级目录下查找
    if label_dir is None:
        # 尝试几种常见的标签目录结构
        possible_dirs = [
            image_path.parent / 'labels',
            image_path.parent / '../labels' / image_path.parent.name,
            image_path.parent.parent / 'labels' / image_path.parent.name,
            image_path.parent  # 同一目录下
        ]
    else:
        possible_dirs = [Path(label_dir)]

    # 查找标签文件
    label_name = image_path.stem + '.txt'
    for dir_path in possible_dirs:
        label_path = dir_path / label_name
        if label_path.exists():
            return str(label_path)

    return None


def visualize_single_image(image_path, label_path=None, output_path=None,
                           show_labels=False, show_image=False):
    """可视化单张图片"""
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在: {image_path}")
        return False

    # 自动查找标签文件
    if label_path is None:
        label_path = find_label_path(image_path)
        if label_path is None:
            print(f"错误: 未找到对应的标签文件: {Path(image_path).stem}.txt")
            return False

    print(f"处理图片: {image_path}")
    print(f"使用标签: {label_path}")

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图片: {image_path}")
        return False

    h, w = image.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # 加载标签
    obbs, class_ids = load_labels(label_path, w, h)
    print(f"找到 {len(obbs)} 个标注对象")

    if not obbs:
        print("警告: 没有找到有效的标注")
        return False

    # 绘制OBB
    image_drawn = draw_obb(image.copy(), obbs, class_ids, show_labels)

    # # 添加信息文字
    # info_text = f"Objects: {len(obbs)} | File: {Path(image_path).name}"
    # cv2.putText(image_drawn, info_text, (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
    # cv2.putText(image_drawn, info_text, (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    # 显示图片
    if show_image:
        # 调整显示尺寸
        if max(h, w) > 1200:
            scale = 1200 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            display_img = cv2.resize(image_drawn, (new_w, new_h))
        else:
            display_img = image_drawn

        cv2.imshow('OBB Visualization', display_img)
        print("按任意键继续，按ESC或q退出...")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 27 or key == ord('q'):  # ESC or 'q'
            return False

    # 保存图片
    if output_path is None:
        output_path = f"vis_{Path(image_path).name}"

    cv2.imwrite(output_path, image_drawn)
    print(f"可视化结果保存到: {output_path}")

    # 打印标注详情
    for i, class_id in enumerate(class_ids):
        class_name, _ = get_class_info(class_id)
        print(f"  对象 {i + 1}: {class_name}({class_id})")

    return True


def visualize_directory(image_dir, label_dir=None, output_dir="output_vis",
                        show_labels=True, max_images=None):
    """批量可视化目录下的图片"""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"错误: 图片目录不存在: {image_dir}")
        return

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 获取所有图片文件
    image_extensions = ['.jpg', '.png', '.jpeg', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
        image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))

    if max_images:
        image_files = image_files[:max_images]

    print(f"找到 {len(image_files)} 张图片")

    success_count = 0
    for i, img_file in enumerate(image_files):
        print(f"\n处理第 {i + 1}/{len(image_files)} 张图片")

        output_path = output_dir / f"vis_{img_file.name}"

        if visualize_single_image(
                str(img_file),
                label_dir,
                str(output_path),
                show_labels,
                False
        ):
            success_count += 1

    print(f"\n批量处理完成! 成功处理 {success_count}/{len(image_files)} 张图片")
    print(f"结果保存在: {output_dir}")


def main():
    name = "BLT011__795__"
    parser = argparse.ArgumentParser(description='DOTA格式OBB可视化工具')
    parser.add_argument('--image', type=str, help='',default=f"/home/zjj/dataset/SDFSD/images/val/{name}.png")
    parser.add_argument('--image-dir', type=str, help='图片目录路径')
    parser.add_argument('--label', type=str, help='标签文件路径（可选）',default=f"/home/zjj/dataset/SDFSD/labels/val/{name}.txt")
    parser.add_argument('--label-dir', type=str, help='标签目录路径（可选）')
    parser.add_argument('--output', type=str, help='输出文件/目录路径')
    parser.add_argument('--show', action='store_true', help='显示图片窗口')
    parser.add_argument('--no-labels', action='store_true', help='不显示类别标签')
    parser.add_argument('--max-images', type=int, help='最大处理图片数量')

    args = parser.parse_args()

    if args.image:
        # 单张图片模式
        print("=== 单张图片可视化模式 ===")
        visualize_single_image(
            args.image,
            args.label,
            args.output,
            args.no_labels,
            args.show
        )
    elif args.image_dir:
        # 批量处理模式
        print("=== 批量可视化模式 ===")
        output_dir = args.output or "output_vis"
        visualize_directory(
            args.image_dir,
            args.label_dir,
            output_dir,
            not args.no_labels,
            args.max_images
        )
    else:
        # 默认使用预设路径
        print("=== 使用默认路径 ===")
        image_dir = "/home/zjj/dataset/SDFSD/images/train"
        label_dir = "/home/zjj/dataset/SDFSD/labels/train"
        output_dir = "output_vis"

        if os.path.exists(image_dir):
            visualize_directory(image_dir, label_dir, output_dir, not args.no_labels, args.max_images)
        else:
            print("请指定 --image 或 --image-dir 参数")
            print("\n使用示例:")
            print("python script.py --image /path/to/image.jpg")
            print("python script.py --image-dir /path/to/images")


import os
import numpy as np


def points_to_xywhr(xy):
    pts = np.array(xy).reshape(4,2)
    cx, cy = pts.mean(axis=0)
    edge1 = pts[1] - pts[0]
    edge2 = pts[2] - pts[1]
    w = np.linalg.norm(edge1)
    h = np.linalg.norm(edge2)
    angle = np.arctan2(edge1[1], edge1[0])
    return cx, cy, w, h, angle

def remove_bad_lines(val_dir):
    files = [f for f in os.listdir(val_dir) if f.endswith('.txt')]
    removed_lines_count = 0

    for file in files:
        filepath = os.path.join(val_dir, file)
        with open(filepath, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            items = line.strip().split()
            if len(items) != 9:
                # 格式异常，保留或跳过？这里先保留
                new_lines.append(line)
                continue
            coords = list(map(float, items[1:]))
            _, _, w, h, _ = points_to_xywhr(coords)
            if w == 0 or h == 0:
                removed_lines_count += 1
                continue  # 不保留这行
            new_lines.append(line)

        # 只在有删行时才覆盖文件
        if len(new_lines) < len(lines):
            with open(filepath, 'w') as f:
                f.writelines(new_lines)
            print(f"File {file} cleaned: removed {len(lines) - len(new_lines)} bad lines")

    print(f"Total removed lines across files: {removed_lines_count}")

def replace_class_to_zero(label_dir):
    files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    for file in files:
        path = os.path.join(label_dir, file)
        with open(path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                # 格式异常，原样保留
                new_lines.append(line)
                continue
            # 替换类别为0
            parts[0] = '0'
            new_line = ' '.join(parts) + '\n'
            new_lines.append(new_line)

        with open(path, 'w') as f:
            f.writelines(new_lines)
        print(f"Processed file: {file}")



label_directory = './val'  # 修改为你标注文件夹路径
def replace_class_to_zero(label_dir):
    files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    for file in files:
        path = os.path.join(label_dir, file)
        with open(path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                # 格式异常，原样保留
                new_lines.append(line)
                continue
            # 替换类别为0
            parts[0] = '0'
            new_line = ' '.join(parts) + '\n'
            new_lines.append(new_line)

        with open(path, 'w') as f:
            f.writelines(new_lines)
        print(f"Processed file: {file}")



label_directory = '/home/zjj/dataset/SDFSD/labels/train'  # 修改为你标注文件夹路径

val_dir = '/home/zjj/dataset/SDFSD/labels/val'  # 你的val目录路径，按实际改
if __name__ == "__main__":
    # remove_bad_lines(val_dir)
    replace_class_to_zero(label_directory)
    # main()
