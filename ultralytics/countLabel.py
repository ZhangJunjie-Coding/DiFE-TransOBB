#
# @author ZhangJunjie
#
import os

# 图像文件路径
file_paths = ['/home/zjj/dataset/SDFSD/images/train/GAX001__934__.png', '/home/zjj/dataset/SDFSD/images/train/HEB061__543__.png', '/home/zjj/dataset/SDFSD/images/train/HEB028__797__.png', '/home/zjj/dataset/SDFSD/images/train/JLA005__555__.png', '/home/zjj/dataset/SDFSD/images/train/HEB060__630__.png', '/home/zjj/dataset/SDFSD/images/train/HEB055__621__.png', '/home/zjj/dataset/SDFSD/images/train/HEB030__698__.png', '/home/zjj/dataset/SDFSD/images/train/WUS014__596__.png']
# 标注文件路径
label_dir = '/home/zjj/dataset/SDFSD/labels/train/'

total_lines = 0

for path in file_paths:
    # 获取图片文件名
    image_name = os.path.basename(path)

    # 生成对应的标注文件路径 (假设是 .txt 文件)
    label_file_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')

    if os.path.exists(label_file_path):
        try:
            with open(label_file_path, 'r') as f:
                lines = sum(1 for _ in f)
                print(f"{image_name}: {lines} lines")
                total_lines += lines
        except Exception as e:
            print(f"⚠️ Error reading {label_file_path}: {e}")
    else:
        print(f"❌ Label file not found for {image_name}: {label_file_path}")

print(f"\nTotal lines across all label files: {total_lines}")
