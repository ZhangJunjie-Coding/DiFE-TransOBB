#
# @author ZhangJunjie
#

from ultralytics.models import RTDETR
from ultralytics.models import YOLO

# if __name__ == '__main__':
#     # model = RTDETR(model='/home/zjj/baseLineCode/ultralytics-main/ultralytics/runs/obb/train92/weights/best.pt')
#     model = RTDETR(model='/home/zjj/baseLineCode/ultralytics-main/ultralytics/runs/obb/train28/weights/best.pt')
#     # model.val(data='DOTAv1.yaml', batch=9, device='0', imgsz=640, workers=8, split='test')
#     model = RTDETR("/home/zjj/baseLineCode/ultralytics-main/runs/obb/best/best.pt").to("cuda:0")
#     prefix = ""
#     imgs = ['/home/zjj/dataset/SDFSD/images/val/ZSB016__624__.png',
#             '/home/zjj/dataset/SDFSD/images/val/ZSB020__505__.png',
#             '/home/zjj/dataset/SDFSD/images/val/ZSB013__671__.png',
#             '/home/zjj/dataset/SDFSD/images/val/ZSB015__714__.png',
#             '/home/zjj/dataset/SDFSD/images/val/ZSB015__715__.png',
#             '/home/zjj/dataset/SDFSD/images/val/ZSB016__624__.png'
#             ]
#
#     folder_path = '/home/zjj/dataset/SDFSD/images/val/'  # 替换为你的目标目录
#     all_files = []
#
#     for filename in os.listdir(folder_path):
#         full_path = os.path.join(folder_path, filename)
#         if os.path.isfile(full_path):  # 只保留文件，排除文件夹
#             all_files.append(os.path.abspath(full_path))
#
#     print(all_files)
#     imgs = ['./test.jpg']
#     imgs = ['/home/zjj/dataset/SDFSD/images/train/ZSB020__545__.png'
#          ]
#     imgs = ['/home/zjj/dataset/SDFSD/labels/train/ZSB020__545__.txt',
#             '/home/zjj/dataset/SDFSD/labels/train/ZSB020__545__.txt']
#     for img in all_files:
#         model(img, save=True)
    # print(results.boxes)


# model = RTDETR('/home/zjj/baseLineCode/ultralytics-main/ultralytics/runs/obb/train16/weights/last.pt')
model = RTDETR('/home/zjj/baseLineCode/ultralytics-main/runs/obb/a_best.pt')
# print(model)
#
# results = model.val(data="DOTAv1.yaml", plots=True,batch=4,show_labels=False,device=0)
results = model.val(data="DOTAv1.yaml", plots=True,batch=4,show_labels=False,device=0,iou=0.5,save_json=True)
print("map75: ",results.box.map75)

