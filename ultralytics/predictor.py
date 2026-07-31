from ultralytics import RTDETR

model = RTDETR(
    '/home/zjj/baseLineCode/ultralytics-main/runs/obb/a_best.pt'
)

model.predict(
    # source='/home/zjj/dataset/DenseSARShipDataSet/images/val',
    # source='./1.png',
    source='./tiles',
    imgsz=640,
    device=0,
    conf=0.5,
    iou=0.5,
    save=True,          # 保存可视化结果
    save_txt=True,     # 是否保存txt
    show=False,         # 不弹窗
    project='runs/predict',
    name='DenseSARShip_val_vis'
)
