import numpy as np

# 原始数据（归一化坐标，相对于800x800）
data = [1, 0.6725, 0.84175, 0.72125, 0.84775, 0.7195, 0.862, 0.670625, 0.855875]
cls = data[0]
coords = np.array(data[1:]).reshape(-1, 2)  # 4x2

# 将归一化坐标还原到 800x800 像素
coords_pixel_800 = coords * 800

# 然后映射到 640x640（等比例缩放）
coords_pixel_640 = coords_pixel_800 * (640 / 800)  # 或者乘以 0.8

# 最后（可选）归一化回到 640x640（如果你想保持归一化形式）
coords_normalized_640 = coords_pixel_640 / 640

# 输出格式
output = [cls] + coords_normalized_640.flatten().tolist()
print(output)
