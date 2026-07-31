#
# @author ZhangJunjie
#
from torchview import draw_graph
from ultralytics import RTDETR

model = RTDETR(model='/home/zjj/baseLineCode/ultralytics-main/ultralytics/runs/obb/train/weights/best.pt').model
model.eval().cuda()

graph = draw_graph(model, input_size=(1, 3, 640, 640), expand_nested=True, device='cuda')
graph.visual_graph.render("model_structure", format="png")

#
# # 导出为ONNX
# torch.onnx.export(
#     model,
#     dummy_input,
#     "rtdetr_obb.onnx",
#     export_params=True,
#     opset_version=16,
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
# )
# print("✅ 导出成功: rtdetr_obb.onnx")


