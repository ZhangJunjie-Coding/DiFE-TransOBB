"""
计算 RT-DETR 模型的 GFLOPs
"""
import torch
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import get_flops


def calc_gflops(model_yaml, imgsz=640, device='cpu'):
    # 构建模型
    model = RTDETRDetectionModel(cfg=model_yaml, ch=3, nc=1, verbose=False)
    model.eval()
    model = model.to(device)

    # 直接用 thop 计算
    try:
        import thop
        im = torch.zeros(1, 3, imgsz, imgsz, device=device)
        flops, params = thop.profile(model, inputs=(im,), verbose=False)
        flops = flops / 1e9 * 2  # thop 返回的是 MACs，乘2换算为 FLOPs
        print(f"  thop 直接计算: {flops:.2f} GFLOPs, {params / 1e6:.2f} M params")
    except ImportError:
        print("  thop 未安装，跳过直接计算")

    # 用 ultralytics get_flops
    flops2 = get_flops(model, imgsz=imgsz)
    if flops2:
        print(f"  get_flops:     {flops2:.2f} GFLOPs")
    else:
        print("  get_flops:     计算失败")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:  {total_params / 1e6:.2f} M")
    print(f"  Trainable:     {trainable_params / 1e6:.2f} M")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Input: {args.imgsz}x{args.imgsz}, device={args.device}")
    calc_gflops(args.model, args.imgsz, args.device)
