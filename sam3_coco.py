from pathlib import Path
import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def init_model(device):
    """初始化模型并设置为评估模式"""
    print("正在加载模型...")
    model = build_sam3_image_model()
    model.to(device=device)
    model.eval()
    processor = Sam3Processor(model)
    return processor


def process_image(processor, image_path: Path, out_dir: Path, device: str):
    image = Image.open(image_path)

    with torch.inference_mode(), torch.autocast(device_type="cuda" if "cuda" in device else "cpu"):
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt="person")

    masks = output["masks"]
    boxes = output["boxes"]  # 获取边界框用于计算位置和面积

    # 1. 转换为 numpy
    masks_np = masks.cpu().numpy() if hasattr(masks, 'cpu') else np.array(masks)
    boxes_np = boxes.cpu().numpy() if hasattr(boxes, 'cpu') else np.array(boxes)

    # 统一 masks 维度到 (N, H, W)
    if masks_np.ndim == 4:
        masks_np = masks_np.squeeze(1)
    elif masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis, ...]
        boxes_np = boxes_np[np.newaxis, ...]

    num_masks = masks_np.shape[0]

    # 2. 筛选画面中心最大的 Mask
    if num_masks == 0:
        print(f"警告: {image_path.name} 未检测到人物。")
        return  # 直接跳过保存
    elif num_masks == 1:
        # 只有一个人，直接使用
        final_mask = masks_np[0]
    else:
        # --- 核心筛选逻辑 ---
        img_w, img_h = image.size
        img_cx, img_cy = img_w / 2, img_h / 2

        best_idx = 0
        best_score = -1

        for i in range(num_masks):
            # 获取 Bounding Box 坐标
            x1, y1, x2, y2 = boxes_np[i]

            # 1. 计算 Bounding Box 面积
            area = (x2 - x1) * (y2 - y1)

            # 2. 计算 Bounding Box 中心点到画面中心的归一化距离 (0 到 1 之间)
            box_cx, box_cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = np.sqrt(((box_cx - img_cx) / img_w) ** 2 + ((box_cy - img_cy) / img_h) ** 2)

            # 3. 综合评分：面积越大越好，距离中心越近越好
            # 距离越近，(1.0 - dist) 的值越接近 1
            score = area * (1.0 - dist)

            if score > best_score:
                best_score = score
                best_idx = i

        # 取出得分最高的那个 Mask
        final_mask = masks_np[best_idx]

    final_mask = final_mask.astype(bool)

    # 3. 抠图处理
    image_rgba = image.convert("RGBA")
    image_array = np.array(image_rgba)

    # 将 mask 之外的区域 alpha 设为 0（透明）
    image_array[~final_mask, 3] = 0
    result_image = Image.fromarray(image_array)

    # 4. 保存结果
    save_path = out_dir / (image_path.stem + ".png")
    result_image.save(save_path, "PNG")
    print(f"处理完成: {image_path.name} -> {save_path.name}")


if __name__ == "__main__":
    image_dir = Path("./data/eval")
    output_dir = Path("./data/eval_seg")
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- [修改] 模型初始化放在循环外 ---
    processor = init_model(device)

    # 支持常见的图片格式
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(ext))

    for image_path in image_files:
        try:
            process_image(processor, image_path, output_dir, device)
        except Exception as e:
            print(f"处理 {image_path.name} 时出错: {e}")