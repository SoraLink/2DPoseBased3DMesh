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


def process_image(processor, image_path: Path, out_dir: Path, device: str, keep_num: int = 2):
    image = Image.open(image_path).convert("RGB")

    with torch.inference_mode(), torch.autocast(device_type="cuda" if "cuda" in device else "cpu"):
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt="person")

    masks = output["masks"]
    boxes = output["boxes"]

    # 1. 转换为 numpy
    masks_np = masks.cpu().numpy() if hasattr(masks, "cpu") else np.array(masks)
    boxes_np = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.array(boxes)

    # 统一 masks 维度到 (N, H, W)
    if masks_np.ndim == 4:
        masks_np = masks_np.squeeze(1)
    elif masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis, ...]

    if boxes_np.ndim == 1:
        boxes_np = boxes_np[np.newaxis, ...]

    num_masks = masks_np.shape[0]

    if num_masks == 0:
        print(f"警告: {image_path.name} 未检测到人物。")
        return

    # 2. 计算每个 mask 的分数，然后选 top-k
    img_w, img_h = image.size
    img_cx, img_cy = img_w / 2, img_h / 2

    scores = []

    for i in range(num_masks):
        x1, y1, x2, y2 = boxes_np[i]

        area = max(0, x2 - x1) * max(0, y2 - y1)

        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2

        dist = np.sqrt(
            ((box_cx - img_cx) / img_w) ** 2 +
            ((box_cy - img_cy) / img_h) ** 2
        )

        # 面积越大越好，越靠近中心越好
        score = area * (1.0 - dist)

        scores.append(score)

    scores = np.asarray(scores)

    # 如果检测到的人少于 keep_num，就保留全部
    keep_num_actual = min(keep_num, num_masks)

    # 得分从高到低排序，取前 keep_num_actual 个
    selected_indices = np.argsort(scores)[::-1][:keep_num_actual]

    print(
        f"{image_path.name}: detected {num_masks} persons, "
        f"keep {keep_num_actual}: {selected_indices.tolist()}"
    )

    # 3. 合并两个/多个 mask
    selected_masks = masks_np[selected_indices]

    # 如果 mask 是 logits 或概率，这里统一转 bool
    selected_masks = selected_masks > 0

    final_mask = np.any(selected_masks, axis=0)

    # 4. 抠图处理
    image_rgba = image.convert("RGBA")
    image_array = np.array(image_rgba)

    # mask 外变透明
    image_array[~final_mask, :3] = [255, 255, 255]
    image_array[~final_mask, 3] = 0

    result_image = Image.fromarray(image_array)

    # 5. 保存结果
    save_path = out_dir / (image_path.stem + ".png")
    result_image.save(save_path, "PNG")

    print(f"处理完成: {image_path.name} -> {save_path.name}")

if __name__ == "__main__":
    image_dir = Path("./eval")
    output_dir = Path("./eval")
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
            raise e