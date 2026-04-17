import json
from pathlib import Path

import cv2
import numpy as np
import torch
import os  # 新增 os 库处理路径
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from pose_extractor import read_kpts_annotation


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)


def segment_subject(image_path, keypoints, pad_ratio=0.3):
    """
    SAM 2 图像分割与扩图函数 (防马赛克、高精度版)
    """
    sam2_checkpoint = "./models/sam2/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # 1. 提取有效关键点
    points_coords = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i]
        if v > 0:
            points_coords.append([x, y])
    input_points = np.array(points_coords)
    input_labels = np.ones(len(input_points))

    # 2. 读取图片 (支持中文路径)
    image_data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. 初始化 SAM 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # 4. 生成 Mask (🌟 修复: 强制 float32 提高精度，消除初级噪声)
    print("正在生成高精度分割 Mask...")
    with torch.inference_mode():
        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )

    # 5. Mask 后处理 (🌟 核心修复: 解决红衣服上的马赛克空洞)
    mask_raw = (masks[0] * 255).astype(np.uint8)

    # 使用 5x5 的椭圆核执行闭运算，强行填补内部的小黑洞/芝麻点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 边缘抗锯齿：轻微模糊后重新二值化
    mask_blurred = cv2.GaussianBlur(mask_closed, (3, 3), 0)
    _, mask_final = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)
    mask_bool = (mask_final > 0)

    # 6. 提取人物 (黑底)
    image_subject_only = np.zeros_like(image_rgb)
    image_subject_only[mask_bool] = image_rgb[mask_bool]

    # 7. 扩图 (Padding)
    h, w = image.shape[:2]
    p_h, p_w = int(h * pad_ratio), int(w * pad_ratio)
    image_padded = cv2.copyMakeBorder(
        image_subject_only,
        p_h, p_h, p_w, p_w,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    # 8. 保存高质量 JPG (100质量)
    image_name = os.path.basename(image_path)
    image_dir = os.path.dirname(image_path)
    output_path = os.path.join(image_dir, "padded_" + image_name.rsplit('.', 1)[0] + ".jpg")

    image_padded_bgr = cv2.cvtColor(image_padded, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    is_success, im_buf_arr = cv2.imencode(".jpg", image_padded_bgr, encode_param)

    if is_success:
        im_buf_arr.tofile(output_path)
        print(f"✅ 成功提取主体并扩图: {output_path}")
        # 🌟 根据你的要求：不在这里修改 keypoints，保持原始坐标系，交给下游校准器处理
        return output_path, mask_final
    else:
        raise ValueError("图像编码保存失败")


def main():
    # ================= 配置路径 =================
    annotation_file = "./data/train_final.json"
    image_dir = "./data/eval"
    output_path = "./test_sam2"

    image_dir = Path(image_dir)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [
        f for f in image_dir.rglob('*') if f.suffix.lower() in valid_extensions
    ]
    for img_path in image_files:
        current_output_dir = Path(output_path) / img_path.stem
        current_output_dir.mkdir(parents=True, exist_ok=True)
        kpts_orig, types_orig = read_kpts_annotation(img_path, annotation_file)
        segment_subject(img_path, kpts_orig)

if __name__ == "__main__":
    main()