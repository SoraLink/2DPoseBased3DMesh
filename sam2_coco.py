import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os  # 新增 os 库处理路径
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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
        return output_path
    else:
        raise ValueError("图像编码保存失败")


def main():
    # ================= 配置路径 =================
    coco_json_path = "./data/ldpose_train_25kpts.json"
    image_dir = "./data/residual_examples"
    target_image_name = "baidu_残疾运动员_841.jpg"

    # SAM 2 模型配置
    sam2_checkpoint = "./models/sam2/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # ============================================

    # 1. 解析 COCO 文件，寻找图片和关键点
    print(f"正在读取 COCO 标注文件...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    image_id = None
    for img_info in coco_data['images']:
        if img_info['file_name'] == target_image_name:
            image_id = img_info['id']
            break

    if image_id is None:
        raise ValueError(f"在 COCO 文件中找不到图片名: {target_image_name}")

    keypoints = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            kpts_raw = ann['keypoints']
            for i in range(0, len(kpts_raw), 3):
                x, y, v = kpts_raw[i], kpts_raw[i + 1], kpts_raw[i + 2]
                # 只取可见点 (v > 0) 作为 SAM 的提示点
                if v > 0:
                    keypoints.append([x, y])
            break

    if not keypoints:
        raise ValueError("未找到有效的可见关键点！")

    input_points = np.array(keypoints)
    input_labels = np.ones(len(input_points))

    print(f"成功提取 {len(input_points)} 个有效关键点坐标。")

    # 2. 读取图片 (终极防弹版，完美支持中文路径)
    img_path = os.path.join(image_dir, target_image_name)
    print(f"尝试读取图片路径: {img_path}")

    # 使用 numpy 读取字节流再用 cv2 解码
    image_data = np.fromfile(img_path, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # 防御性拦截
    if image is None:
        raise FileNotFoundError(f"无法读取图片！请检查路径是否拼写正确，或者文件是否损坏: {img_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. 初始化 SAM 2 模型
    print("正在加载 SAM 2 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # 4. 推理 (加入半精度与推理模式优化)
    print("正在生成分割 Mask...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)

        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )

    # ================= 新增：裁剪人物并保存透明背景图 =================
    print("正在裁剪人物并生成透明背景图像...")
    # 将 RGB 图像转换为带有 Alpha 通道的 RGBA 图像
    image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    # 获取掩码（通常是 True/False 或 1/0），并将其映射到 0-255 的透明度
    # masks[0] 对应置信度最高的那个 mask
    alpha_channel = (masks[0] * 255).astype(np.uint8)

    # 将透明度通道赋值给 RGBA 图像的第四个通道
    image_rgba[:, :, 3] = alpha_channel

    # 构造输出路径（替换为 .png 格式以支持透明度）
    output_image_name = "cropped_" + target_image_name.rsplit('.', 1)[0] + ".png"
    output_path = os.path.join(image_dir, output_image_name)

    # 将 RGBA 转为 BGRA 用于 OpenCV 保存
    image_bgra = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA)

    # 使用 imencode 防弹法保存，完美绕过中文路径报错
    is_success, im_buf_arr = cv2.imencode(".png", image_bgra)
    if is_success:
        im_buf_arr.tofile(output_path)
        print(f"✅ 裁剪成功！透明背景人像已保存至: {output_path}")
    else:
        print("❌ 保存裁剪图片失败！")
    # =================================================================

    # 5. 可视化结果
    print("推理完成，正在显示可视化结果...")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_points(input_points, input_labels, plt.gca())
    plt.title(f"SAM 2 Segmentation (Score: {scores[0]:.3f})", fontsize=18)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()