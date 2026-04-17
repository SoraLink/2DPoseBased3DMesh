import json
from pathlib import Path

import cv2
import numpy as np
import torch
import os  # 新增 os 库处理路径
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from pose_extractor import read_kpts_annotation

class SAM2Predictor:
    def __init__(self):
        sam2_checkpoint = "./models/sam2/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        # ============================================

        # 🌟 修复 2：在循环外只初始化一次模型
        print("正在加载 SAM 2 模型 (只加载一次)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                       linewidth=1.25)


    def segment_subject(self, image_path, output_path, keypoints, pad_ratio=0.3):
        """
        SAM 2 图像分割与扩图函数 (防马赛克、高精度版)
        """

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

        # 4. 生成 Mask (🌟 修复: 强制 float32 提高精度，消除初级噪声)
        print("正在生成高精度分割 Mask...")
        with torch.inference_mode():
            self.predictor.set_image(image_rgb)
            masks, scores, logits = self.predictor.predict(
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
        output_path = os.path.join(output_path, "padded_" + image_name.rsplit('.', 1)[0] + ".jpg")

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


    def segment_subject2(self, image_path, output_dir, keypoints, types, pad_ratio=0.3):
        """
        SAM 2 图像分割与扩图函数 (带关键点可视化)
        """
        # 1. 提取有效关键点
        points_coords = []
        input_labels = []
        # 假设 read_kpts_annotation 返回的是 [N, 3] 的数组
        for i, pt in enumerate(keypoints):
            x, y, v = pt[0], pt[1], pt[2]
            if v > 0 and types[i] != 2:
                points_coords.append([x, y])
                if types[i] == 0:
                    input_labels.append(1)
                elif types[i] == 1:
                    input_labels.append(0)


        if not points_coords:
            print(f"⚠️ 跳过 {image_path}: 未找到有效可见点")
            return None, None

        input_points = np.array(points_coords)
        input_labels = np.array(input_labels)

        # 2. 读取图片 (支持中文路径)
        image_data = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. 生成 Mask
        print(f"正在处理: {os.path.basename(image_path)}")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image_rgb)
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )

        areas = np.sum(masks, axis=(1, 2))
        largest_idx = np.argmax(areas)

        # 强制使用最大的 Mask
        mask_raw = (masks[largest_idx] * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel, iterations=2)

        mask_blurred = cv2.GaussianBlur(mask_closed, (3, 3), 0)
        _, mask_final = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)
        mask_bool = (mask_final > 0)

        # 5. 提取人物 (黑底)
        image_subject_only = np.full_like(image_rgb, 255)
        image_subject_only[mask_bool] = image_rgb[mask_bool]

        # 6. 扩图 (Padding)
        h, w = image.shape[:2]
        p_h, p_w = int(h * pad_ratio), int(w * pad_ratio)
        image_padded = cv2.copyMakeBorder(
            image_subject_only,
            p_h, p_h, p_w, p_w,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        image_padded_bgr = cv2.cvtColor(image_padded, cv2.COLOR_RGB2BGR)

        # =======================================================
        # 🌟 新增：在最终的 Padded 图片上绘制 Keypoints
        # =======================================================
        for i, pt in enumerate(keypoints):
            x, y, v = pt[0], pt[1], pt[2]
            pt_type = types[i]
            # 只绘制置信度大于 0 (或你需要的阈值) 的点
            if v > 0:
                # 核心：因为图像四周加了黑边，原本的坐标必须加上 offset
                shifted_x = int(x + p_w)
                shifted_y = int(y + p_h)
                if pt_type == 0:
                    inner_color = (0, 255, 0)  # 绿色：正样本 (SAM Label = 1)
                elif pt_type == 1:
                    inner_color = (0, 0, 255)  # 红色：负样本 (SAM Label = 0)
                else:
                    inner_color = (255, 0, 0)  # 蓝色：忽略点或其他 (types == 2)

                # 画一个带白边的绿色实心圆，方便在黑底和人物衣服上都能看清
                cv2.circle(image_padded_bgr, (shifted_x, shifted_y), radius=6, color=(255, 255, 255), thickness=2)
                # 画实心内圆
                cv2.circle(image_padded_bgr, (shifted_x, shifted_y), radius=4, color=inner_color, thickness=-1)
                # 可选：如果想标出关键点的 ID，可以解开下面这行的注释
                # cv2.putText(image_padded_bgr, str(i), (shifted_x + 5, shifted_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # =======================================================

        # 7. 保存高质量 JPG 到对应子文件夹
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, "padded2_" + image_name.rsplit('.', 1)[0] + ".jpg")

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        is_success, im_buf_arr = cv2.imencode(".jpg", image_padded_bgr, encode_param)

        if is_success:
            im_buf_arr.tofile(output_path)
            print(f"✅ 成功提取、扩图并标记关键点: {output_path}")
            return output_path, mask_final
        else:
            raise ValueError("图像编码保存失败")


def main():
    # ================= 配置路径 =================
    annotation_file = "./data/train_final.json"
    image_dir = "./data/eval"
    output_path = "./test_sam2"
    # ============================================

    # 🌟 修复 2：在循环外只初始化一次模型
    print("正在加载 SAM 2 模型 (只加载一次)...")
    image_dir_path = Path(image_dir)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [
        f for f in image_dir_path.rglob('*') if f.suffix.lower() in valid_extensions
    ]
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = SAM2Predictor()
    for img_path in image_files:
        # 🌟 修复 3：创建独立的子文件夹 (e.g., ./test_sam2/image_001/)
        try:
            kpts_orig, kpts, types_orig = read_kpts_annotation(str(img_path), annotation_file)
            # 传入 predictor 和独立的输出目录
            predictor.segment_subject2(str(img_path), str(output_dir), kpts, types_orig)
        except Exception as e:
            raise e


if __name__ == "__main__":
    main()