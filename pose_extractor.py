import json
import os

import cv2
import numpy as np
import requests
import torch
import warnings

# MMPose v1.x 导入
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

warnings.filterwarnings("ignore")


class PoseExtractor:
    def __init__(self, config_file: str, checkpoint_file: str, device: str = 'cuda:0'):
        print(f"[Pose] 初始化 LDPose 模型...")
        register_all_modules()
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.model.eval()

        # 判定点是否可见的全局阈值
        self.score_threshold = 0.3

    def extract_31_keypoints(self, image_url: str) -> np.ndarray:
        """
        执行推理并返回清洗后的 31 个关键点矩阵，以及一个包含所有残肢信息的列表。
        返回:
            kpts_31: np.ndarray, shape (31, 3)
        """
        try:
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"❌ 无法下载图像 URL: {e}")

        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("❌ 图像解码失败，URL 可能未返回有效的图片数据。")

        h, w = img.shape[:2]
        bbox = np.array([[0, 0, w, h]])
        with torch.no_grad():
            batch_results = inference_topdown(self.model, img, bboxes=bbox)

        pred_instances = batch_results[0].pred_instances
        keypoints = pred_instances.keypoints[0]          # shape: [31, 2]
        keypoint_scores = pred_instances.keypoint_scores[0] # shape: [31] 👈 提取置信度
        types = pred_instances.keypoint_types[0]

        kpts_31 = np.zeros((31, 3))
        kpts_31[:, :2] = keypoints
        kpts_31[:, 2] = 1

        # 你的自定义逻辑：如果 type != 0 (可能是被遮挡或不存在)，将置信度强制归零
        for i, kpt_type in enumerate(types):
            if kpt_type != 0:
                kpts_31[i, 2] = 0.0

        return kpts_31

def read_kpts_annotation(image_path, annotation_path):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    image_id = None
    image_name = os.path.basename(image_path)
    for img_info in coco_data['images']:
        if img_info['file_name'] == image_name:
            image_id = img_info['id']
            break

    if image_id is None:
        raise ValueError(f"在 COCO 文件中找不到图片名: {image_name}")

    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            kpts = ann['keypoints']
            types = ann['keypoint_types']
            ori_kpts = np.zeros((31, 3))
            for i, kpt_type in enumerate(types):
                x, y, v = kpts[i * 3], kpts[i * 3 + 1], kpts[i * 3 + 2]
                ori_kpts[i, :2] = [x, y]
                ori_kpts[i, 2] = v
                if kpt_type != 0:
                    ori_kpts[i, 2] = 0.0
            return ori_kpts, kpts, types


if __name__ == "__main__":
    extractor = PoseExtractor("./configs/ldpose.py", "./checkpoints/ldpose.pth")
    kpts, stumps_info = extractor.extract_25_keypoints("./data/test.jpg")

    print("清洗后的 25 点矩阵 shape:", kpts.shape)

    if not stumps_info:
        print("✅ 该图像为健全人 (未检测到可见的残肢点)。")
    else:
        print(f"⚠️ 检测到 {len(stumps_info)} 处截肢:")
        for stump in stumps_info:
            print(f"  - 肢体: {stump['limb']}")
            print(f"    上级关节坐标: {stump['joint_upper']}")
            print(f"    残肢端点坐标: {stump['joint_stump']}")