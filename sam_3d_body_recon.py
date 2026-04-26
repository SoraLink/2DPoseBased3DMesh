import os
import sys

import cv2
import numpy as np
import trimesh
SAM3D_DIR = '/home/sora/workspace/sam-3d-body'
if SAM3D_DIR not in sys.path:
    sys.path.append(SAM3D_DIR)

# 这样 Python 就能找到这个目录下的模块了
from utils import setup_sam_3d_body


class ReconstructionEngine:
    def __init__(self, hf_repo_id="facebook/sam-3d-body-dinov3", device='cuda'):
        """
        初始化 SAM 3D Body 引擎
        """
        print(f">>> 正在加载 Meta SAM 3D Body 模型 ({hf_repo_id})...")
        # 直接使用 Meta 提供的 setup 初始化端到端模型
        self.estimator = setup_sam_3d_body(hf_repo_id=hf_repo_id)
        self.device = device
        print(">>> 🚀 SAM 3D Body 模型加载完成！")

    def predict_mesh(self, image_path: str, save_path: str):
        # 1. 读取原图获取尺寸（用于计算相机中心点）
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            raise ValueError(f"❌ 无法读取图像: {image_path}")
        height, width = img_cv2.shape[:2]

        print(f">>> 正在处理图像: {image_path}")

        # 2. 调用 SAM 3D Body 进行推理
        outputs = self.estimator.process_one_image(image_path)

        if not outputs or len(outputs) == 0:
            raise ValueError("❌ 预测失败，SAM 3D Body 未在图像中检测到人物。")

        # 获取第一个人的数据
        person_data = outputs[0]

        # ==========================================
        # 3. 提取 3D 网格 (Mesh) -> 根据 utils.py，确切 Key 是 'pred_vertices'
        # ==========================================
        vertices = person_data["pred_vertices"]
        if hasattr(vertices, 'cpu'):
            vertices = vertices.detach().cpu().numpy()

        faces = self.estimator.faces
        if hasattr(faces, 'cpu'):
            faces = faces.detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(save_path)
        print(f"[SAM-3D-Body] ✨ 成功! Mesh 已保存至: {save_path}")

        # ==========================================
        # 4. 构建全局相机参数 (global_cam) -> 根据 utils.py，使用 focal_length
        # ==========================================
        # SMPLest-X 格式通常需要分别指定 x 和 y 方向的焦距，这里用同一个值
        focal_length = float(person_data["focal_length"])

        global_cam = {
            'focal': np.array([focal_length, focal_length]),
            'princpt': np.array([width / 2.0, height / 2.0]),
            'cam_t': person_data["pred_cam_t"]  # 保留平移参数以防下游需要
        }

        # ==========================================
        # 5. 提取并映射关节点 (pred_joints_dict)
        # ==========================================
        # utils.py 中提取了 'pred_keypoints_2d'，我们提取对应的 3D 关节点
        joints_3d = person_data.get("pred_keypoints_3d")
        if joints_3d is not None and hasattr(joints_3d, 'cpu'):
            joints_3d = joints_3d.detach().cpu().numpy()

        pred_joints_dict = {}
        if joints_3d is not None:
            is_smpl = len(joints_3d) >= 24
            pred_joints_dict = {
                'pelvis': joints_3d[0],
                'left_hip': joints_3d[1] if is_smpl else joints_3d[11],
                'right_hip': joints_3d[2] if is_smpl else joints_3d[12],
                'left_knee': joints_3d[4] if is_smpl else joints_3d[13],
                'right_knee': joints_3d[5] if is_smpl else joints_3d[14],
                'left_ankle': joints_3d[7] if is_smpl else joints_3d[15],
                'right_ankle': joints_3d[8] if is_smpl else joints_3d[16],
                'neck': joints_3d[12] if is_smpl else joints_3d[0],
                'left_shoulder': joints_3d[16] if is_smpl else joints_3d[5],
                'right_shoulder': joints_3d[17] if is_smpl else joints_3d[6],
                'left_elbow': joints_3d[18] if is_smpl else joints_3d[7],
                'right_elbow': joints_3d[19] if is_smpl else joints_3d[8],
                'left_wrist': joints_3d[20] if is_smpl else joints_3d[9],
                'right_wrist': joints_3d[21] if is_smpl else joints_3d[10],
            }

        # 五官部分依然通过顶点索引获取，最稳定
        if len(vertices) > 10000:  # SMPL-X
            pred_joints_dict.update({
                'nose': vertices[9120], 'left_eye': vertices[9448], 'right_eye': vertices[9929],
                'left_ear': vertices[6], 'right_ear': vertices[616],
            })
        else:  # SMPL
            pred_joints_dict.update({
                'nose': vertices[331], 'left_eye': vertices[2802], 'right_eye': vertices[6262],
                'left_ear': vertices[3489], 'right_ear': vertices[3990],
            })

        return save_path, pred_joints_dict, global_cam, mesh