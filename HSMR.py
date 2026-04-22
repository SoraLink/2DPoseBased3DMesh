import os
import torch
import numpy as np
import cv2
from pathlib import Path

# 导入你提供的官方库组件
from lib.modeling.pipelines.hsmr import build_inference_pipeline
from lib.modeling.pipelines.vitdet import build_detector
from lib.kits.hsmr_demo import imgs_det2patches, prepare_mesh

# 官方预处理参数
IMG_MEAN_255 = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.
IMG_STD_255 = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.


class ReconstructionEngine:
    def __init__(self, device='cuda:0'):
        # 1. 路径定义（严格指向 ViTH-r1d1 目录）
        self.model_root = os.path.expanduser('~/workspace/HSMR/data_inputs/released_models/HSMR-ViTH-r1d1')
        self.device = device

        # ⛩️ 官方初始化顺序：1. Detector -> 2. Pipeline
        print('🧱 Building detector (ViTDet)...')
        self.detector = build_detector(
            batch_size=1,  # 单图预测
            max_img_size=512,
            device=self.device
        )

        print('🧱 Building recovery pipeline (HSMR)...')
        self.pipeline = build_inference_pipeline(
            model_root=self.model_root,
            device=self.device
        )

        # 为了 predict_mesh 导出，依然挂载 SMPLX
        from human_models.human_models import SMPLX
        human_model_path = os.path.join(os.path.dirname(self.model_root), '../../../../human_models/human_model_files')
        # 注意：这里路径根据你实际存放位置调整，或者写死绝对路径
        self.smpl_x = SMPLX(os.path.expanduser('~/workspace/HSMR/human_models/human_model_files'))

    @torch.no_grad()
    def predict(self, image_path):
        # ⛩️ 1. Preprocess (官方 Load Inputs & Detecting)
        img = cv2.imread(image_path)
        if img is None: return None
        raw_imgs = [img]  # 包装成 list 适配 detector

        # 调用官方 Detector
        detector_outputs = self.detector(raw_imgs)

        # 调用官方 Patching (关键：这里会处理 BBox 并切好 256x256 的图)
        patches, det_meta = imgs_det2patches(raw_imgs, *detector_outputs, max_instances_per_img=1)

        if len(patches) == 0:
            print("🚫 No human instance detected.")
            return None

        # ⛩️ 2. Recovery (官方 Recovery 循环)
        # 归一化
        patches_normalized = (patches - IMG_MEAN_255) / IMG_STD_255
        patches_normalized = torch.from_numpy(patches_normalized).permute(0, 3, 1, 2).to(self.device)

        # 模型推理
        outputs = self.pipeline(patches_normalized)

        # 提取参数
        pd_params = {k: v.detach().cpu() for k, v in outputs['pd_params'].items()}
        pd_cam_t = outputs['pd_cam_t'].detach().cpu()

        # ⛩️ 3. Prepare Meshes (调用官方 prepare_mesh 获取顶点)
        # 注意：官方这个函数返回 m_skin['v'], m_skin['f']
        m_skin, _ = prepare_mesh(self.pipeline, pd_params)

        # 计算全局相机参数 (沿用官方渲染逻辑中的修正)
        raw_h, raw_w = img.shape[:2]
        raw_cx, raw_cy = raw_w / 2.0, raw_h / 2.0
        bbx_cs = det_meta['bbx_cs'][0]  # [cx, cy, s]

        # 官方相机平移修正逻辑
        corrected_cam_t = pd_cam_t[0].clone()
        corrected_cam_t[2] = pd_cam_t[0, 2] * 256.0 / bbx_cs[2]
        corrected_cam_t[1] += (bbx_cs[1] - raw_cy) / 5000.0 * corrected_cam_t[2]
        corrected_cam_t[0] += (bbx_cs[0] - raw_cx) / 5000.0 * corrected_cam_t[2]

        return {
            'pd_params': pd_params,
            'vertices': m_skin['v'][0].numpy(),
            'cam_t': corrected_cam_t.numpy(),
            'bbox': bbx_cs,  # 官方返回的是 [cx, cy, s]
            'global_cam': {
                'focal': np.array([5000.0, 5000.0]),
                'princpt': np.array([raw_cx, raw_cy])
            }
        }

    def predict_mesh(self, image_path, save_path):
        res = self.predict(image_path)
        if res is None: return None

        vertices = res['vertices']  # (10475, 3)
        faces = self.smpl_x.face

        # ⛩️ 关键步骤：利用回归矩阵计算标准的 55/24 个关节点
        # regressor 是 (55, 10475) 的矩阵
        regressor = self.smpl_x.J_regressor
        synced_joints = np.matmul(regressor, vertices)  # 得到 (55, 3)

        # 导出 Mesh
        import trimesh
        mesh_obj = trimesh.Trimesh(vertices, faces)
        mesh_obj.export(save_path)

        # ⛩️ 组装完整的 24+5 关键点字典
        # 这里的索引 0-23 遵循 SMPL-X 标准骨架定义
        pred_joints_dict = {
            # --- 躯干与四肢 (来自回归矩阵) ---
            'pelvis': synced_joints[0],
            'left_hip': synced_joints[1],
            'right_hip': synced_joints[2],
            'spine1': synced_joints[3],
            'left_knee': synced_joints[4],
            'right_knee': synced_joints[5],
            'spine2': synced_joints[6],
            'left_ankle': synced_joints[7],
            'right_ankle': synced_joints[8],
            'spine3': synced_joints[9],
            'left_foot': synced_joints[10],
            'right_foot': synced_joints[11],
            'neck': synced_joints[12],
            'left_collar': synced_joints[13],
            'right_collar': synced_joints[14],
            'head': synced_joints[15],
            'left_shoulder': synced_joints[16],
            'right_shoulder': synced_joints[17],
            'left_elbow': synced_joints[18],
            'right_elbow': synced_joints[19],
            'left_wrist': synced_joints[20],
            'right_wrist': synced_joints[21],
            'left_jaw': synced_joints[22],
            'right_jaw': synced_joints[23],

            # --- 五官点 (直接从顶点索引取，最精准) ---
            'nose': vertices[9120],
            'left_eye': vertices[9448],
            'right_eye': vertices[9929],
            'left_ear': vertices[6],
            'right_ear': vertices[616],
        }

        print(f"✨ 导出完成：包含 24 个骨架点和 5 个五官点。")
        return save_path, pred_joints_dict, res['global_cam']