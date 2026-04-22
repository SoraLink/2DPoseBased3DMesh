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
        self.model_root = os.path.expanduser('~/workspace/HSMR/data_inputs/released_models/HSMR-ViTH-r1d1')
        self.device = device

        # ⛩️ 1. 官方 Detector
        self.detector = build_detector(batch_size=1, max_img_size=512, device=self.device)

        # ⛩️ 2. 官方 HSMR Pipeline
        # 这个 pipeline 启动后，内部会自动加载它自带的 SMPL-X/SKEL 模型
        self.pipeline = build_inference_pipeline(model_root=self.model_root, device=self.device)

        # ⛩️ 3. 关键：直接从 pipeline 引用模型信息
        self.faces = self.pipeline.skel_model.skin_f.detach().cpu().numpy()
        self.j_regressor = self.pipeline.skel_model.J_regressor.detach().cpu().numpy()

        print(">>> ✅ 已通过 HSMR 内部 skel_model 获取 SMPL-X 拓扑，无需外部库。")

    @torch.no_grad()
    def predict(self, image_path):
        # 1. 官方读取方式
        img = cv2.imread(image_path)
        if img is None: return None
        raw_h, raw_w = img.shape[:2]
        raw_imgs = [img]

        # 2. ⛩️ 官方 Detector (ViTDet)
        detector_outputs = self.detector(raw_imgs)

        # 3. ⛩️ 官方 Patching 逻辑
        # imgs_det2patches 会把 BBox 自动对齐到 192:256 比例，这是 HSMR 精度的关键
        patches, det_meta = imgs_det2patches(raw_imgs, *detector_outputs, max_instances_per_img=1)

        if len(patches) == 0:
            print("🚫 没有检测到人物")
            return None

        # 4. ⛩️ 官方预处理与推理
        patches_normalized = (patches - IMG_MEAN_255) / IMG_STD_255
        patches_normalized = torch.from_numpy(patches_normalized).permute(0, 3, 1, 2).to(self.device)

        outputs = self.pipeline(patches_normalized)

        # 获取 3D 数据
        pd_params = {k: v.detach().cpu() for k, v in outputs['pd_params'].items()}
        pd_cam_t = outputs['pd_cam_t'].detach().cpu()

        # 5. ⛩️ 官方 Mesh 准备 (获取顶点)
        m_skin, _ = prepare_mesh(self.pipeline, pd_params)
        vertices = m_skin['v'][0].numpy()

        # 6. ⛩️ 官方相机修正公式 (从 Visualize_full_img 抄过来的)
        raw_cx, raw_cy = raw_w / 2.0, raw_h / 2.0
        # bbx_cs 包含 [center_x, center_y, scale]
        bbx_cs = det_meta['bbx_cs'][0]

        corrected_cam_t = pd_cam_t[0].clone()
        # 深度 Z 修正
        corrected_cam_t[2] = pd_cam_t[0, 2] * 256.0 / bbx_cs[2]
        # X, Y 位移修正 (5000 是官方焦距常数)
        corrected_cam_t[0] += (bbx_cs[0] - raw_cx) / 5000.0 * corrected_cam_t[2]
        corrected_cam_t[1] += (bbx_cy - raw_cy) / 5000.0 * corrected_cam_t[2]

        # 同步修正顶点位置，让它在全局坐标系下
        global_vertices = vertices + (corrected_cam_t.numpy() - pd_cam_t[0].numpy())

        return {
            'vertices': global_vertices,
            'bbox_cs': bbx_cs,
            'global_cam': {
                'focal': np.array([5000.0, 5000.0]),
                'princpt': np.array([raw_cx, raw_cy])
            }
        }

    @torch.no_grad()
    def predict_mesh(self, image_path, save_path):
        res = self.predict(image_path)
        if res is None: return None

        vertices = res['vertices']  # (10475, 3)

        # ⛩️ 使用内部回归矩阵计算 55 个关节点
        # HSMR 的 regressor 通常已经在 GPU 上，这里 res['vertices'] 如果是 numpy 需要处理下
        synced_joints = np.matmul(self.j_regressor, vertices)

        # 导出 Mesh (使用内部获取的 faces)
        import trimesh
        mesh_obj = trimesh.Trimesh(vertices, self.faces)
        mesh_obj.export(save_path)

        # ⛩️ 组装关节点字典
        pred_joints_dict = {
            # 躯干 24 点 (使用 synced_joints)
            'pelvis': synced_joints[0],
            'left_hip': synced_joints[1],
            'right_hip': synced_joints[2],
            'left_knee': synced_joints[4],
            'right_knee': synced_joints[5],
            'left_ankle': synced_joints[7],
            'right_ankle': synced_joints[8],
            'neck': synced_joints[12],
            'left_shoulder': synced_joints[16],
            'right_shoulder': synced_joints[17],
            'left_elbow': synced_joints[18],
            'right_elbow': synced_joints[19],
            'left_wrist': synced_joints[20],
            'right_wrist': synced_joints[21],

            # 五官点 (SMPL-X 拓扑固定，索引不变)
            'nose': vertices[9120],
            'left_eye': vertices[9448],
            'right_eye': vertices[9929],
            'left_ear': vertices[6],
            'right_ear': vertices[616],
        }

        return save_path, pred_joints_dict, res['global_cam']