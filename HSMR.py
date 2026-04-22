import os
import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

# 假设你的 HSMR 库在 lib.kits 或对应目录下
from lib.kits.hsmr_demo import build_inference_pipeline, IMG_MEAN_255, IMG_STD_255


class ReconstructionEngine:
    def __init__(self, device='cuda'):
        # 1. HSMR 初始化 (HSMR 通常自带模型权重管理，只需指定 root)
        # model_root 应该包含 hsmr_models 和对应的配置文件
        self.model_root = os.path.join(ROOT_DIR, 'pretrained_models', 'hsmr')
        self.device = torch.device(device)

        print(">>> 🧱 正在加载 HSMR 推理流水线...")
        self.pipeline = build_inference_pipeline(model_root=self.model_root, device=device)

        # 2. 依然保留外部使用的 SMPLX 层用于导出 Mesh
        from human_models.human_models import SMPLX
        human_model_path = os.path.join(ROOT_DIR, 'human_models', 'human_model_files')
        self.smpl_x = SMPLX(human_model_path)

        # 3. YOLO 检测器保持不变
        bbox_model_path = os.path.join(ROOT_DIR, 'pretrained_models/yolov8x.pt')
        self.detector = YOLO(bbox_model_path)

        print(">>> 🚀 HSMR 已就绪！")

    @torch.no_grad()
    def predict(self, image_path):
        original_img = cv2.imread(image_path)  # BGR
        if original_img is None: return None
        h, w = original_img.shape[:2]

        # 1. YOLO 检测
        yolo_results = self.detector.predict(original_img, device=self.device, classes=0, verbose=False)
        boxes = yolo_results[0].boxes.xyxy.detach().cpu().numpy()
        if len(boxes) < 1: return None

        # 2. 预处理 Patch (HSMR 喜欢 256x256)
        # 注意：HSMR 内部其实有 imgs_det2patches，但为了兼容你的 predict 逻辑，我们手动裁剪
        det = boxes[0]
        # 稍微放大点 BBox 以包含全身，HSMR 对裁剪边缘较敏感
        center = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]
        scale = max(det[2] - det[0], det[3] - det[1]) * 1.2

        # 简化版裁剪（建议使用 HSMR 自带的裁剪工具类以获得最佳效果）
        # 这里演示核心逻辑：输入必须是 (3, 256, 256) 且经过特定归一化
        img_patch = self._crop_and_resize(original_img, center, scale, 256)

        # HSMR 特有的归一化
        patch_tensor = (img_patch.astype(np.float32) - IMG_MEAN_255) / IMG_STD_255
        patch_tensor = torch.from_numpy(patch_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 3. HSMR 推理
        # HSMR 返回的是一个复杂的 dict，包含全层级的参数
        outputs = self.pipeline(patch_tensor)

        # 提取关键数据
        # vertices 是在相机空间下的 (N, 10475, 3)
        mesh = outputs['pd_vertices'].detach().cpu().numpy()[0]
        # pd_cam_t 是相机平移 (N, 3)
        cam_t = outputs['pd_cam_t'].detach().cpu().numpy()[0]

        # HSMR 通常直接输出对齐后的 3D 关节
        joints_3d = outputs['pd_joints'].detach().cpu().numpy()[0]

        # 4. 计算全局相机参数 (HSMR 相机逻辑)
        # HSMR 的相机模型通常假设 focal 是 5000 (或者配置中定义的常数)
        # 映射回原图需要利用 BBox 的尺度
        focal_length = 5000.0  # HSMR 默认虚拟焦距

        # 计算在原图尺度下的真实焦距
        global_focal = focal_length * (max(h, w) / 256.0)  # 粗略估计

        return {
            'joints_3d': joints_3d,
            'mesh': mesh,
            'bbox': [det[0], det[1], det[2] - det[0], det[3] - det[1]],
            'cam_t': cam_t,
            'global_cam': {
                'focal': np.array([global_focal, global_focal]),
                'princpt': np.array([w / 2, h / 2])  # 简化处理：主点居中
            }
        }

    def _crop_and_resize(self, img, center, scale, res):
        # 这是一个简化的裁剪函数，实际项目中建议引用 HSMR 内部的裁剪逻辑
        x1, y1 = int(center[0] - scale / 2), int(center[1] - scale / 2)
        x2, y2 = int(center[0] + scale / 2), int(center[1] + scale / 2)

        # 边界处理...
        crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
        return cv2.resize(crop, (res, res))

    def predict_mesh(self, image_path, save_path: str):
        res = self.predict(image_path)
        if res is None: return None, None, None

        vertices = res['mesh']
        # HSMR 的 Mesh 是基于标准 SMPL-X 拓扑的
        faces = self.smpl_x.face
        import trimesh
        # 注意：HSMR 的输出可能需要根据 cam_t 进行平移对齐
        # vertices = vertices + res['cam_t']

        mesh_obj = trimesh.Trimesh(vertices, faces)
        mesh_obj.export(save_path)

        # 关节对齐逻辑 (保留你之前的精准 Pelvis 对齐)
        regressor = self.smpl_x.J_regressor
        synced_joints = np.matmul(regressor, vertices)
        raw_joints = res['joints_3d']

        # 偏移修正
        offset = synced_joints[0] - raw_joints[0]
        final_joints = raw_joints + offset

        pred_joints_dict = {
            # === Part A: 躯干和四肢 (必须用 synced_joints，保证截肢 100% 准确) ===
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

            # === Part B: 五官点 (用平移修正后的 safe_144) ===
            'nose': vertices[9120],
            'left_eye': vertices[9448],
            'right_eye': vertices[9929],
            'left_ear': vertices[6],
            'right_ear': vertices[616],
        }

        return save_path, pred_joints_dict, res['global_cam']