import os
import sys
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path

# 路径管理
ROOT_DIR = '/home/sora/workspace/SMPLest-X'
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# 导入作者自己的模块
from main.config import Config
from main.base import Tester
from utils.data_utils import load_img, process_bbox, generate_patch_image
from ultralytics import YOLO
import trimesh
import torch
import numpy as np


class ReconstructionEngine:
    def __init__(self, ckpt_name='smplest_x_h', device='cuda'):
        # 1. 核心路径定义（真相大白：是 human_model_files 而不是 data！）
        human_model_path = os.path.join(ROOT_DIR, 'human_models', 'human_model_files')
        checkpoint_path = os.path.join(ROOT_DIR, 'pretrained_models', ckpt_name, f'{ckpt_name}.pth.tar')
        config_path = os.path.join(ROOT_DIR, 'pretrained_models', ckpt_name, 'config_base.py')
        dummy_log_dir = os.path.join(ROOT_DIR, 'outputs', 'api_logs')

        # 2. 初始化 Config
        self.cfg = Config.load_config(config_path)

        # 3. 注入所有关键路径
        new_config = {
            "model": {
                "pretrained_model_path": checkpoint_path,
                "human_model_path": human_model_path
            },
            "log": {
                "exp_name": "api_inference",
                "log_dir": dummy_log_dir
            }
        }
        self.cfg.update_config(new_config)
        self.cfg.prepare_log()

        # 🔥 强行挂载全局的 SMPLX
        from human_models.human_models import SMPLX
        try:
            self.smpl_x = SMPLX(human_model_path)
        except Exception as e:
            # 万一他又包了一层，做个防御性退化
            print(f"警告: 默认路径挂载失败，尝试备用路径... ({e})")
            SMPLX.config = human_model_path

        # 4. 初始化 Tester
        self.demoer = Tester(self.cfg)
        self.demoer._make_model()
        self.demoer.model.eval()

        # 5. YOLO
        bbox_model_path = os.path.join(ROOT_DIR, 'pretrained_models/yolov8x.pt')
        self.detector = YOLO(bbox_model_path)

        self.transform = transforms.ToTensor()
        self.device = torch.device(device)
        print(">>> 🚀 终于搞定了！模型已进入显存。")

    @torch.no_grad()
    def predict(self, image_path):
        original_img = load_img(image_path)
        height, width = original_img.shape[:2]

        # YOLO 检测
        yolo_results = self.detector.predict(original_img, device='cuda', classes=0, verbose=False)
        yolo_bbox = yolo_results[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox) < 1:
            return None

        # 构建 xywh
        det = yolo_bbox[0]
        yolo_bbox_xywh = np.array([det[0], det[1], det[2] - det[0], det[3] - det[1]])

        # 调用 process_bbox (这会返回 [top_left_x, top_left_y, width, height])
        bbox = process_bbox(
            bbox=yolo_bbox_xywh,
            img_width=width,
            img_height=height,
            input_img_shape=self.cfg.model.input_img_shape
        )

        img_patch, _, _ = generate_patch_image(
            cvimg=original_img,
            bbox=bbox,
            scale=1.0,
            rot=0.0,
            do_flip=False,
            out_shape=self.cfg.model.input_img_shape
        )

        img_tensor = self.transform(img_patch.astype(np.float32)) / 255
        img_tensor = img_tensor.cuda()[None, :, :, :]
        inputs = {'img': img_tensor}

        out = self.demoer.model(inputs, {}, {}, 'test')

        joints_3d = out['smplx_joint_cam'].detach().cpu().numpy()[0]
        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

        # vertices = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
        # regressor = self.smpl_x.J_regressor
        # synced_joints = np.matmul(regressor, vertices)
        # aligned_144 = self.align_joints(joints_3d, synced_joints)
        # ==========================================
        # 👑 官方解法：按 BBox 比例直接缩放相机内参
        # ==========================================
        # 为了防报错，做个属性兼容
        input_shape = getattr(self.cfg.model, 'input_body_shape', self.cfg.model.input_img_shape)

        global_focal = [
            self.cfg.model.focal[0] / input_shape[1] * bbox[2],
            self.cfg.model.focal[1] / input_shape[0] * bbox[3]
        ]
        global_princpt = [
            self.cfg.model.princpt[0] / input_shape[1] * bbox[2] + bbox[0],
            self.cfg.model.princpt[1] / input_shape[0] * bbox[3] + bbox[1]
        ]

        return {
            'joints_3d': joints_3d,
            'mesh': mesh,
            'bbox': bbox,
            'global_cam': {
                'focal': np.array(global_focal),
                'princpt': np.array(global_princpt)
            }
        }

    def align_joints(self, original_joints, synced_joints):
        """
        使用 Procrustes 分析将 original_joints (144点) 对齐到 synced_joints (55点)
        """
        common_idx = np.arange(22)
        A = original_joints[common_idx]
        B = synced_joints[common_idx]

        # 1. 计算质心
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        # 2. 去中心化
        AA = A - centroid_A
        BB = B - centroid_B

        # 3. 计算缩放因子 s (不计算旋转矩阵 R)
        # 用去中心化后的向量模长比值代表缩放
        s = np.mean(np.linalg.norm(BB, axis=1)) / np.mean(np.linalg.norm(AA, axis=1))

        # 4. 计算平移向量 t
        # 因为没有旋转，直接用目标质心减去缩放后的源质心
        t = centroid_B - s * centroid_A

        # 5. 应用变换到 144 个点
        aligned_joints_144 = s * original_joints + t

        return aligned_joints_144

    def predict_mesh(self, image_path, save_path: str):
        res = self.predict(image_path)
        if res is None:
            return None, None, None

        vertices = res['mesh']
        faces = self.smpl_x.face
        import trimesh
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(save_path)
        print(f"[SMPLest-X] ✨ 成功! Mesh 已保存至: {save_path}")

        regressor = self.smpl_x.J_regressor
        synced_joints = np.matmul(regressor, vertices)
        raw_144 = res['joints_3d']

        # 针对五官，我们只做一个极简的平移 (让 raw_144 的骨盆对齐到 synced_joints 的骨盆)
        # 绝对不用 SVD 旋转！
        root_raw = raw_144[0]
        root_synced = synced_joints[0]

        # 2. 去中心化（把两套骨架都拉回原点）
        raw_centered = raw_144 - root_raw
        synced_centered = synced_joints - root_synced
        scale = np.mean(np.linalg.norm(synced_centered[1:22], axis=1)) / \
                (np.mean(np.linalg.norm(raw_centered[1:22], axis=1)) + 1e-8)
        safe_144 = raw_centered * scale + root_synced
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

        # 🌟 直接返回绝对精准的全局相机参数
        return save_path, pred_joints_dict, res['global_cam'], mesh


# if __name__ == "__main__":
#     # 测试一下
#     api = SMPLestXPredictor(ckpt_name='smplest_x_h')
#     res = api.predict('/home/sora/workspace/SMPLest-X/baidu_841.jpg')
#     if res:
#         raw_joints = res['joints_3d']
#         mesh_vertices = res['mesh']  # 这就是 3D 表面的顶点坐标 (N, 3)
#
#         print(f"成功提取 3D 节点，维度: {raw_joints.shape}")
#         print(f"成功提取 3D 网格顶点，维度: {mesh_vertices.shape}")
#
#         # ==========================================
#         # 导出 OBJ 文件的魔法代码
#         # ==========================================
#         # 1. 拿到模型的 Faces (面片连接关系)
#         faces = api.smpl_x.face
#
#         # 2. 指定输出路径
#         obj_path = "/home/sora/workspace/SMPLest-X/output_mesh.obj"
#
#         # 3. 写入 OBJ 格式
#         with open(obj_path, 'w') as f:
#             # 写入所有的顶点 (v x y z)
#             for v in mesh_vertices:
#                 f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
#
#             # 写入所有的面 (f v1 v2 v3)，注意 OBJ 的索引是从 1 开始的
#             for face in faces:
#                 f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
#
#         print(f"🎉 OBJ 文件已成功导出至: {obj_path}")
#         print("赶紧把它拖进 Blender 或者 3D Viewer 里看看吧！")