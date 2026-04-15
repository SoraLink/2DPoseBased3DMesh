import os
import torch
import trimesh
from PIL import Image
from torchvision import transforms

# === 核心 Trick：覆盖 torch.load 解决权重加载报错 ===
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# ====================================================

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT

class ReconstructionEngine:
    def __init__(self):
        print("[3D Engine] 初始化 HMR 2.0...")
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model, _ = load_hmr2(DEFAULT_CHECKPOINT)
        self.model = self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_mesh(self, img_pil: Image.Image, save_path: str):
        """执行推理并导出 OBJ，同时返回 3D 关节坐标字典"""
        print("[3D Engine] 预测 3D 拓扑...")
        batch_images = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model({'img': batch_images})
            vertices = out['pred_vertices'][0].cpu().numpy()
            faces = self.model.smpl.faces
            joints_3d = out['pred_keypoints_3d'][0].cpu().numpy()

        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(save_path)
        print(f"✨ 成功! Mesh 已保存至: {save_path}")

        # ⬇️ 新增：构造 SMPL 关节字典 (SMPL 标准前 24 个关节的索引映射)
        # 这样你在 main 函数里就能直接用名字取坐标了，不用去记复杂的数字索引
        pred_joints_dict = {
            'pelvis': joints_3d[0],
            'left_hip': joints_3d[1],
            'right_hip': joints_3d[2],
            'left_knee': joints_3d[4],
            'right_knee': joints_3d[5],
            'left_ankle': joints_3d[7],
            'right_ankle': joints_3d[8],
            'neck': joints_3d[12],
            'left_shoulder': joints_3d[16],
            'right_shoulder': joints_3d[17],
            'left_elbow': joints_3d[18],
            'right_elbow': joints_3d[19],
            'left_wrist': joints_3d[20],
            'right_wrist': joints_3d[21],
        }

        # 现在返回两个值：路径 + 3D 字典
        return save_path, pred_joints_dict