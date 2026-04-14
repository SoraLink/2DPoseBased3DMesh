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
        """执行推理并导出 OBJ"""
        print("[3D Engine] 预测 3D 拓扑...")
        batch_images = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model({'img': batch_images})
            vertices = out['pred_vertices'][0].cpu().numpy()
            faces = self.model.smpl.faces

        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(save_path)
        print(f"✨ 成功! Mesh 已保存至: {save_path}")
        return save_path