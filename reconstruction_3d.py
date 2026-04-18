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
        # 1. 读图
        original_img = load_img(image_path)
        height, width = original_img.shape[:2]

        # 2. YOLO 检测
        yolo_results = self.detector.predict(original_img, device='cuda', classes=0, verbose=False)
        yolo_bbox = yolo_results[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox) < 1:
            return None

        # 3. 仿照作者逻辑处理 bbox (取第一个人)
        det = yolo_bbox[0]
        yolo_bbox_xywh = np.array([det[0], det[1], det[2] - det[0], det[3] - det[1]])

        bbox = process_bbox(
            bbox=yolo_bbox_xywh,
            img_width=width,
            img_height=height,
            input_img_shape=self.cfg.model.input_img_shape
        )

        # 4. 生成 Patch (抠图)
        img_patch, _, _ = generate_patch_image(
            cvimg=original_img,
            bbox=bbox,
            scale=1.0,
            rot=0.0,
            do_flip=False,
            out_shape=self.cfg.model.input_img_shape
        )

        # 5. 准备输入 Tensor
        img_tensor = self.transform(img_patch.astype(np.float32)) / 255
        img_tensor = img_tensor.cuda()[None, :, :, :]

        inputs = {'img': img_tensor}

        # 6. 核心推理
        # 'test' 是作者代码要求的 mode
        out = self.demoer.model(inputs, {}, {}, 'test')

        # 7. 提取数据
        return {
            'joints_3d': out['smplx_joint_cam'].detach().cpu().numpy()[0],  # 3D 关键点
            'mesh': out['smplx_mesh_cam'].detach().cpu().numpy()[0],  # 3D 网格
            'bbox': bbox  # 抠图用的框，可能后续评估需要
        }


    def predict_mesh(self, image_path, save_path: str):
        """
        完全兼容 4D-Humans 风格的推理函数
        返回: save_path (OBJ路径), pred_joints_dict (14个核心关节), pred_cam (相机参数)
        """
        # 1. 执行推理 (调用我们之前写好的推理逻辑)
        # 注意：这里我们直接拿到内存中的结果，不需要反复读写硬盘
        res = self.predict(image_path)
        if res is None:
            return None, None, None

        # 2. 提取顶点和面片
        vertices = res['mesh']  # (10475, 3) 对于 SMPL-X
        faces = self.smpl_x.face  # SMPL-X 的拓扑结构面片索引

        # 3. 导出 OBJ 文件 (使用 trimesh，保持与您原代码一致)
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(save_path)
        print(f"[SMPLest-X] ✨ 成功! Mesh 已保存至: {save_path}")

        # 4. 提取关节坐标 (从 137 个关节中提取前 24 个，它们与 SMPL 定义一致)
        joints_3d = res['joints_3d']

        # ========================================================
        # 🌟 COCO 17 关键点 + SMPL 原有核心点 完美映射
        # ========================================================
        pred_joints_dict = {
            # ----- 1. 头部 (COCO 0-4) -----
            # 鼻尖：SMPL-X 附带的 68 面部特征点中的第 30 个 (55 + 30 = 85)
            'nose': joints_3d[85],
            'left_eye': joints_3d[23],  # SMPL-X 标准左眼关节
            'right_eye': joints_3d[24],  # SMPL-X 标准右眼关节
            # 耳朵：标准 137 点中无显式耳朵，使用面部外轮廓(0 和 16)的极点作为高精度近似
            'left_ear': joints_3d[71],  # 左耳底/下颌角起点 (55 + 16 = 71)
            'right_ear': joints_3d[55],  # 右耳底/下颌角起点 (55 + 0 = 55)

            # ----- 2. 躯干与上肢 (COCO 5-10) -----
            'left_shoulder': joints_3d[16],
            'right_shoulder': joints_3d[17],
            'left_elbow': joints_3d[18],
            'right_elbow': joints_3d[19],
            'left_wrist': joints_3d[20],
            'right_wrist': joints_3d[21],

            # ----- 3. 下肢 (COCO 11-16) -----
            'left_hip': joints_3d[1],
            'right_hip': joints_3d[2],
            'left_knee': joints_3d[4],
            'right_knee': joints_3d[5],
            'left_ankle': joints_3d[7],
            'right_ankle': joints_3d[8],

            # ----- 4. 兼容附加核心点 -----
            'pelvis': joints_3d[0],  # 全局根节点
            'neck': joints_3d[12],  # 颈部
        }

        # 6. 处理相机参数
        # 4D-Humans 返回的是 [s, tx, ty] (弱透视相机)
        # SMPLest-X 默认返回的是完整的相机内参，为了保持返回格式，我们返回一个包含内参的数组
        # 如果您的下游逻辑需要位移/缩放，可以直接使用 bbox 信息计算
        pred_cam = {
            'focal': self.cfg.model.focal,
            'princpt': self.cfg.model.princpt,
            'bbox': res['bbox']
        }

        return save_path, pred_joints_dict, pred_cam


if __name__ == "__main__":
    # 测试一下
    api = SMPLestXPredictor(ckpt_name='smplest_x_h')
    res = api.predict('/home/sora/workspace/SMPLest-X/baidu_841.jpg')
    if res:
        raw_joints = res['joints_3d']
        mesh_vertices = res['mesh']  # 这就是 3D 表面的顶点坐标 (N, 3)

        print(f"成功提取 3D 节点，维度: {raw_joints.shape}")
        print(f"成功提取 3D 网格顶点，维度: {mesh_vertices.shape}")

        # ==========================================
        # 导出 OBJ 文件的魔法代码
        # ==========================================
        # 1. 拿到模型的 Faces (面片连接关系)
        faces = api.smpl_x.face

        # 2. 指定输出路径
        obj_path = "/home/sora/workspace/SMPLest-X/output_mesh.obj"

        # 3. 写入 OBJ 格式
        with open(obj_path, 'w') as f:
            # 写入所有的顶点 (v x y z)
            for v in mesh_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # 写入所有的面 (f v1 v2 v3)，注意 OBJ 的索引是从 1 开始的
            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

        print(f"🎉 OBJ 文件已成功导出至: {obj_path}")
        print("赶紧把它拖进 Blender 或者 3D Viewer 里看看吧！")