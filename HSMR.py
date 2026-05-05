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
        self.j_regressor = self.pipeline.skel_model.J_regressor.to_dense().detach().cpu().numpy()
        print(">>> ✅ 已通过 HSMR 内部 skel_model 获取 SMPL-X 拓扑，无需外部库。")

    @torch.no_grad()
    def predict(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        raw_h, raw_w = img.shape[:2]
        raw_imgs = [img]

        # 1. 官方检测与切片
        detector_outputs = self.detector(raw_imgs)
        patches, det_meta = imgs_det2patches(raw_imgs, *detector_outputs, max_instances_per_img=1)

        if len(patches) == 0: return None

        # 2. 官方模型推理
        patches_normalized = (patches - IMG_MEAN_255) / IMG_STD_255
        patches_normalized = torch.from_numpy(patches_normalized).permute(0, 3, 1, 2).to(self.device)
        outputs = self.pipeline(patches_normalized)

        # 3. 提取 Mesh 和参数
        pd_params = {k: v.detach().cpu() for k, v in outputs['pd_params'].items()}
        pd_cam_t = outputs['pd_cam_t'].detach().cpu()[0]  # [tx, ty, tz]

        m_skin, _ = prepare_mesh(self.pipeline, pd_params)
        vertices = m_skin['v'][0].numpy()  # 原始 6890 个顶点

        # 4. ⛩️ 官方相机修正逻辑 (确定的证据：来自 lib/kits/hsmr_demo.py)
        raw_cx, raw_cy = raw_w / 2.0, raw_h / 2.0
        bbx_cs = det_meta['bbx_cs'][0]  # [cx, cy, s]
        focal = 5000.0  # 官方硬编码虚拟焦距

        corrected_cam_t = pd_cam_t.clone()
        # 深度修正 (这是最重要的一步，解决了你之前坐标爆炸到几亿的问题)
        corrected_cam_t[2] = pd_cam_t[2] * 256.0 / bbx_cs[2]
        # X, Y 位移补偿
        corrected_cam_t[0] += (bbx_cs[0] - raw_cx) / focal * corrected_cam_t[2]
        corrected_cam_t[1] += (bbx_cs[1] - raw_cy) / focal * corrected_cam_t[2]

        # ⛩️ 5. 将顶点转换到全局原图相机空间
        # 只有这样，外层的 project_mesh_overlay 才能用统一的 K 矩阵投回原图
        global_vertices = vertices + corrected_cam_t.numpy()

        return {
            'vertices': global_vertices,
            'bbox_cs': bbx_cs,
            'global_cam': {
                'focal': np.array([focal, focal]),
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
            'nose': vertices[331],
            'left_eye': vertices[332],
            'right_eye': vertices[329],
            'left_ear': vertices[348],
            'right_ear': vertices[349],
        }

        # 手：沿 elbow -> wrist 方向找 wrist 远端最远 vertex
        left_hand_dir = pred_joints_dict["left_wrist"] - pred_joints_dict["left_elbow"]
        right_hand_dir = pred_joints_dict["right_wrist"] - pred_joints_dict["right_elbow"]

        # 脚：沿 knee -> ankle 方向找 ankle 远端最远 vertex
        # 这里是为了近似完整肢体 hallucination 的最远端。如果你想严格 toe tip，
        # 后续最好可视化确认固定 toe vertex。
        left_foot_dir = pred_joints_dict["left_ankle"] - pred_joints_dict["left_knee"]
        right_foot_dir = pred_joints_dict["right_ankle"] - pred_joints_dict["right_knee"]

        pred_joints_dict.update({
            "L_Middle_Tip": _terminal_vertex_by_direction(
                vertices,
                pred_joints_dict["left_wrist"],
                left_hand_dir,
                radius=0.35,
            ),
            "R_Middle_Tip": _terminal_vertex_by_direction(
                vertices,
                pred_joints_dict["right_wrist"],
                right_hand_dir,
                radius=0.35,
            ),
            "L_Toe_Tip": _terminal_vertex_by_direction(
                vertices,
                pred_joints_dict["left_ankle"],
                left_foot_dir,
                radius=0.45,
            ),
            "R_Toe_Tip": _terminal_vertex_by_direction(
                vertices,
                pred_joints_dict["right_ankle"],
                right_foot_dir,
                radius=0.45,
            ),
        })

        return save_path, pred_joints_dict, res['global_cam'], mesh_obj

    @torch.no_grad()
    def render_wholebody_projection(self, image_path: str, out_path: str, mesh=None, pred_cam=None):
        """
        Whole-body projection using HSMR official style:
        skin mesh + skeleton mesh.
        """
        import cv2
        import torch
        from lib.kits.hsmr_demo import imgs_det2patches, prepare_mesh
        from lib.utils.vis.py_renderer import render_meshes_overlay_img

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        raw_h, raw_w = img.shape[:2]
        raw_cx, raw_cy = raw_w / 2.0, raw_h / 2.0

        raw_imgs = [img]

        detector_outputs = self.detector(raw_imgs)
        patches, det_meta = imgs_det2patches(
            raw_imgs,
            *detector_outputs,
            max_instances_per_img=1,
        )

        if len(patches) == 0:
            raise ValueError("HSMR failed to detect a person.")

        patches_normalized = (patches - IMG_MEAN_255) / IMG_STD_255
        patches_normalized = torch.from_numpy(patches_normalized).permute(0, 3, 1, 2).to(self.device)

        outputs = self.pipeline(patches_normalized)

        pd_params = {k: v.detach().cpu() for k, v in outputs["pd_params"].items()}
        pd_cam_t = outputs["pd_cam_t"].detach().cpu()

        m_skin, m_skel = prepare_mesh(self.pipeline, pd_params)

        bbx_cs = torch.as_tensor(det_meta["bbx_cs"], dtype=torch.float32)

        raw_cam_t = pd_cam_t.clone().float()
        raw_cam_t[:, 2] = pd_cam_t[:, 2] * 256.0 / bbx_cs[:, 2]
        raw_cam_t[:, 0] += (bbx_cs[:, 0] - raw_cx) / 5000.0 * raw_cam_t[:, 2]
        raw_cam_t[:, 1] += (bbx_cs[:, 1] - raw_cy) / 5000.0 * raw_cam_t[:, 2]

        K4 = [5000.0, 5000.0, raw_cx, raw_cy]

        skin_img = render_meshes_overlay_img(
            faces_all=m_skin["f"],
            verts_all=m_skin["v"].float(),
            cam_t_all=raw_cam_t,
            mesh_color="blue",
            img=img.copy(),
            K4=K4,
            view="front",
        )

        skel_img = render_meshes_overlay_img(
            faces_all=m_skel["f"],
            verts_all=m_skel["v"].float(),
            cam_t_all=raw_cam_t,
            mesh_color="human_yellow",
            img=img.copy(),
            K4=K4,
            view="front",
        )

        out_img = cv2.addWeighted(skin_img, 0.7, skel_img, 0.3, 0)
        cv2.imwrite(out_path, out_img)
        return out_path

    def render_cut_projection(self, image_path: str, out_path: str, mesh, pred_cam):
        from paper_render_utils import render_cut_mesh_overlay
        return render_cut_mesh_overlay(
            image_path=image_path,
            mesh=mesh,
            pred_cam=pred_cam,
            out_path=out_path,
            color=(0.15, 0.70, 1.00),
            alpha=0.80,
        )

    def render_paper_projections(self, image_path: str, out_dir: str, whole_mesh=None, cut_mesh=None, pred_cam=None):
        import os
        os.makedirs(out_dir, exist_ok=True)

        whole_path = os.path.join(out_dir, "paper_projection_whole.jpg")
        cut_path = os.path.join(out_dir, "paper_projection_cut.jpg")

        self.render_wholebody_projection(
            image_path=image_path,
            out_path=whole_path,
            mesh=whole_mesh,
            pred_cam=pred_cam,
        )

        if cut_mesh is not None:
            self.render_cut_projection(
                image_path=image_path,
                out_path=cut_path,
                mesh=cut_mesh,
                pred_cam=pred_cam,
            )

        return {
            "whole": whole_path,
            "cut": cut_path if cut_mesh is not None else None,
        }

def _safe_norm(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n

def _terminal_vertex_by_direction(vertices, anchor, direction, radius=None):
    """
    从 anchor 附近/全身 mesh 中，沿 direction 找投影最远的 vertex。
    用于近似手尖/脚尖 terminal landmark。
    """
    direction = _safe_norm(direction)
    if direction is None:
        return None

    verts = np.asarray(vertices, dtype=np.float32)
    anchor = np.asarray(anchor, dtype=np.float32)

    rel = verts - anchor[None, :]

    if radius is not None:
        dist = np.linalg.norm(rel, axis=1)
        mask = dist < radius
        if np.any(mask):
            cand = verts[mask]
            rel_cand = cand - anchor[None, :]
        else:
            cand = verts
            rel_cand = rel
    else:
        cand = verts
        rel_cand = rel

    scores = rel_cand @ direction
    return cand[int(np.argmax(scores))]