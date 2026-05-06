import sys
import os
import cv2
import numpy as np
import trimesh
import torch

# ==========================================
# 1. 暴力注入 Meta 仓库路径（解决瞎子问题）
# ==========================================
REPO_ROOT = '/home/sora/workspace/sam-3d-body'
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ==========================================
# 2. 从核心包直接导入 API（避开 utils 命名冲突）
# ==========================================
from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator


class ReconstructionEngine:
    def __init__(self, hf_repo_id="facebook/sam-3d-body-dinov3", device='cuda'):
        """
        初始化 SAM 3D Body 引擎
        """
        print(f">>> 正在加载 Meta SAM 3D Body 模型 ({hf_repo_id})...")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载核心模型
        model, model_cfg = load_sam_3d_body_hf(hf_repo_id, device=device)

        # 初始化 Estimator
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None
        )
        self.device = device
        print(">>> 🚀 SAM 3D Body 模型加载完成！")

    def _to_numpy(self, x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _build_pred_joints_dict(self, joints_3d):
        """
        按你现在的 MHR70 -> 语义关节点映射。
        """
        return {
            'nose': joints_3d[0],
            'left_eye': joints_3d[1],
            'right_eye': joints_3d[2],
            'left_ear': joints_3d[3],
            'right_ear': joints_3d[4],

            'left_shoulder': joints_3d[5],
            'right_shoulder': joints_3d[6],
            'left_elbow': joints_3d[7],
            'right_elbow': joints_3d[8],
            'left_wrist': joints_3d[62],
            'right_wrist': joints_3d[41],

            'left_hip': joints_3d[9],
            'right_hip': joints_3d[10],
            'left_knee': joints_3d[11],
            'right_knee': joints_3d[12],
            'left_ankle': joints_3d[13],
            'right_ankle': joints_3d[14],

            'L_Middle_Tip': joints_3d[50],
            'R_Middle_Tip': joints_3d[29],
            'L_Heel': joints_3d[17],
            'R_Heel': joints_3d[20],
            'L_Toe_Tip': joints_3d[15],
            'R_Toe_Tip': joints_3d[18],

            'pelvis': (joints_3d[9] + joints_3d[10]) / 2.0,
            'neck': (joints_3d[5] + joints_3d[6]) / 2.0,
        }

    def _project_bbox_from_joints(self, pred_joints_3d, pred_cam):
        """
        用 3D 关节点投影得到 bbox，用于和 GT annotation 匹配。
        """
        focal = pred_cam["focal"]
        princpt = pred_cam["princpt"]

        pts = []
        for name in [
            "nose", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]:
            if name not in pred_joints_3d:
                continue

            p = pred_joints_3d[name]
            if p[2] <= 1e-5:
                continue

            x = focal[0] * (p[0] / p[2]) + princpt[0]
            y = focal[1] * (p[1] / p[2]) + princpt[1]
            pts.append([x, y])

        if len(pts) == 0:
            return None

        pts = np.asarray(pts, dtype=np.float32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def predict_mesh_all(self, image_path: str, save_dir: str):
        """
        多人版：一张图检测到几个人，就返回几个人的 mesh / joints / camera。
        """
        os.makedirs(save_dir, exist_ok=True)

        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            raise ValueError(f"❌ 无法读取图像: {image_path}")

        height, width = img_cv2.shape[:2]

        outputs = self.estimator.process_one_image(image_path)
        if not outputs or len(outputs) == 0:
            raise ValueError("❌ 预测失败，未检测到人物。")

        faces = self.estimator.faces
        faces = self._to_numpy(faces)

        results = []

        print(f">>> SAM 3D Body detected persons: {len(outputs)}")

        for person_idx, person_data in enumerate(outputs):
            cam_t = self._to_numpy(person_data.get("pred_cam_t"))
            if cam_t is not None:
                cam_t = cam_t.reshape(-1)[0:3]

            vertices = self._to_numpy(person_data["pred_vertices"])
            vertices = np.squeeze(vertices)

            if cam_t is not None:
                vertices = vertices + cam_t

            mesh = trimesh.Trimesh(vertices, faces, process=False)

            mesh_save_path = os.path.join(save_dir, f"whole_body_mesh_person_{person_idx:02d}.obj")
            mesh.export(mesh_save_path)

            focal_length = float(person_data["focal_length"])
            pred_cam = {
                "focal": np.array([focal_length, focal_length]),
                "princpt": np.array([width / 2.0, height / 2.0]),
                "cam_t": cam_t,
            }

            joints_3d = self._to_numpy(person_data.get("pred_keypoints_3d"))
            if joints_3d is not None:
                joints_3d = np.squeeze(joints_3d)
                if cam_t is not None:
                    joints_3d = joints_3d + cam_t
                pred_joints_dict = self._build_pred_joints_dict(joints_3d)
            else:
                pred_joints_dict = {}

            pred_bbox = self._project_bbox_from_joints(pred_joints_dict, pred_cam)

            results.append({
                "person_idx": person_idx,
                "mesh_path": mesh_save_path,
                "mesh": mesh,
                "whole_mesh": mesh.copy(),
                "pred_joints_3d": pred_joints_dict,
                "pred_cam": pred_cam,
                "pred_bbox": pred_bbox,
                "raw_output": person_data,
            })

        return results

    def predict_mesh(self, image_path: str, save_path: str):
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            raise ValueError(f"❌ 无法读取图像: {image_path}")
        height, width = img_cv2.shape[:2]

        outputs = self.estimator.process_one_image(image_path)
        if not outputs or len(outputs) == 0:
            raise ValueError("❌ 预测失败，未检测到人物。")

        person_data = outputs[0]

        # ==========================================
        # 🚨 关键修复：提取相机平移参数 cam_t
        # ==========================================
        cam_t = person_data.get("pred_cam_t")
        if cam_t is not None and hasattr(cam_t, 'cpu'):
            cam_t = cam_t.detach().cpu().numpy()

        # 1. 提取 Mesh 顶点，并立刻加上 cam_t (将局部坐标转换到相机物理坐标)
        vertices = person_data["pred_vertices"]
        if hasattr(vertices, 'cpu'):
            vertices = vertices.detach().cpu().numpy()

        if cam_t is not None:
            vertices = vertices + cam_t  # 🌟 加上偏移！

        faces = self.estimator.faces
        if hasattr(faces, 'cpu'):
            faces = faces.detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(save_path)

        # 2. 提取并更新相机内参 (下游用不到 cam_t 了，因为已经加过了)
        focal_length = float(person_data["focal_length"])
        global_cam = {
            'focal': np.array([focal_length, focal_length]),
            'princpt': np.array([width / 2.0, height / 2.0]),
            'cam_t': cam_t
        }

        joints_3d = person_data.get("pred_keypoints_3d")
        if joints_3d is not None and hasattr(joints_3d, 'cpu'):
            joints_3d = joints_3d.detach().cpu().numpy()

        if joints_3d is not None and cam_t is not None:
            joints_3d = joints_3d + cam_t  # 🌟 加上偏移，转换到相机绝对坐标系

        pred_joints_dict = {}
        if joints_3d is not None:
            # 🌟 核心修复：SAM 3D Body 的 MHR70 格式，前 17 个点就是完美的 COCO 17！
            # 顺序和你代码里的 METAINFO 'keypoint_info' 做到 100% 绝对对齐。
            pred_joints_dict = {
                'nose': joints_3d[0],
                'left_eye': joints_3d[1],
                'right_eye': joints_3d[2],
                'left_ear': joints_3d[3],
                'right_ear': joints_3d[4],

                'left_shoulder': joints_3d[5],
                'right_shoulder': joints_3d[6],
                'left_elbow': joints_3d[7],
                'right_elbow': joints_3d[8],
                'left_wrist': joints_3d[62],
                'right_wrist': joints_3d[41],

                # 🚨 错位发生地：手腕不知道去哪了，9和10直接就是跨部！
                'left_hip': joints_3d[9],
                'right_hip': joints_3d[10],

                # 🚨 膝盖和脚踝全部提前了2个位置
                'left_knee': joints_3d[11],
                'right_knee': joints_3d[12],
                'left_ankle': joints_3d[13],
                'right_ankle': joints_3d[14],

                'L_Middle_Tip': joints_3d[50],
                'R_Middle_Tip': joints_3d[29],
                'L_Heel': joints_3d[17],
                'R_Heel': joints_3d[20],
                'L_Toe_Tip': joints_3d[15],  # or average of 15 and 16
                'R_Toe_Tip': joints_3d[18],  # or average of 18 and 19

                # 衍生点
                'pelvis': (joints_3d[9] + joints_3d[10]) / 2.0,
                'neck': (joints_3d[5] + joints_3d[6]) / 2.0
            }

        return save_path, pred_joints_dict, global_cam, mesh

    def render_wholebody_projection(self, image_path: str, out_path: str, mesh=None, pred_cam=None):
        """
        Whole-body projection using SAM 3D Body official visualization.
        The official visualization returns a 4-panel image, so we crop the mesh
        projection panel for paper use.
        """
        import cv2
        import numpy as np
        from tools.vis_utils import visualize_sample_together

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

        outputs = self.estimator.process_one_image(image_path)
        if outputs is None or len(outputs) == 0:
            raise ValueError("SAM 3D Body failed to detect a person.")

        rendered = visualize_sample_together(
            img_bgr,
            outputs,
            self.estimator.faces,
        )

        rendered = np.asarray(rendered).astype(np.uint8)

        # Optional: save the original 4-panel visualization for debugging
        full_out_path = out_path.replace(".jpg", "_full.jpg").replace(".png", "_full.png")
        cv2.imwrite(full_out_path, rendered)

        # SAM3D official visualization is usually:
        # [input | 2D/skeleton | mesh projection | standalone mesh]
        render_h, render_w = rendered.shape[:2]

        # Split into 4 equal panels and keep the 3rd one.
        panel_w = render_w // 4
        projection_panel = rendered[:, 2 * panel_w: 3 * panel_w]

        # Resize back to the original input image size if needed
        h, w = img_bgr.shape[:2]
        if projection_panel.shape[:2] != (h, w):
            projection_panel = cv2.resize(
                projection_panel,
                (w, h),
                interpolation=cv2.INTER_AREA,
            )

        cv2.imwrite(out_path, projection_panel)
        return out_path

    def render_cut_projection(self, image_path: str, out_path: str, mesh, pred_cam):
        from paper_render_utils import render_cut_mesh_overlay
        return render_cut_mesh_overlay(
            image_path=image_path,
            mesh=mesh,
            pred_cam=pred_cam,
            out_path=out_path,
            color=(0.78, 0.78, 0.78),
            alpha=1.0,
        )

    def render_paper_projections(
            self,
            image_path: str,
            out_dir: str,
            whole_mesh=None,
            cut_mesh=None,
            pred_cam=None,
            whole_mesh_cam_items=None,
            cut_mesh_cam_items=None,
    ):
        import os

        os.makedirs(out_dir, exist_ok=True)

        whole_path = os.path.join(out_dir, "paper_projection_whole.jpg")
        cut_path = os.path.join(out_dir, "paper_projection_cut.jpg")

        print(f"[PaperVis] out_dir = {out_dir}")
        print(f"[PaperVis] whole_mesh_cam_items = {0 if whole_mesh_cam_items is None else len(whole_mesh_cam_items)}")
        print(f"[PaperVis] cut_mesh_cam_items   = {0 if cut_mesh_cam_items is None else len(cut_mesh_cam_items)}")

        # ---------------------------------------------------------
        # Multi-person crop-based mode
        # ---------------------------------------------------------
        if whole_mesh_cam_items is not None or cut_mesh_cam_items is not None:
            whole_exists = False
            cut_exists = False

            if whole_mesh_cam_items is not None and len(whole_mesh_cam_items) > 0:
                print(f"[PaperVis] rendering WHOLE -> {whole_path}")
                self.render_multi_wholebody_projection(
                    image_path=image_path,
                    out_path=whole_path,
                    whole_mesh_cam_items=whole_mesh_cam_items,
                )
                whole_exists = os.path.exists(whole_path)
                print(f"[PaperVis] whole exists? {whole_exists}")

            if cut_mesh_cam_items is not None and len(cut_mesh_cam_items) > 0:
                print(f"[PaperVis] rendering CUT -> {cut_path}")
                self.render_multi_cut_projection(
                    image_path=image_path,
                    out_path=cut_path,
                    cut_mesh_cam_items=cut_mesh_cam_items,
                )
                cut_exists = os.path.exists(cut_path)
                print(f"[PaperVis] cut exists? {cut_exists}")

            if whole_mesh_cam_items is not None and len(whole_mesh_cam_items) > 0 and not whole_exists:
                raise RuntimeError(f"[PaperVis] whole projection was not saved: {whole_path}")

            if cut_mesh_cam_items is not None and len(cut_mesh_cam_items) > 0 and not cut_exists:
                raise RuntimeError(f"[PaperVis] cut projection was not saved: {cut_path}")

            return {
                "whole": whole_path if whole_exists else None,
                "cut": cut_path if cut_exists else None,
            }

        # ---------------------------------------------------------
        # Legacy single-person mode
        # ---------------------------------------------------------
        print("[PaperVis] legacy single-person mode")

        self.render_wholebody_projection(
            image_path=image_path,
            out_path=whole_path,
            mesh=whole_mesh,
            pred_cam=pred_cam,
        )

        if not os.path.exists(whole_path):
            raise RuntimeError(f"[PaperVis] whole projection was not saved: {whole_path}")

        cut_exists = False
        if cut_mesh is not None:
            self.render_cut_projection(
                image_path=image_path,
                out_path=cut_path,
                mesh=cut_mesh,
                pred_cam=pred_cam,
            )
            cut_exists = os.path.exists(cut_path)
            if not cut_exists:
                raise RuntimeError(f"[PaperVis] cut projection was not saved: {cut_path}")

        return {
            "whole": whole_path,
            "cut": cut_path if cut_exists else None,
        }

    def _render_multi_mesh_overlay(
            self,
            image_path: str,
            out_path: str,
            mesh_cam_items,
            color=(0.78, 0.78, 0.78),
            alpha=0.78,
    ):
        """
        Multi-person projection renderer.
        Each mesh uses its own camera intrinsics, then all masks are merged.

        mesh_cam_items:
            [
                {"mesh": mesh0, "pred_cam": cam0},
                {"mesh": mesh1, "pred_cam": cam1},
                ...
            ]
        """
        import cv2
        import numpy as np
        color = (0.58, 0.58, 0.58)
        alpha = 1
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h, w = img.shape[:2]
        merged_mask = np.zeros((h, w), dtype=np.uint8)

        for item_idx, item in enumerate(mesh_cam_items):
            mesh = item["mesh"]
            pred_cam = item["pred_cam"]

            if mesh is None or pred_cam is None:
                continue

            f = pred_cam["focal"]
            c = pred_cam["princpt"]

            K = np.array([
                [f[0], 0, c[0]],
                [0, f[1], c[1]],
                [0, 0, 1],
            ], dtype=np.float32)

            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)

            if len(vertices) == 0 or len(faces) == 0:
                continue

            pts_3d = vertices.T  # (3, N)
            pts_2d_homo = K @ pts_3d
            zs = pts_2d_homo[2, :]

            zs_clamped = np.maximum(zs, 1e-5)
            u = pts_2d_homo[0, :] / zs_clamped
            v = pts_2d_homo[1, :] / zs_clamped
            projected_pts = np.stack([u, v], axis=1).astype(np.int32)

            person_mask = np.zeros((h, w), dtype=np.uint8)

            for face in faces:
                # 只画在相机前方的面
                if np.all(zs[face] > 0.1):
                    pts = projected_pts[face].reshape((-1, 1, 2))
                    cv2.fillPoly(person_mask, [pts], 255)

            merged_mask = cv2.bitwise_or(merged_mask, person_mask)

        # 颜色转 0-255
        overlay_color = np.array(color, dtype=np.float32)
        if overlay_color.max() <= 1.0:
            overlay_color = (overlay_color * 255.0).astype(np.uint8)
        else:
            overlay_color = overlay_color.astype(np.uint8)

        overlay = img.copy()
        color_layer = np.zeros_like(img, dtype=np.uint8)
        color_layer[:] = overlay_color.tolist()

        cv2.addWeighted(color_layer, alpha, overlay, 1 - alpha, 0, dst=overlay)

        img_out = img.copy()
        mask_bool = merged_mask > 127
        img_out[mask_bool] = overlay[mask_bool]

        ok, buf = cv2.imencode(".jpg", img_out)
        if not ok:
            raise RuntimeError(f"Failed to encode image for saving: {out_path}")
        buf.tofile(out_path)

        if not os.path.exists(out_path):
            raise RuntimeError(f"Failed to save image: {out_path}")

        return out_path

    def render_multi_wholebody_projection(self, image_path: str, out_path: str, whole_mesh_cam_items):
        """
        Multi-person whole-body projection for paper.
        """
        print(f"[PaperVis] render_multi_wholebody_projection: persons={len(whole_mesh_cam_items)}, out={out_path}")

        return self._render_multi_mesh_overlay(
            image_path=image_path,
            out_path=out_path,
            mesh_cam_items=whole_mesh_cam_items,
            color=(0.78, 0.78, 0.78),
            alpha=0.78,
        )

    def render_multi_cut_projection(self, image_path: str, out_path: str, cut_mesh_cam_items):
        """
        Multi-person cut-mesh projection for paper.
        """
        print(f"[PaperVis] render_multi_cut_projection: persons={len(cut_mesh_cam_items)}, out={out_path}")

        return self._render_multi_mesh_overlay(
            image_path=image_path,
            out_path=out_path,
            mesh_cam_items=cut_mesh_cam_items,
            color=(0.78, 0.78, 0.78),
            alpha=0.78,
        )