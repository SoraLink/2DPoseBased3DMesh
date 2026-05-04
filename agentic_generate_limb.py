import json
import os
import re
from pathlib import Path

import cv2
import time

import dashscope
import numpy as np
import torch
from dashscope import MultiModalConversation
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message

from image_ops import ImageProcessor

from pose_extractor import PoseExtractor

_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else _original_load(*args, **kwargs)

dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

class LimbCompositingAgent:
    def __init__(self, pose_extractor, edit_model='wan2.7-image-pro', eval_model='qwen3.6-plus'):
        self.edit_model = edit_model
        self.pose_extractor = pose_extractor
        self.eval_model = eval_model

    def _to_int_tuple(self, p):
        return int(round(float(p[0]))), int(round(float(p[1])))

    def _metric_color(self, value, thr):
        # 绿: 很安全；黄: 接近阈值；红: 超阈值
        if value <= thr * 0.5:
            return (0, 180, 0)
        elif value <= thr:
            return (0, 215, 255)
        else:
            return (0, 0, 255)

    def _save_pose_displacement_vis(
            self,
            orig_image_path,
            gen_image_path,
            kpts_orig,
            kpts_gen,
            target_indices,
            scale,
            E_pose,
            pose_ok,
            save_path,
            conf_thr=None,
            pose_tau=None,
            kpt_types_orig=None,
    ):
        """
        可视化非目标关键点的位移。
        左侧：生成图上的 overlay（蓝色=原点，彩色=生成点，线段=位移）
        右侧：位移最大的 keypoints 列表
        """
        if conf_thr is None:
            conf_thr = getattr(self, "conf_thr", 0.10)
        if pose_tau is None:
            pose_tau = getattr(self, "pose_tau", 0.045)

        orig = cv2.imread(orig_image_path)
        gen = cv2.imread(gen_image_path)
        if orig is None or gen is None:
            return None

        # 保证同一坐标系
        if orig.shape[:2] != gen.shape[:2]:
            gen = cv2.resize(gen, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_AREA)

        vis = gen.copy()
        records = []

        num_kpts = min(len(kpts_orig), len(kpts_gen))
        for i in range(num_kpts):
            if i in target_indices:
                continue

            # Keep visualization consistent with E_pose computation.
            if kpt_types_orig is not None and i < len(kpt_types_orig) and kpt_types_orig[i] in [1, 2]:
                continue

            p_orig = self._safe_get_point(kpts_orig, i, conf_thr=conf_thr)
            p_gen = self._safe_get_point(kpts_gen, i, conf_thr=conf_thr)
            if p_orig is None or p_gen is None:
                continue

            dist_norm = float(np.linalg.norm(p_gen - p_orig) / max(scale, 1e-6))
            color = self._metric_color(dist_norm, pose_tau)

            p1 = self._to_int_tuple(p_orig)
            p2 = self._to_int_tuple(p_gen)

            cv2.circle(vis, p1, 3, (255, 0, 0), -1)  # 原图点：蓝
            cv2.circle(vis, p2, 4, color, -1)  # 生成图点：按阈值上色
            cv2.line(vis, p1, p2, color, 1)

            # 在图上只标注最大的几个，避免太乱
            records.append((i, dist_norm, p1, p2))

        records_sorted = sorted(records, key=lambda x: x[1], reverse=True)

        # 给 top-k 最大位移点加编号标注
        for rank, (idx, dist_norm, p1, p2) in enumerate(records_sorted[:12]):
            color = self._metric_color(dist_norm, pose_tau)
            cv2.putText(
                vis,
                f"{idx}:{dist_norm:.3f}",
                (p2[0] + 4, p2[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

        h, w = vis.shape[:2]
        panel_w = 420
        panel = np.ones((h, panel_w, 3), dtype=np.uint8) * 255

        y = 28
        cv2.putText(panel, "Pose Displacement Debug", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
        y += 32
        cv2.putText(panel, f"E_pose = {E_pose:.4f}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
        y += 28
        cv2.putText(panel, f"tau_pose = {pose_tau:.4f}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 1, cv2.LINE_AA)
        y += 28
        cv2.putText(panel, f"passed = {pose_ok}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (0, 150, 0) if pose_ok else (0, 0, 255), 2, cv2.LINE_AA)
        y += 32
        cv2.putText(panel, "Top keypoint displacements:", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 0, 0), 1, cv2.LINE_AA)
        y += 22

        for idx, dist_norm, _, _ in records_sorted[:20]:
            color = self._metric_color(dist_norm, pose_tau)
            cv2.putText(panel, f"kpt {idx:02d}: {dist_norm:.4f}", (16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
            y += 22
            if y > h - 16:
                break

        # legend
        cv2.putText(panel, "Legend:", (12, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, "blue = original keypoint", (16, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, "green/yellow/red = gen keypoint", (16, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)

        canvas = np.hstack([vis, panel])
        cv2.imwrite(save_path, canvas)
        return save_path

    def _save_direction_projection_vis(
            self,
            orig_image_path,
            gen_image_path,
            kpts_orig,
            kpts_gen,
            rules,
            direction_checks,
            scale,
            save_path,
            conf_thr=None,
            perp_tau=None,
    ):
        """
        可视化残肢方向检查。
        左图：原图上的 residual segment（anchor -> residual endpoint）
        右图：生成图上的参考方向、生成肢段、投影点、垂线
        右侧面板：alpha / d_perp / angle(诊断用)
        """
        if conf_thr is None:
            conf_thr = getattr(self, "conf_thr", 0.10)
        if perp_tau is None:
            perp_tau = getattr(self, "perp_tau", 0.15)

        orig = cv2.imread(orig_image_path)
        gen = cv2.imread(gen_image_path)
        if orig is None or gen is None:
            return None

        if orig.shape[:2] != gen.shape[:2]:
            gen = cv2.resize(gen, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_AREA)

        left = orig.copy()
        right = gen.copy()

        # 标题
        cv2.putText(left, "Original residual segment", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right, "Generated segment / projection", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

        # 根据 res_idx 快速查找检查结果
        check_map = {c["res_idx"]: c for c in direction_checks}

        summary_rows = []

        for r in rules:
            res_idx = r["res_idx"]
            anchor_idx = r["anchor_idx"]
            downstream_idx = r["downstream_idx"]

            p_a = self._safe_get_point(kpts_orig, anchor_idx, conf_thr=conf_thr)
            p_r = self._safe_get_point(kpts_orig, res_idx, conf_thr=conf_thr)

            p_a_hat = self._safe_get_point(kpts_gen, anchor_idx, conf_thr=conf_thr)
            p_d_hat = self._safe_get_point(kpts_gen, downstream_idx, conf_thr=conf_thr)

            check = check_map.get(res_idx, None)

            if p_a is None or p_r is None or p_a_hat is None or p_d_hat is None:
                summary_rows.append({
                    "res_idx": res_idx,
                    "alpha": None,
                    "d_perp": None,
                    "angle_deg": None,
                    "passed": False,
                    "reason": "missing keypoints"
                })
                continue

            # 左图：原始残肢向量
            p_a_i = self._to_int_tuple(p_a)
            p_r_i = self._to_int_tuple(p_r)
            cv2.circle(left, p_a_i, 4, (255, 0, 0), -1)
            cv2.circle(left, p_r_i, 4, (255, 0, 0), -1)
            cv2.arrowedLine(left, p_a_i, p_r_i, (255, 0, 0), 2, tipLength=0.12)
            cv2.putText(left, f"res {res_idx}", (p_r_i[0] + 4, p_r_i[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

            # 右图：生成肢段及其投影
            v_res = p_r - p_a
            q_gen = p_d_hat - p_a_hat

            norm_v = float(np.linalg.norm(v_res))
            norm_q = float(np.linalg.norm(q_gen))
            if norm_v < 1e-6 or norm_q < 1e-6:
                summary_rows.append({
                    "res_idx": res_idx,
                    "alpha": None,
                    "d_perp": None,
                    "angle_deg": None,
                    "passed": False,
                    "reason": "degenerate vector"
                })
                continue

            u = v_res / norm_v
            proj_len = float(np.dot(q_gen, u))
            proj_pt = p_a_hat + proj_len * u
            ref_end = p_a_hat + max(norm_q, 40.0) * u

            # 诊断角度（不参与 pass/fail）
            cos_theta = float(np.dot(q_gen, v_res) / max(norm_q * norm_v, 1e-6))
            cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_theta)))

            p_a_hat_i = self._to_int_tuple(p_a_hat)
            p_d_hat_i = self._to_int_tuple(p_d_hat)
            proj_pt_i = self._to_int_tuple(proj_pt)
            ref_end_i = self._to_int_tuple(ref_end)

            # 蓝：参考 residual direction
            cv2.circle(right, p_a_hat_i, 4, (255, 0, 0), -1)
            cv2.arrowedLine(right, p_a_hat_i, ref_end_i, (255, 0, 0), 2, tipLength=0.12)

            # 绿：生成肢段
            cv2.circle(right, p_d_hat_i, 4, (0, 255, 0), -1)
            cv2.arrowedLine(right, p_a_hat_i, p_d_hat_i, (0, 255, 0), 2, tipLength=0.12)

            # 青：投影点
            cv2.circle(right, proj_pt_i, 4, (255, 255, 0), -1)

            # 红：垂线
            cv2.line(right, proj_pt_i, p_d_hat_i, (0, 0, 255), 2)

            # 文字
            label = f"res {res_idx}"
            cv2.putText(right, label, (p_d_hat_i[0] + 4, p_d_hat_i[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            alpha = check["alpha"] if check is not None else None
            d_perp = check["d_perp"] if check is not None else None
            passed = check["passed"] if check is not None else False

            summary_rows.append({
                "res_idx": res_idx,
                "alpha": alpha,
                "d_perp": d_perp,
                "angle_deg": angle_deg,
                "passed": passed,
                "reason": "" if check is None else check["reason"],
            })

        h, w = left.shape[:2]
        middle = np.hstack([left, right])

        panel_w = 460
        panel = np.ones((h, panel_w, 3), dtype=np.uint8) * 255

        y = 28
        cv2.putText(panel, "Direction / Projection Debug", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
        y += 32
        cv2.putText(panel, f"tau_perp = {perp_tau:.4f}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 1, cv2.LINE_AA)
        y += 28
        cv2.putText(panel, "angle is diagnostic only", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 80, 80), 1, cv2.LINE_AA)
        y += 30

        for row in summary_rows:
            passed = row["passed"]
            color = (0, 150, 0) if passed else (0, 0, 255)

            cv2.putText(panel, f"res {row['res_idx']}: passed={passed}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
            y += 22

            alpha_txt = "None" if row["alpha"] is None else f"{row['alpha']:.4f}"
            dperp_txt = "None" if row["d_perp"] is None else f"{row['d_perp']:.4f}"
            angle_txt = "None" if row["angle_deg"] is None else f"{row['angle_deg']:.2f}"

            cv2.putText(panel, f"alpha  = {alpha_txt}", (26, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
            y += 20
            cv2.putText(panel, f"d_perp = {dperp_txt}", (26, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        self._metric_color(row["d_perp"] if row["d_perp"] is not None else 999, perp_tau),
                        1, cv2.LINE_AA)
            y += 20
            cv2.putText(panel, f"angle  = {angle_txt} deg", (26, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
            y += 28

            if y > h - 40:
                break

        cv2.putText(panel, "Blue  = residual reference direction", (12, h - 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, "Green = generated limb segment", (12, h - 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 150, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, "Red   = perpendicular deviation", (12, h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "Cyan  = projection point", (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 0), 1, cv2.LINE_AA)

        canvas = np.hstack([middle, panel])
        cv2.imwrite(save_path, canvas)
        return save_path

    def _resize_generated_to_original(self, gen_image_path, orig_image_path):
        orig = cv2.imread(orig_image_path)
        gen = cv2.imread(gen_image_path)

        if orig is None:
            raise ValueError(f"Cannot read original image: {orig_image_path}")
        if gen is None:
            raise ValueError(f"Cannot read generated image: {gen_image_path}")

        h, w = orig.shape[:2]
        gh, gw = gen.shape[:2]

        if (gh, gw) == (h, w):
            return gen_image_path

        gen_resized = cv2.resize(gen, (w, h), interpolation=cv2.INTER_AREA)
        base, ext = os.path.splitext(gen_image_path)
        resized_path = f"{base}_resized_to_orig{ext}"
        cv2.imwrite(resized_path, gen_resized)

        return resized_path

    def _get_residual_eval_rules(self, keypoint_types):
        """
        Build residual-limb evaluation rules for the pose-based geometric critic.

        keypoint_types[res_idx] == 0 means this residual endpoint exists.
        keypoint_types[res_idx] == 2 means absent.
        """
        mapping_dict = {
            # res_idx: (anchor_idx, first_downstream_idx, exclude_indices)

            # Upper limbs
            23: (5, 7, [7, 9, 17]),  # left upper-arm residual
            24: (6, 8, [8, 10, 18]),  # right upper-arm residual
            25: (7, 9, [9, 17]),  # left forearm residual
            26: (8, 10, [10, 18]),  # right forearm residual

            # Lower limbs
            27: (11, 13, [13, 15, 19, 21]),  # left thigh residual
            28: (12, 14, [14, 16, 20, 22]),  # right thigh residual
            29: (13, 15, [15, 19, 21]),  # left lower-leg residual
            30: (14, 16, [16, 20, 22]),  # right lower-leg residual
        }

        rules = []
        for res_idx, (anchor_idx, downstream_idx, exclude_indices) in mapping_dict.items():
            if res_idx >= len(keypoint_types):
                continue

            # Only existing residual endpoints should be evaluated.
            if keypoint_types[res_idx] != 0:
                continue

            rules.append({
                "res_idx": res_idx,
                "anchor_idx": anchor_idx,
                "downstream_idx": downstream_idx,  # 用于 direction check
                "exclude_indices": exclude_indices,  # 用于 E_pose 和可视化排除
            })

        return rules

    def _bbox_scale_from_keypoints(self, kpts, conf_thr=None):
        """
        Compute subject scale as sqrt(bounding-box area) from visible keypoints.
        This matches the scale-normalization idea used in the paper.
        """
        if conf_thr is None:
            conf_thr = getattr(self, "conf_thr", 0.10)

        kpts = np.asarray(kpts)
        visible = kpts[kpts[:, 2] > conf_thr]

        if len(visible) < 2:
            return 1.0

        min_xy = np.min(visible[:, :2], axis=0)
        max_xy = np.max(visible[:, :2], axis=0)

        w, h = max_xy - min_xy
        area = max(float(w * h), 1.0)

        return float(np.sqrt(area))

    def _safe_get_point(self, kpts, idx, conf_thr=None):
        """
        Safely fetch a visible 2D keypoint.
        Return None if the keypoint is missing or low-confidence.
        """
        if conf_thr is None:
            conf_thr = getattr(self, "conf_thr", 0.10)

        if idx >= len(kpts):
            return None

        if kpts[idx][2] <= conf_thr:
            return None

        return kpts[idx, :2].astype(np.float32)

    def _gen_conf_color(self, conf):
        """
        Color for generated pose keypoints based on confidence.
        high conf -> green, medium -> yellow, low -> red
        """
        if conf >= 0.5:
            return (0, 180, 0)
        elif conf >= 0.2:
            return (0, 215, 255)
        else:
            return (0, 0, 255)

    def _orig_kpt_color(self, idx, kpt_types=None):
        """
        Color for original annotated keypoints.
        - normal body keypoints: green
        - prosthetic-related points (type=1): orange
        - residual endpoint keypoints 23-30 with type=0: magenta
        """
        if kpt_types is not None and idx < len(kpt_types):
            t = kpt_types[idx]
            if t == 1:
                return (0, 165, 255)  # orange
            if idx >= 23 and t == 0:
                return (255, 0, 255)  # magenta
        if idx >= 23:
            return (255, 0, 255)
        return (0, 180, 0)

    def _is_valid_orig_draw_kpt(self, kpts, idx, kpt_types=None):
        """
        Decide whether an original annotated keypoint should be drawn.
        """
        if idx >= len(kpts):
            return False

        # third dim should exist and be > 0
        if kpts[idx][2] <= 0:
            return False

        if kpt_types is not None and idx < len(kpt_types):
            # type=2 means absent
            if kpt_types[idx] == 2:
                return False
            # for residual endpoints 23-30, only draw if type==0 (exists)
            if idx >= 23 and kpt_types[idx] != 0:
                return False

        return True

    def _is_valid_gen_draw_kpt(self, kpts, idx, conf_thr=None):
        """
        Decide whether a generated-image predicted keypoint should be drawn.
        """
        if conf_thr is None:
            conf_thr = getattr(self, "conf_thr", 0.10)

        if idx >= len(kpts):
            return False

        return kpts[idx][2] > conf_thr

    def _draw_pose_overlay(
            self,
            image,
            kpts,
            mode="gen",
            kpt_types=None,
            title=None,
            conf_thr=None,
            show_indices=True,
    ):
        """
        Draw pose keypoints and skeleton on one image.

        mode:
            - "orig": draw original annotation
            - "gen": draw generated pose estimation
        """
        if conf_thr is None:
            conf_thr = getattr(self, "conf_thr", 0.10)

        canvas = image.copy()

        # Standard body skeleton edges (main 17-body structure)
        skeleton_edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6),
            (5, 7), (7, 9),
            (6, 8), (8, 10),
            (5, 11), (6, 12),
            (11, 12),
            (11, 13), (13, 15),
            (12, 14), (14, 16),
        ]

        def is_valid(idx):
            if mode == "orig":
                return self._is_valid_orig_draw_kpt(kpts, idx, kpt_types=kpt_types)
            else:
                return self._is_valid_gen_draw_kpt(kpts, idx, conf_thr=conf_thr)

        def get_color(idx):
            if mode == "orig":
                return self._orig_kpt_color(idx, kpt_types=kpt_types)
            else:
                conf = float(kpts[idx][2]) if idx < len(kpts) else 0.0
                return self._gen_conf_color(conf)

        # draw skeleton for standard 0-16 body keypoints
        for a, b in skeleton_edges:
            if a >= len(kpts) or b >= len(kpts):
                continue
            if not is_valid(a) or not is_valid(b):
                continue

            pa = self._to_int_tuple(kpts[a, :2])
            pb = self._to_int_tuple(kpts[b, :2])

            if mode == "orig":
                edge_color = (180, 180, 180)
            else:
                edge_color = (120, 220, 120)

            cv2.line(canvas, pa, pb, edge_color, 2)

        # draw all valid keypoints
        for idx in range(len(kpts)):
            if not is_valid(idx):
                continue

            p = self._to_int_tuple(kpts[idx, :2])
            color = get_color(idx)

            cv2.circle(canvas, p, 4, color, -1)

            if show_indices:
                if mode == "gen":
                    conf = float(kpts[idx][2])
                    label = f"{idx}:{conf:.2f}"
                else:
                    label = f"{idx}"

                cv2.putText(
                    canvas,
                    label,
                    (p[0] + 4, p[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        if title is not None:
            cv2.putText(
                canvas,
                title,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return canvas

    def _save_pose_estimation_comparison_vis(
            self,
            orig_image_path,
            gen_image_path,
            kpts_orig,
            kpts_gen,
            save_path,
            kpt_types_orig=None,
            conf_thr=None,
            show_indices=True,
    ):
        """
        Save one comparison image:
        - left: original image + original annotation
        - right: generated image + pose estimation result
        """
        if conf_thr is None:
            conf_thr = getattr(self, "conf_thr", 0.10)

        orig = cv2.imread(orig_image_path)
        gen = cv2.imread(gen_image_path)

        if orig is None:
            raise ValueError(f"Cannot read original image: {orig_image_path}")
        if gen is None:
            raise ValueError(f"Cannot read generated image: {gen_image_path}")

        # keep both panels in the same size
        if orig.shape[:2] != gen.shape[:2]:
            gen = cv2.resize(gen, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_AREA)

        left = self._draw_pose_overlay(
            image=orig,
            kpts=np.asarray(kpts_orig, dtype=np.float32),
            mode="orig",
            kpt_types=kpt_types_orig,
            title="Original annotation",
            conf_thr=conf_thr,
            show_indices=show_indices,
        )

        right = self._draw_pose_overlay(
            image=gen,
            kpts=np.asarray(kpts_gen, dtype=np.float32),
            mode="gen",
            kpt_types=None,
            title="Generated pose estimation",
            conf_thr=conf_thr,
            show_indices=show_indices,
        )

        h = left.shape[0]
        panel_w = 420
        panel = np.ones((h, panel_w, 3), dtype=np.uint8) * 255

        # summary stats
        orig_valid = 0
        for i in range(len(kpts_orig)):
            if self._is_valid_orig_draw_kpt(kpts_orig, i, kpt_types=kpt_types_orig):
                orig_valid += 1

        gen_valid = 0
        low_conf = []
        for i in range(len(kpts_gen)):
            if self._is_valid_gen_draw_kpt(kpts_gen, i, conf_thr=conf_thr):
                gen_valid += 1
                conf = float(kpts_gen[i][2])
                low_conf.append((i, conf))

        low_conf = sorted(low_conf, key=lambda x: x[1])

        y = 28
        cv2.putText(panel, "Pose Estimation Debug", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)
        y += 34
        cv2.putText(panel, f"orig valid kpts = {orig_valid}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24
        cv2.putText(panel, f"gen valid kpts  = {gen_valid}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24
        cv2.putText(panel, f"conf_thr = {conf_thr:.2f}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        y += 34
        cv2.putText(panel, "Lowest-confidence generated kpts:", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        y += 22

        for idx, conf in low_conf[:20]:
            color = self._gen_conf_color(conf)
            cv2.putText(panel, f"kpt {idx:02d}: {conf:.3f}", (16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            y += 20
            if y > h - 90:
                break

        # legend
        cv2.putText(panel, "Legend:", (12, h - 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, "Left  = original annotation", (16, h - 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, "Right = generated pose estimation", (16, h - 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, "Green/Yellow/Red = gen conf high/mid/low", (16, h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, "Magenta = residual endpoints in annotation", (16, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)

        canvas = np.hstack([left, right, panel])
        cv2.imwrite(save_path, canvas)
        return save_path

    def evaluate_pose_geometric(
            self,
            kpts_orig,
            kpts_gen,
            kpt_types_orig,
            orig_image_path=None,
            gen_image_path=None,
            vis_dir=None,
            vis_prefix="pose_eval",
    ):
        """
        Pose-based geometric critic.

        It validates two properties:

        1. Body preservation:
           Non-target visible keypoints should not move too much after image editing.

        2. Residual-limb direction/path consistency:
           The generated limb segment adjacent to the residual region should extend
           along the original residual-limb direction. Instead of using angular error,
           we compute the perpendicular deviation from the original residual-limb ray.

        If image paths + vis_dir are provided, two debug visualizations will be saved:
        - pose displacement debug
        - direction / projection debug
        """
        pose_tau = getattr(self, "pose_tau", 0.045)
        perp_tau = getattr(self, "perp_tau", 0.15)
        conf_thr = getattr(self, "conf_thr", 0.10)

        kpts_orig = np.asarray(kpts_orig, dtype=np.float32)
        kpts_gen = np.asarray(kpts_gen, dtype=np.float32)

        rules = self._get_residual_eval_rules(kpt_types_orig)

        if not rules:
            return {
                "passed": True,
                "reason": "No residual/prosthetic target found for pose evaluation.",
                "E_pose": 0.0,
                "pose_ok": True,
                "direction_ok": True,
                "direction_checks": [],
                "pose_vis_path": None,
                "direction_vis_path": None,
            }

        scale = self._bbox_scale_from_keypoints(kpts_orig, conf_thr=conf_thr)

        # ------------------------------------------------------------------
        # 1. Body-preservation error
        # ------------------------------------------------------------------
        target_indices = set(range(23, 31))
        for r in rules:
            target_indices.update(r.get("exclude_indices", []))
            target_indices.add(r["res_idx"])
            target_indices.add(r["downstream_idx"])

        keep_errors = []

        num_kpts = min(len(kpts_orig), len(kpts_gen))
        for i in range(num_kpts):
            if i in target_indices:
                continue

            # type=1: prosthetic point, type=2: absent point
            # Both should not be used for body-preservation error.
            if i < len(kpt_types_orig) and kpt_types_orig[i] in [1, 2]:
                continue

            p_orig = self._safe_get_point(kpts_orig, i, conf_thr=conf_thr)
            p_gen = self._safe_get_point(kpts_gen, i, conf_thr=conf_thr)

            if p_orig is None or p_gen is None:
                continue

            dist = np.linalg.norm(p_gen - p_orig)
            keep_errors.append(dist / scale)

        E_pose = float(np.mean(keep_errors)) if keep_errors else 0.0
        pose_ok = E_pose <= pose_tau

        # ------------------------------------------------------------------
        # 2. Residual-limb direction/path consistency
        # ------------------------------------------------------------------
        direction_checks = []
        direction_ok = True

        for r in rules:
            res_idx = r["res_idx"]
            anchor_idx = r["anchor_idx"]
            downstream_idx = r["downstream_idx"]

            p_a = self._safe_get_point(kpts_orig, anchor_idx, conf_thr=conf_thr)
            p_r = self._safe_get_point(kpts_orig, res_idx, conf_thr=conf_thr)

            p_a_hat = self._safe_get_point(kpts_gen, anchor_idx, conf_thr=conf_thr)
            p_d_hat = self._safe_get_point(kpts_gen, downstream_idx, conf_thr=conf_thr)

            check = {
                "res_idx": res_idx,
                "anchor_idx": anchor_idx,
                "downstream_idx": downstream_idx,
                "passed": False,
                "alpha": None,
                "d_perp": None,
                "reason": ""
            }

            if p_a is None:
                check["reason"] = f"Original anchor keypoint {anchor_idx} is missing."
                direction_ok = False
                direction_checks.append(check)
                continue

            if p_r is None:
                check["reason"] = f"Original residual endpoint {res_idx} is missing."
                direction_ok = False
                direction_checks.append(check)
                continue

            if p_a_hat is None:
                check["reason"] = f"Generated anchor keypoint {anchor_idx} is missing."
                direction_ok = False
                direction_checks.append(check)
                continue

            if p_d_hat is None:
                check["reason"] = f"Generated downstream keypoint {downstream_idx} is missing."
                direction_ok = False
                direction_checks.append(check)
                continue

            v_res = p_r - p_a
            q_gen = p_d_hat - p_a_hat

            denom = float(np.dot(v_res, v_res))
            if denom < 1e-6:
                check["reason"] = "Original residual-limb vector is too short or degenerate."
                direction_ok = False
                direction_checks.append(check)
                continue

            alpha = float(np.dot(q_gen, v_res) / denom)

            perp_vec = q_gen - alpha * v_res
            d_perp = float(np.linalg.norm(perp_vec) / scale)

            check["alpha"] = alpha
            check["d_perp"] = d_perp

            if alpha > 0 and d_perp < perp_tau:
                check["passed"] = True
                check["reason"] = (
                    f"Passed: alpha={alpha:.4f}, d_perp={d_perp:.4f}."
                )
            else:
                check["passed"] = False
                check["reason"] = (
                    f"Failed: alpha={alpha:.4f}, d_perp={d_perp:.4f}, "
                    f"required alpha > 0 and d_perp < {perp_tau:.4f}."
                )
                direction_ok = False

            direction_checks.append(check)

        passed = pose_ok and direction_ok

        pose_vis_path = None
        direction_vis_path = None

        if orig_image_path is not None and gen_image_path is not None and vis_dir is not None:
            try:
                os.makedirs(vis_dir, exist_ok=True)

                pose_vis_path = os.path.join(vis_dir, f"{vis_prefix}_pose_vis.png")
                direction_vis_path = os.path.join(vis_dir, f"{vis_prefix}_direction_vis.png")

                self._save_pose_displacement_vis(
                    orig_image_path=orig_image_path,
                    gen_image_path=gen_image_path,
                    kpts_orig=kpts_orig,
                    kpts_gen=kpts_gen,
                    target_indices=target_indices,
                    scale=scale,
                    E_pose=E_pose,
                    pose_ok=pose_ok,
                    save_path=pose_vis_path,
                    conf_thr=conf_thr,
                    pose_tau=pose_tau,
                    kpt_types_orig=kpt_types_orig,
                )

                self._save_direction_projection_vis(
                    orig_image_path=orig_image_path,
                    gen_image_path=gen_image_path,
                    kpts_orig=kpts_orig,
                    kpts_gen=kpts_gen,
                    rules=rules,
                    direction_checks=direction_checks,
                    scale=scale,
                    save_path=direction_vis_path,
                    conf_thr=conf_thr,
                    perp_tau=perp_tau,
                )

            except Exception as e:
                print(f"⚠️ Visualization saving failed: {e}")
                pose_vis_path = None
                direction_vis_path = None

        return {
            "passed": passed,
            "reason": (
                f"E_pose={E_pose:.4f}, pose_ok={pose_ok}, "
                f"direction_ok={direction_ok}"
            ),
            "E_pose": E_pose,
            "pose_ok": pose_ok,
            "direction_ok": direction_ok,
            "direction_checks": direction_checks,
            "pose_vis_path": pose_vis_path,
            "direction_vis_path": direction_vis_path,
        }
    def generate_prompt(self, image_path, kpts, kpt_types):
        """
        评估图片是否存在残肢。
        基于确定的缺陷类型，同时生成：
        1. 给修复模型的精准 Master Prompt
        2. 给复查裁判的专属 VLM Instruction (绝不含糊其辞)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片用于 Eval: {image_path}")
        height, width = img.shape[:2]
        fallback_center_x = width / 2.0

        def get_valid_x(kpt_id):
            if kpt_id < len(kpt_types) and kpts[kpt_id][2] > 0:
                return kpts[kpt_id][0]
            return None

        # 计算身体中线用于判断左右
        ls_x, rs_x = get_valid_x(5), get_valid_x(6)
        upper_xs = [x for x in (ls_x, rs_x) if x is not None]
        upper_center_x = sum(upper_xs) / len(upper_xs) if upper_xs else fallback_center_x

        lh_x, rh_x = get_valid_x(11), get_valid_x(12)
        lower_xs = [x for x in (lh_x, rh_x) if x is not None]
        lower_center_x = sum(lower_xs) / len(lower_xs) if lower_xs else fallback_center_x

        res_mapping = {
            23: (17, '手臂', 'upper'),
            24: (18, '手臂', 'upper'),
            27: (21, '腿部', 'lower'),
            28: (22, '腿部', 'lower'),
            25: (17, '手臂', 'upper'),
            26: (18, '手臂', 'upper'),
            29: (21, '腿部', 'lower'),
            30: (22, '腿部', 'lower')
        }

        defects_found = []
        vlm_checks = []  # 专门存放精准的验收标准
        defect_parts_set = set()

        for res_id, (downstream_id, limb_name, body_part_type) in res_mapping.items():
            if res_id < len(kpt_types) and kpt_types[res_id] == 0:
                res_x = kpts[res_id][0]
                ref_center_x = upper_center_x if body_part_type == 'upper' else lower_center_x
                screen_side = "画面左侧" if res_x < ref_center_x else "画面右侧"
                defect_key = f"{screen_side}的{limb_name}"

                if defect_key not in defect_parts_set:
                    defect_parts_set.add(defect_key)

                    downstream_type = kpt_types[downstream_id] if downstream_id < len(kpt_types) else 2

                    # 动态明确末端名称，杜绝“手/脚”
                    end_part = "手" if limb_name == "手臂" else "脚"

                    if downstream_type == 1:
                        # 明确已知是假肢
                        defects_found.append(
                            f"将{defect_key}的假肢/义肢完全替换为具有完美解剖结构的真实正常肢体（包含末端的{end_part}）"
                        )
                        vlm_checks.append(
                            f"- 【{defect_key}】：只进行宽松的 HMR 可用性检查，而不是照片真实感检查。"
                            f"请判断生成出的生物肢体是否具有大致合理的长度、粗细和连续的肢体结构，"
                            f"是否能够作为 3D HMR 模型的输入。"
                            f"只有在出现严重问题时才判为不合格，例如：肢体长度明显过长或过短、粗细明显异常、"
                            f"肢体断裂、严重扭曲、关节连接明显错误、多生成肢体，或主要肢段缺失。"
                            f"不要因为肤色不完全一致、鞋子/赤脚不一致、衣物纹理不一致、轻微 AI 生成痕迹、"
                            f"手指或脚趾细节不清晰而判为不合格。"
                        )
                    else:
                        # 明确已知是缺失断肢
                        defects_found.append(
                            f"沿着{defect_key}的残肢端点方向自然延伸，补全为具有完美解剖结构的完整{limb_name}（包含末端的{end_part}）。"
                            f"【强烈注意】：绝对不要对{defect_key}现有的残肢部分进行任何像素改变、不要改变其原有的位置和方向"
                        )
                        vlm_checks.append(
                            f"- 【{defect_key}】：只进行宽松的 HMR 可用性检查，而不是照片真实感检查。"
                            f"请判断沿残肢补全出的肢体是否具有大致合理的长度、粗细和连续的肢体结构，"
                            f"是否能够作为 3D HMR 模型的输入。"
                            f"只有在出现严重问题时才判为不合格，例如：肢体长度明显过长或过短、粗细明显异常、"
                            f"肢体断裂、严重扭曲、关节连接明显错误、多生成肢体，或主要肢段缺失。"
                            f"不要因为肤色不完全一致、鞋子/赤脚不一致、衣物纹理不一致、轻微 AI 生成痕迹、"
                            f"手指或脚趾细节不清晰而判为不合格。"
                        )

        if not defects_found:
            return {
                "passed": True,
                "reason": "未检测到生物性残肢断点，无需补全。",
                "prompt": "",
                "eval_instruction": ""
            }
        else:
            defect_desc = "；并且，".join(defects_found)
            repair_prompt = (
                f"当前图片存在需要修改的肢体。请严格执行以下局部修改指令：\n{defect_desc}。\n"
                f"【全局绝对指令】：只允许对上述明确指定的缺陷部位进行处理！绝对不要移动、修改或重绘任何现有的健全肢体、躯干、脸部以及背景环境！必须保持原图人物主体动作和画风完全不变。"
            )

            # 将精准的验收标准组装给大模型
            vlm_checks_str = "\n".join(vlm_checks)
            eval_instruction = (
                f"[任务: HMR 代理图像可用性评估]\n"
                f"请严格按照以下标准，只检查图片中指定目标肢体区域是否足以作为 3D human mesh recovery "
                f"(HMR) 的 proxy image 使用，而不是判断它是否达到照片级真实感：\n"
                f"{vlm_checks_str}\n\n"
                f"评估原则：\n"
                f"1. 只关注生成肢体的整体可用性，包括长度、粗细、主要肢段结构和连接是否大致合理。\n"
                f"2. 如果生成肢体整体形态合理，即使存在轻微纹理不自然、肤色不一致、鞋子/赤脚不一致、"
                f"衣物细节不一致、手指或脚趾不清晰，也应判为通过。\n"
                f"3. 只有当生成结果会明显干扰 HMR，例如肢体严重畸形、长度或粗细极端异常、"
                f"肢体断裂、主要肢段缺失、多生成肢体或连接到错误位置时，才判为不通过。\n"
                f"4. 不要检查人物整体位置、姿态方向或关键点对齐，这些由几何评估器完成。\n\n"
                f"输出必须严格按照JSON格式:\n"
                f"{{\n"
                f"    \"passed\": true/false,\n"
                f"    \"reason\": \"简要说明目标肢体是否足以用于HMR。如果不通过，只说明严重的长度、粗细、结构或连接问题。\"\n"
                f"}}"
            )

            return {
                "passed": False,
                "reason": f"检测到异常部位：{'、'.join(defect_parts_set)}",
                "prompt": repair_prompt,
                "eval_instruction": eval_instruction
            }

    def evaluate_image(self, image_path, prompt):
        print(f"👁️ [复查] 召唤视觉大模型 {self.eval_model} 担任裁判...")

        # 1. 编码图片 (由于 DashScope 多模态支持 file:// 协议，如果 ImageProcessor 返回的是 base64 也可以直接用)
        image_encoded = ImageProcessor.encode_file(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_encoded},
                    {"text": prompt}
                ]
            }
        ]

        api_attempt = 0
        while api_attempt < 3:
            try:
                # 调用多模态视觉模型
                response = MultiModalConversation.call(
                    model=self.eval_model,
                    messages=messages,
                )

                if response.status_code == 200:
                    # DashScope 多模态返回的 content 通常是一个包含 dict 的 list
                    raw_content = response.output.choices[0].message.content
                    text_result = ""
                    if isinstance(raw_content, list):
                        for item in raw_content:
                            if 'text' in item:
                                text_result += item['text']
                    else:
                        text_result = str(raw_content)

                    # 鲁棒解析 JSON (防止大模型加上 ```json 等 markdown 标记)
                    json_match = re.search(r'\{.*\}', text_result, re.DOTALL)
                    if json_match:
                        result_dict = json.loads(json_match.group())
                        # 确保 key 存在
                        if "passed" in result_dict:
                            return result_dict

                    print(f"⚠️ VLM 裁判返回格式异常，未找到合法的 JSON。原始输出: {text_result}")
                    raise ValueError("JSON parse failed")

                else:
                    time.sleep(3)
                    raise RuntimeError(f"VLM API 调用失败: {response.code} - {response.message}")

            except Exception as e:
                api_attempt += 1
                print(f"⚠️ VLM 裁判开小差了 (重试 {api_attempt}/3): {str(e)}")
                time.sleep(2)

        # 兜底：如果 API 彻底挂了，默认让它重画
        return {"passed": False, "reason": "VLM 裁判多次调用失败，默认复查不通过。"}

    def _get_pose_bbox_from_annotation(self, annotation, image_path, pad_ratio=0.15):
        """
        Return bbox in xyxy format for MMPose top-down inference.
        Prefer COCO annotation bbox if available. Otherwise compute bbox from visible keypoints.

        COCO bbox: [x, y, w, h]
        MMPose top-down bbox: [[x1, y1, x2, y2]]
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h, w = img.shape[:2]

        kpts = np.array(annotation["keypoints"], dtype=np.float32).reshape(-1, 3)
        visible = kpts[kpts[:, 2] > 0]

        if len(visible) < 2:
            return np.array([[0, 0, w, h]], dtype=np.float32)

        x1, y1 = np.min(visible[:, :2], axis=0)
        x2, y2 = np.max(visible[:, :2], axis=0)

        bw = x2 - x1
        bh = y2 - y1

        # expand bbox a bit
        pad_x = bw * pad_ratio
        pad_y = bh * pad_ratio

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w - 1, x2 + pad_x)
        y2 = min(h - 1, y2 + pad_y)

        return np.array([[x1, y1, x2, y2]], dtype=np.float32)

    def run(self, image_path, base_output_dir, original_annotation, max_attempts=3):
        print("\n" + "=" * 50)
        print("🚀 AgentHMR-Res proxy generation with dual-critic validation")
        print("=" * 50)

        kpt_types_orig = original_annotation.get("keypoint_types", [])
        kpts_orig = np.array(original_annotation["keypoints"]).reshape(-1, 3)

        res = self.generate_prompt(image_path, kpts_orig, kpt_types_orig)

        if res["passed"]:
            print(f"✅ {res['reason']}")
            return image_path

        master_prompt = res["prompt"]
        print(f"📋 初诊发现缺陷，生成局部修改指令: {res['reason']}")
        print(f"📝 Master Prompt: {master_prompt}")

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_dir = os.path.join(base_output_dir, base_name)
        os.makedirs(save_dir, exist_ok=True)

        for attempt in range(max_attempts):
            print(f"\n🔄 尝试生成 proxy image ({attempt + 1}/{max_attempts})...")

            gen_image_path = self.generate_full_image(
                image_path,
                save_dir,
                master_prompt,
                str(attempt)
            )

            if not gen_image_path:
                print("❌ 图片生成失败，跳过本次重试。")
                continue

            gen_image_path = self._resize_generated_to_original(gen_image_path, image_path)

            # 1. Pose-based geometric critic
            try:
                bbox = self._get_pose_bbox_from_annotation(original_annotation, image_path)
                kpts_gen = self.pose_extractor.extract_17_keypoints(gen_image_path, bbox)
            except Exception as e:
                print(f"❌ 生成图 pose extraction 失败: {e}")
                continue
            pose_est_vis_path = os.path.join(save_dir, f"attempt_{attempt}_pose_estimation_debug.png")
            self._save_pose_estimation_comparison_vis(
                orig_image_path=image_path,
                gen_image_path=gen_image_path,
                kpts_orig=kpts_orig,
                kpts_gen=kpts_gen,
                save_path=pose_est_vis_path,
                kpt_types_orig=kpt_types_orig,
                conf_thr=0.10,
                show_indices=True,
            )
            print(f"   pose estimation debug: {pose_est_vis_path}")

            pose_eval = self.evaluate_pose_geometric(
                kpts_orig=kpts_orig,
                kpts_gen=kpts_gen,
                kpt_types_orig=kpt_types_orig,
                orig_image_path=image_path,
                gen_image_path=gen_image_path,
                vis_dir=save_dir,
                vis_prefix=f"attempt_{attempt}"
            )

            print(f"📐 Pose critic: {'✅ 通过' if pose_eval['passed'] else '❌ 未通过'}")
            print(f"   {pose_eval['reason']}")

            if pose_eval.get("pose_vis_path") is not None:
                print(f"   pose vis: {pose_eval['pose_vis_path']}")
            if pose_eval.get("direction_vis_path") is not None:
                print(f"   direction vis: {pose_eval['direction_vis_path']}")

            if not pose_eval["passed"]:
                print("⚠️ Pose critic failed, regenerate from original image.")
                continue

            # 2. VLM-based visual-quality critic
            gen_eval_res = self.evaluate_image(gen_image_path, res["eval_instruction"])

            print(f"👁️ VLM critic: {'✅ 通过' if gen_eval_res['passed'] else '❌ 未通过'}")
            print(f"   {gen_eval_res.get('reason', '')}")

            if gen_eval_res["passed"]:
                print("🎉 Proxy image passed dual-critic validation!")
                return gen_image_path

            print("⚠️ VLM critic failed, regenerate from original image.")
        print("❌ No proxy image passed dual-critic validation.")
        return None

    def generate_full_image(self, image_path, save_dir, prompt, iter):
        """
        调用 Qwen 大模型，让它自由发挥补全肢体（生成素材库）。
        """
        print(f"\n🎨 [生成] 调用 {self.edit_model} 生成完整肢体素材...")

        # 1. 编码原图 (沿用你之前的工具类)
        image_encoded = ImageProcessor.encode_file(image_path)

        # 2. 组装最干净的 Payload：只有原图和补全指令
        content_list = [
            {"image": image_encoded},
            {"text": prompt.strip()}
        ]
        messages = Message(
            role="user",
            content= content_list
        )

        # 3. 稳健的网络请求重试机制
        api_attempt = 0
        while api_attempt < 3:
            try:
                print(os.getenv("DASHSCOPE_API_KEY"))
                # 沿用你之前跑通的 API 调用参数
                response = ImageGeneration.call(
                    model=self.edit_model,
                    messages=[messages],
                    n=1,
                )

                if response.status_code == 200:
                    for i, choice in enumerate(response.output.choices):
                        for j, content in enumerate(choice["message"]["content"]):
                            if content.get("type") == "image":
                                print(f"✅ 素材库生成成功！")
                                # 下载并保存图片 (这里传 attempt_suffix="raw_gen" 以示区分)
                                path = ImageProcessor.save_image_from_url(
                                    content['image'],
                                    'compositing_material',
                                    iter,
                                    save_dir
                                )
                                return path
                    raise RuntimeError("❌ API 返回了 200，但未找到图片链接。")
                else:
                    time.sleep(3)
                    raise RuntimeError(f"❌ 图像生成失败: HTTP {response.status_code}, {response.message}")

            except Exception as e:
                api_attempt += 1
                time.sleep(3)
                print(f"❌ API 调用崩溃 (网络重试 {api_attempt}/3): {str(e)}")

        return None


if __name__ == "__main__":
    image_dir = Path('./3D_data/images_seg')
    # image_dir = Path('./eval')
    save_dir = Path('./workdir4')

    # pose_extractor = PoseExtractor(
    #     config_file='./models/pose/vit_config.py',
    #     checkpoint_file='./models/pose/epoch_1.pth',
    #     device='cuda:0'
    # )

    pose_extractor = PoseExtractor(
        config_file = "/home/sora/workspace/mmpose-custom/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py",
        checkpoint_file='./models/pose/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth',
        device='cuda:0'
    )
    agent = LimbCompositingAgent(pose_extractor)

    # 1. 一次性读取 COCO 格式的 JSON
    annotation_path = Path('./3D_data/annotations_propose_coco.json')
    with open(annotation_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 2. 解析 images 列表，建立 image_id -> file_name 的快速映射
    image_id_to_name = {}
    for img_info in coco_data.get('images', []):
        # 假设 JSON 里存的是 "baidu_残疾人跑步_21.png" 或者带路径的相对地址
        # 用 Path(xxx).name 只取纯文件名，方便后续和本地文件完美匹配
        pure_name = Path(img_info['file_name']).name
        image_id_to_name[img_info['id']] = pure_name

    # 3. 解析 annotations 列表，建立 file_name -> annotation 的终极映射字典
    anno_dict = {}
    for anno in coco_data.get('annotations', []):
        image_id = anno.get('image_id')
        if image_id in image_id_to_name:
            file_name = image_id_to_name[image_id]
            # 假设每张图只有一个核心标注目标
            # 注意：COCO 格式中关键点通常存在 anno['keypoints']，如果有你自定义的 'keypoint_types' 也会在这里面
            anno_dict[file_name] = anno

    # 4. 开始遍历本地文件夹里的图片
    for i, image_path in enumerate(image_dir.glob('*.png')):
        if i > 10: break
        img_name = image_path.name  # 取纯文件名

        # 5. 精准匹配
        if img_name in anno_dict:
            current_image_annotation = anno_dict[img_name]

            print(f"\n[{i + 1}] 正在处理: {img_name}")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            target_dir = os.path.join(save_dir, base_name)
            # if os.path.exists(target_dir):
            #     print(f"⚠️ 目录 {target_dir} 已存在，跳过该图以避免覆盖。")
            #     continue
            # 传给 agent！
            agent.run(str(image_path), str(save_dir), current_image_annotation)
        else:
            print(f"⚠️ 警告: JSON 中没有找到图片 {img_name} 的关联标注，跳过该图。")