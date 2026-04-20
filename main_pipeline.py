import os
from pathlib import Path

import dashscope
import numpy as np
import requests
import time
import cv2
import argparse
from PIL import Image

from pose_extractor import PoseExtractor, read_kpts_annotation
from image_ops import OSSProcessor
from agentic_critic import AgenticImageEditor, GeometricRefinerAgent
from reconstruction_3d import ReconstructionEngine
from residual_mesh_cutter import ResidualMeshCutter
from sam2_coco import SAM2Predictor

dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def parse_args():
    parser = argparse.ArgumentParser(description="🚀 Neuro-Symbolic Agentic 3D Pipeline")

    # 基础输入参数
    parser.add_argument('--img_dir', type=str,
                        default='./eval',
                        help='Dir of the input images')

    # PoseExtractor 参数 (必填或提供默认路径)
    parser.add_argument('--pose_config', type=str,
                        default='./models/pose/vit_config.py',
                        help='Path to the LDPose MMPose config file')
    parser.add_argument('--pose_ckpt', type=str,
                        default='./models/pose/epoch_1.pth',
                        help='Path to the LDPose checkpoint file')
    parser.add_argument('--device', type=str,
                        default='cuda:0',
                        help='Device to run pose extraction on (e.g., cuda:0 or cpu)')
    parser.add_argument('--annotation_file', type=str,
                        default='./data/train_final.json',
                        help='Path to the annotation file (optional)')
    parser.add_argument('--output_dir', default='./workdir5', type=str)

    return parser.parse_args()


def predict(args, img_path, output_path, pose_extractor, reconstructor, geometric_refiner, image_editor, sam2_predictor):
    print("====== 🚀 Neuro-Symbolic Agentic 3D Pipeline ======")

    if not os.path.exists(img_path):
        print(f"❌ 找不到输入图片: {img_path}")
        return

    # 1. 初始化所有类 (传入 argparse 解析好的参数)

    print(f"\n[Processing] 开始处理图像: {img_path}")
    kpts_orig, kpts, types_orig = read_kpts_annotation(img_path, args.annotation_file)
    sam2_img_path, mask = sam2_predictor.segment_subject(img_path, output_path, kpts_orig)

    image_url = OSSProcessor().upload_and_get_url(local_file_path=sam2_img_path)

    # ==========================================
    # 4. Agentic Loop (生成 -> Critic -> 再生成)
    # ==========================================
    if not isinstance(output_path, str):
        output_path = f"./workdir/output_{int(time.time())}"
    generated_image_urls = image_editor.run(image_url)
    last_generated_image_url = generated_image_urls[-1]
    save_image_from_url(generated_image_urls, "image_editor", output_path)
    generated_image_urls = geometric_refiner.run(kpts_orig, last_generated_image_url, output_path)
    all_path = save_image_from_url(generated_image_urls, "geometric_refiner", output_path)
    final_image_url = generated_image_urls[-1]
    final_local_path = all_path[-1]

    # ==========================================
    # 5. 3D Mesh 恢复
    # ==========================================
    print("\n=== 进入 3D 恢复阶段 ===")
    try:
        # 1. 从 URL 获取图片数据
        print(f"⬇️ 正在拉取处理完成的网络图片...")
        response = requests.get(final_image_url, timeout=15)
        response.raise_for_status()

        # 2. 将二进制数据转换为 numpy 数组，再通过 cv2 解码
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError("❌ 图片解码失败，可能链接已损坏或不是有效图片格式。")

        # 3. 转换为 RGB (为了你的 3D 模型推理)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 4. 执行 3D 恢复
        # 注意: 这里的 img_path 可能依然是你最开始的本地路径，用于提取基础文件名
        base_name = os.path.basename(img_path).split('.')[0]
        mesh_save_path = os.path.join(output_path, f"{base_name}_mesh.obj")

        mesh_save_path, pred_joints_3d, pred_cam = reconstructor.predict_mesh(final_local_path, mesh_save_path)
        print(f"✅ 3D Mesh 已保存至: {mesh_save_path}")
        # ==========================================
        # 6. 精确截肢 (Multi-Mesh Truncation)
        # ==========================================
        # 定义残肢点名称 -> 对应 SMPL 3D 关节的映射字典
        RES_BONE_MAPPING = {
            'L-Elbow-Res-Above': ('left_shoulder', 'left_elbow'),
            'R-Elbow-Res-Above': ('right_shoulder', 'right_elbow'),
            'L-Elbow-Res-Below': ('left_elbow', 'left_wrist'),
            'R-Elbow-Res-Below': ('right_elbow', 'right_wrist'),
            'L-Knee-Res-Above': ('left_hip', 'left_knee'),
            'R-Knee-Res-Above': ('right_hip', 'right_knee'),
            'L-Knee-Res-Below': ('left_knee', 'left_ankle'),
            'R-Knee-Res-Below': ('right_knee', 'right_ankle')
        }
        METAINFO = {
            'dataset_name': 'ld_pros_pose',
            'classes': ('person',),
            'num_keypoints': 31,
            # === 1. 关键点定义 (完全对应截图) ===
            'keypoint_info': {
                # --- Part A: COCO 原生 17 点 (ID 0-16) ---
                0: dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
                1: dict(name='left_eye', id=1, color=[51, 153, 255], type='upper', swap='right_eye'),
                2: dict(name='right_eye', id=2, color=[51, 153, 255], type='upper', swap='left_eye'),
                3: dict(name='left_ear', id=3, color=[51, 153, 255], type='upper', swap='right_ear'),
                4: dict(name='right_ear', id=4, color=[51, 153, 255], type='upper', swap='left_ear'),
                5: dict(name='left_shoulder', id=5, color=[0, 255, 0], type='upper', swap='right_shoulder'),
                6: dict(name='right_shoulder', id=6, color=[255, 128, 0], type='upper', swap='left_shoulder'),
                7: dict(name='left_elbow', id=7, color=[0, 255, 0], type='upper', swap='right_elbow'),
                8: dict(name='right_elbow', id=8, color=[255, 128, 0], type='upper', swap='left_elbow'),
                9: dict(name='left_wrist', id=9, color=[0, 255, 0], type='upper', swap='right_wrist'),
                10: dict(name='right_wrist', id=10, color=[255, 128, 0], type='upper', swap='left_wrist'),
                11: dict(name='left_hip', id=11, color=[0, 255, 0], type='lower', swap='right_hip'),
                12: dict(name='right_hip', id=12, color=[255, 128, 0], type='lower', swap='left_hip'),
                13: dict(name='left_knee', id=13, color=[0, 255, 0], type='lower', swap='right_knee'),
                14: dict(name='right_knee', id=14, color=[255, 128, 0], type='lower', swap='left_knee'),
                15: dict(name='left_ankle', id=15, color=[0, 255, 0], type='lower', swap='right_ankle'),
                16: dict(name='right_ankle', id=16, color=[255, 128, 0], type='lower', swap='left_ankle'),

                # --- Part B: 自定义残肢/假肢点 (ID 17-24) ---
                17: dict(name='L_Middle_Tip', id=17, color=[255, 0, 255], type='upper', swap='R_Middle_Tip'),
                18: dict(name='R_Middle_Tip', id=18, color=[255, 0, 255], type='upper', swap='L_Middle_Tip'),
                19: dict(name='L_Heel', id=19, color=[255, 0, 255], type='lower', swap='R_Heel'),
                20: dict(name='R_Heel', id=20, color=[255, 0, 255], type='lower', swap='L_Heel'),
                21: dict(name='L_Toe_Tip', id=21, color=[255, 0, 255], type='lower', swap='R_Toe_Tip'),
                22: dict(name='R_Toe_Tip', id=22, color=[255, 0, 255], type='lower', swap='L_Toe_Tip'),

                # 23-30: 残肢点 (Res KPs)
                23: dict(name='L-Elbow-Res-Above', id=23, color=[255, 0, 0], type='upper', swap='R-Elbow-Res-Above'),
                24: dict(name='R-Elbow-Res-Above', id=24, color=[255, 0, 0], type='upper', swap='L-Elbow-Res-Above'),
                25: dict(name='L-Elbow-Res-Below', id=25, color=[255, 0, 0], type='upper', swap='R-Elbow-Res-Below'),
                26: dict(name='R-Elbow-Res-Below', id=26, color=[255, 0, 0], type='upper', swap='L-Elbow-Res-Below'),
                27: dict(name='L-Knee-Res-Above', id=27, color=[255, 0, 0], type='lower', swap='R-Knee-Res-Above'),
                28: dict(name='R-Knee-Res-Above', id=28, color=[255, 0, 0], type='lower', swap='L-Knee-Res-Above'),
                29: dict(name='L-Knee-Res-Below', id=29, color=[255, 0, 0], type='lower', swap='R-Knee-Res-Below'),
                30: dict(name='R-Knee-Res-Below', id=30, color=[255, 0, 0], type='lower', swap='L-Knee-Res-Below'),
            },

            # === 2. 骨架连接 (根据 ID 和解剖逻辑推导) ===
            'skeleton_info': {
                # --- Part A: 基础连线 (0-16 为原生或基础结构) ---
                0: dict(link=('nose', 'left_eye'), id=0, color=[51, 153, 255]),
                1: dict(link=('nose', 'right_eye'), id=1, color=[51, 153, 255]),
                2: dict(link=('left_eye', 'left_ear'), id=2, color=[51, 153, 255]),
                3: dict(link=('right_eye', 'right_ear'), id=3, color=[51, 153, 255]),
                4: dict(link=('left_shoulder', 'right_shoulder'), id=4, color=[51, 153, 255]),
                5: dict(link=('left_shoulder', 'left_elbow'), id=5, color=[0, 255, 0]),
                6: dict(link=('left_elbow', 'left_wrist'), id=6, color=[0, 255, 0]),
                7: dict(link=('right_shoulder', 'right_elbow'), id=7, color=[255, 128, 0]),
                8: dict(link=('right_elbow', 'right_wrist'), id=8, color=[255, 128, 0]),
                9: dict(link=('left_shoulder', 'left_hip'), id=9, color=[51, 153, 255]),
                10: dict(link=('right_shoulder', 'right_hip'), id=10, color=[51, 153, 255]),
                11: dict(link=('left_hip', 'right_hip'), id=11, color=[51, 153, 255]),
                12: dict(link=('left_hip', 'left_knee'), id=12, color=[0, 255, 0]),
                13: dict(link=('left_knee', 'left_ankle'), id=13, color=[0, 255, 0]),
                14: dict(link=('right_hip', 'right_knee'), id=14, color=[255, 128, 0]),
                15: dict(link=('right_knee', 'right_ankle'), id=15, color=[255, 128, 0]),

                # --- Part B: 新增点连线 (17-22: 肢体末端) ---
                16: dict(link=('left_wrist', 'L_Middle_Tip'), id=16, color=[0, 255, 255]),
                17: dict(link=('right_wrist', 'R_Middle_Tip'), id=17, color=[255, 0, 255]),
                18: dict(link=('left_ankle', 'L_Heel'), id=18, color=[0, 255, 255]),
                19: dict(link=('left_ankle', 'L_Toe_Tip'), id=19, color=[0, 255, 255]),
                20: dict(link=('right_ankle', 'R_Heel'), id=20, color=[255, 0, 255]),
                21: dict(link=('right_ankle', 'R_Toe_Tip'), id=21, color=[255, 0, 255]),

                # --- Part C: 残肢连线 (23-30: 对应 RES_KPS) ---
                22: dict(link=('left_shoulder', 'L-Elbow-Res-Above'), id=22, color=[255, 0, 0]),
                23: dict(link=('right_shoulder', 'R-Elbow-Res-Above'), id=23, color=[255, 0, 0]),
                24: dict(link=('left_elbow', 'L-Elbow-Res-Below'), id=24, color=[255, 0, 0]),
                25: dict(link=('right_elbow', 'R-Elbow-Res-Below'), id=25, color=[255, 0, 0]),
                26: dict(link=('left_hip', 'L-Knee-Res-Above'), id=26, color=[255, 0, 0]),
                27: dict(link=('right_hip', 'R-Knee-Res-Above'), id=27, color=[255, 0, 0]),
                28: dict(link=('left_knee', 'L-Knee-Res-Below'), id=28, color=[255, 0, 0]),
                29: dict(link=('right_knee', 'R-Knee-Res-Below'), id=29, color=[255, 0, 0]),
            },

            # === 3. 翻转时对应的 ID 列表 ===
            # 这个列表非常关键，MMPose 训练时 Flip 增强就是靠这个 list 知道谁和谁互换
            # 格式：[1.0] * 25
            'joint_weights': [1.] * 31,

            # Sigma (用于 OKS 计算)，给自定义点一个默认值 0.05
            'sigmas': [
                0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
                0.089, 0.089, 0.089, 0.089, 0.089, 0.089,
                0.072, 0.072, 0.062, 0.062, 0.087, 0.087, 0.089, 0.089
            ],
        }

        cut_tasks = []
        kpts_gen = pose_extractor.extract_31_keypoints(final_image_url)  # 重新检测生成图的关键点
        M_inv = get_final_calibration_matrix(kpts_orig, kpts_gen=kpts_gen, image_path=all_path[-1])  # 这里暂时传 None，后续可以改成实际生成图的关键点

        mask_gen = sam2_predictor.get_mask_only(
            final_local_path,
            kpts_gen,
            types_orig
        )

        # 2. 提取 Orig Image 的 SAM2 Mask
        # (如果你前面已经切过了，可以直接用前面的变量。这里为了保险重新调一次 segment_subject2)
        mask_orig = sam2_predictor.get_mask_only(
            img_path,
            kpts_orig,
            types_orig
        )

        # 3. 核心调用：生成叠加差异图
        visualize_alignment_diff(
            orig_img_path=img_path,
            mask_orig=mask_orig,
            mask_gen=mask_gen,
            kpts_orig=kpts_orig,
            kpts_gen=kpts_gen,
            M_orig_to_gen=M_inv,  # get_final_calibration_matrix 返回的是 Orig->Gen
            output_dir=output_path
        )

        # 遍历 METAINFO 中的最后 8 个残肢点 (ID 23 到 30)
        for i in range(23, 31):
            # 判断: 只有 type == 0 才是有效残肢点，且确保坐标数组够长
            if types_orig[i] == 0:
                res_name = METAINFO['keypoint_info'][i]['name']

                # 获取 2D 坐标 (假设 kpts_orig 的格式是 [x, y, conf])
                pt_2d_orig = kpts_orig[i][0:2]
                pt_2d_orig_homo = np.array([pt_2d_orig[0], pt_2d_orig[1], 1.0])
                pt_2d_gen = (M_inv @ pt_2d_orig_homo)[:2]  # Warp the point
                # 查表找到对应的 3D 骨骼起点和终点
                if res_name in RES_BONE_MAPPING:
                    start_joint_name, end_joint_name = RES_BONE_MAPPING[res_name]

                    start_3d = pred_joints_3d[start_joint_name]
                    end_3d = pred_joints_3d[end_joint_name]

                    # 组装切割任务
                    cut_tasks.append({
                        'name': res_name,
                        'pt_2d': pt_2d_gen,
                        'start_3d': start_3d,
                        'end_3d': end_3d
                    })

        # 如果收集到了切割任务，才去执行截断
                    # 如果收集到了切割任务，才去执行截断
        if cut_tasks:
            gen_h, gen_w = img_bgr.shape[:2]

            # 🌟 直接使用原作者推导出的官方全局相机内参
            global_focal = pred_cam['focal']
            global_cx = pred_cam['princpt'][0]
            global_cy = pred_cam['princpt'][1]

            mesh_cutter = ResidualMeshCutter(
                focal_length=global_focal[0],
                img_center=(global_cx, global_cy)
            )
            mesh = mesh_cutter.process_multiple_cuts(
                mesh_path=mesh_save_path,
                cut_tasks=cut_tasks,
                M_inv=None,
            )
        else:
            raise ValueError("No residual bone cutting tasks found.")
        global_cam = {
            'focal': global_focal,
            'princpt': np.array([global_cx, global_cy])
        }

        print(f"\n--- 🧪 坐标系法医鉴定报告 ---")
        print(f"1. 生成图尺寸 (H, W): {img_bgr.shape[:2]}")
        print(f"2. 模型原始 Focal: {pred_cam['focal']}")
        print(f"3. 放大后的 Global Focal: {global_focal}")
        print(f"4. 你设定的投影中心: ({global_cx}, {global_cy})")

        # 计算生成图里人的实际 2D 中心
        person_center_gen = np.mean(kpts_gen[kpts_gen[:, 2] > 0.3, :2], axis=0)
        print(f"5. 生成图里人的实际 2D 中心: {person_center_gen}")

        # 计算偏差
        offset_y = global_cy - person_center_gen[1]
        print(f"6. 垂直方向偏差 (Offset Y): {offset_y:.2f} 像素 (正值代表投影偏高)")
        # ================= DEBUG BLOCK END =================

        orig_proj_path, pred_mask_orig, gen_proj_path, pred_mask_gen = project_mesh_overlay(img_path, all_path[-1], mesh, M_inv, global_cam, output_path)  # 将最终 Mesh 投影回原图坐标系，生成 Overlay
        sam2_img_path, mask_gt = sam2_predictor.segment_subject2(img_path, output_path, kpts, types_orig)
        miou_score = calculate_miou(pred_mask_orig, mask_gt)
        print(f"📊 [量化评估] 掩码 mIoU 评分: {miou_score:.4f}")

        INTACT_MAPPING = {METAINFO['keypoint_info'][i]['name']: i for i in range(0, 17)}

        for task in cut_tasks:
            # 如果你在 cutter 里存了，可以直接这样取：
            if 'cut_origin' in task:
                pred_joints_3d[task['name']] = task['cut_origin']
            else:
                # Fallback: 如果拿不到，暂时用 start 和 end 的中点代替以防止报错
                pred_joints_3d[task['name']] = (task['start_3d'] + task['end_3d']) / 2.0

        RES_MAPPING = {METAINFO['keypoint_info'][i]['name']: i for i in range(23, 31)}



        mpjpe_intact = calculate_2d_mpjpe(pred_joints_3d, kpts_orig, M_inv, global_cam, INTACT_MAPPING)
        mpjpe_residual = calculate_2d_mpjpe(pred_joints_3d, kpts_orig, M_inv, global_cam, RES_MAPPING)

        print(f"📊 [量化评估] 完整关节 2D MPJPE: {mpjpe_intact:.2f} pixels")
        print(f"📊 [量化评估] 残肢端点 2D MPJPE: {mpjpe_residual:.2f} pixels")

        return miou_score, mpjpe_intact, mpjpe_residual
    except Exception as e:
        raise e


def project_mesh_overlay(image_path, gen_image_path, mesh, M_inv, global_cam, output_dir):
    """
    实现 3D Mesh 到 Gen 图和 Orig 图的双重精准投影
    """
    # 1. 提取生成图空间的相机内参
    f_gen = global_cam['focal']  # [fx, fy]
    c_gen = global_cam['princpt']  # [cx, cy]

    # 2. 构造生成图空间的投影矩阵 K_gen (3x3)
    K_gen = np.array([
        [f_gen[0], 0, c_gen[0]],
        [0, f_gen[1], c_gen[1]],
        [0, 0, 1]
    ])

    vertices = mesh.vertices
    pts_3d = vertices.T  # (3, N)
    valid_faces = mesh.faces.astype(np.int32)

    # ====================================================
    # 🌟 新增逻辑：A. 将 Mesh 投影到 Gen 图 (直接使用 K_gen)
    # ====================================================
    pts_2d_homo_gen = K_gen @ pts_3d

    zs_gen = pts_2d_homo_gen[2, :]
    zs_gen[zs_gen < 1e-5] = 1e-5

    u_gen = pts_2d_homo_gen[0, :] / zs_gen
    v_gen = pts_2d_homo_gen[1, :] / zs_gen
    projected_pts_gen = np.stack([u_gen, v_gen], axis=1).astype(np.int32)

    # 读取 Gen 图
    gen_image_data = np.fromfile(gen_image_path, dtype=np.uint8)
    gen_image = cv2.imdecode(gen_image_data, cv2.IMREAD_COLOR)
    h_gen, w_gen = gen_image.shape[:2]

    mask_gen = np.zeros((h_gen, w_gen), dtype=np.uint8)

    for face in valid_faces:
        if np.all(zs_gen[face] > 0.1):
            pts = projected_pts_gen[face].reshape((-1, 1, 2))
            cv2.fillPoly(mask_gen, [pts], 255)

    overlay_color_gen = np.zeros_like(gen_image)
    overlay_color_gen[:] = [0, 255, 0]
    mask_bool_gen = (mask_gen > 127)
    output_image_gen = gen_image.copy()
    output_image_gen = cv2.addWeighted(overlay_color_gen, 0.5, output_image_gen, 0.5, 0)
    output_image_gen[~mask_bool_gen] = gen_image[~mask_bool_gen]

    gen_out_name = os.path.basename(gen_image_path).replace(".jpg", "_gen_projection.jpg")
    gen_out_path = os.path.join(output_dir, gen_out_name)
    cv2.imencode(".jpg", output_image_gen)[1].tofile(gen_out_path)


    # ====================================================
    # 原有逻辑：B. 将 Mesh 投影回 Orig 原图 (使用仿射逆变换)
    # ====================================================
    # 3. 将 2x3 的 M_inv (Orig -> Gen) 扩展为 3x3 齐次矩阵并求逆
    M_orig_to_gen = np.vstack([M_inv, [0, 0, 1]])
    M_gen_to_orig = np.linalg.inv(M_orig_to_gen)

    # 4. 合并变换：投影到 Gen 平面 -> 仿射变换回 Orig 平面
    P_final = M_gen_to_orig @ K_gen

    # 5. 计算 Orig 投影坐标
    pts_2d_homo_orig = P_final @ pts_3d

    zs_orig = pts_2d_homo_orig[2, :]
    zs_orig[zs_orig < 1e-5] = 1e-5

    u_orig = pts_2d_homo_orig[0, :] / zs_orig
    v_orig = pts_2d_homo_orig[1, :] / zs_orig
    projected_pts_orig = np.stack([u_orig, v_orig], axis=1).astype(np.int32)

    # 读取 Orig 原图
    orig_image_data = np.fromfile(image_path, dtype=np.uint8)
    original_image = cv2.imdecode(orig_image_data, cv2.IMREAD_COLOR)
    h_orig, w_orig = original_image.shape[:2]

    mask_orig = np.zeros((h_orig, w_orig), dtype=np.uint8)

    for face in valid_faces:
        if np.all(zs_orig[face] > 0.1):
            pts = projected_pts_orig[face].reshape((-1, 1, 2))
            cv2.fillPoly(mask_orig, [pts], 255)

    overlay_color_orig = np.zeros_like(original_image)
    overlay_color_orig[:] = [0, 255, 0]
    mask_bool_orig = (mask_orig > 127)
    output_image_orig = original_image.copy()
    output_image_orig = cv2.addWeighted(overlay_color_orig, 0.5, output_image_orig, 0.5, 0)
    output_image_orig[~mask_bool_orig] = original_image[~mask_bool_orig]

    orig_out_name = os.path.basename(image_path).replace(".jpg", "_orig_projection.jpg")
    orig_out_path = os.path.join(output_dir, orig_out_name)
    cv2.imencode(".jpg", output_image_orig)[1].tofile(orig_out_path)

    # 返回两个图的路径和 Mask，方便后续计算 mIoU
    return orig_out_path, mask_orig, gen_out_path, mask_gen


def visualize_alignment_diff(orig_img_path, mask_orig, mask_gen, kpts_orig, kpts_gen, M_orig_to_gen, output_dir):
    """
    可视化原图与生成图的分割 Mask 和 Keypoints 之间的绝对形变差异
    """
    if mask_orig is None or mask_gen is None:
        print("⚠️ 警告: 缺少 Mask，跳过形变差异可视化。")
        return None

    # 1. 读取原图作为底板
    orig_img_data = np.fromfile(orig_img_path, dtype=np.uint8)
    orig_img = cv2.imdecode(orig_img_data, cv2.IMREAD_COLOR)
    h_orig, w_orig = orig_img.shape[:2]

    # 2. 将 M_orig_to_gen (2x3) 转换为 M_gen_to_orig (2x3)
    M_aug = np.vstack([M_orig_to_gen, [0, 0, 1]])
    M_gen_to_orig = np.linalg.inv(M_aug)[:2, :]

    # 3. 将 Gen 的 Mask 仿射变换回 Orig 坐标系
    # 注意：mask_gen 是在 Gen 图上切出来的，我们需要把它拉平回原图的视角
    mask_gen_aligned = cv2.warpAffine(mask_gen, M_gen_to_orig, (w_orig, h_orig), flags=cv2.INTER_NEAREST)

    # 4. 绘制 Mask 叠加层
    mask_overlay = np.zeros_like(orig_img)
    mask_overlay[mask_orig > 127] = [0, 255, 0]  # 🟢 绿色：原图真实轮廓
    mask_overlay[mask_gen_aligned > 127] = [0, 0, 255]  # 🔴 红色：生成图变形后的轮廓

    # 计算重叠部分变为黄色 (Green + Red = Yellow)
    overlap = (mask_orig > 127) & (mask_gen_aligned > 127)
    mask_overlay[overlap] = [0, 255, 255]  # 🟡 黄色：完美对齐的区域

    # 将 Mask 半透明叠加到底板上
    alpha = 0.5
    comp_img = cv2.addWeighted(mask_overlay, alpha, orig_img, 1 - alpha, 0)

    # 5. 绘制 Keypoints 和形变向量 (拉扯线)
    for i in range(len(kpts_orig)):
        v_orig = kpts_orig[i][2] if len(kpts_orig[i]) > 2 else 1
        v_gen = kpts_gen[i][2] if len(kpts_gen[i]) > 2 else 1

        # 只对比两边都可见(>0.3)的有效关键点
        if v_orig > 0.3 and v_gen > 0.3:
            pt_orig = (int(kpts_orig[i][0]), int(kpts_orig[i][1]))

            # 将 Gen 关键点变换回 Orig 坐标系
            pt_gen_homo = np.array([kpts_gen[i][0], kpts_gen[i][1], 1.0])
            pt_gen_aligned = M_gen_to_orig @ pt_gen_homo
            pt_gen_aligned = (int(pt_gen_aligned[0]), int(pt_gen_aligned[1]))

            # 连线：黄色形变向量 (可以看出大模型把这个关节往哪边拉扯了)
            cv2.line(comp_img, pt_orig, pt_gen_aligned, (0, 255, 255), 2)

            # 画点：绿点(Orig) vs 红点(Gen)
            cv2.circle(comp_img, pt_orig, 5, (0, 255, 0), -1)
            cv2.circle(comp_img, pt_gen_aligned, 4, (0, 0, 255), -1)

    # 添加图例文字
    cv2.putText(comp_img, "Green: Orig  Red: Gen  Yellow: Overlap", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 6. 保存图片
    image_name = os.path.basename(orig_img_path).replace(".jpg", "_alignment_diff.jpg")
    save_path = os.path.join(output_dir, image_name)
    cv2.imencode(".jpg", comp_img)[1].tofile(save_path)
    print(f"👁️ [形变评估] 差异对比图已保存: {save_path}")

    return save_path

def calculate_miou(pred_mask, gt_mask):
    """
    计算平均交并比 (mIoU)
    :param pred_mask: 预测的 2D 掩码 (np.array)
    :param gt_mask: 真实的 2D 掩码 (np.array)
    """
    # 确保转换为二值布尔数组
    pred_bin = (pred_mask > 127).astype(np.bool_)
    gt_bin = (gt_mask > 127).astype(np.bool_)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def calculate_2d_mpjpe(pred_joints_3d, gt_kpts_orig, M_inv, global_cam, mapping_dict):
    # Gen -> Orig
    M_augmented = np.vstack([M_inv, [0, 0, 1]])
    M_gen_to_orig = np.linalg.inv(M_augmented)[:2, :]

    focal = global_cam['focal']
    princpt = global_cam['princpt']

    errors = []
    for joint_name, gt_idx in mapping_dict.items():
        if joint_name not in pred_joints_3d: continue
        gt_x, gt_y, conf = gt_kpts_orig[gt_idx]
        if conf <= 0: continue

        vert = pred_joints_3d[joint_name]

        if vert[2] <= 1e-5: continue

        # 投影到 Gen
        proj_gen_x = focal[0] * (vert[0] / vert[2]) + princpt[0]
        proj_gen_y = focal[1] * (vert[1] / vert[2]) + princpt[1]

        # 映射回 Orig
        pt_orig = M_gen_to_orig @ np.array([proj_gen_x, proj_gen_y, 1.0])

        err = np.linalg.norm(pt_orig[:2] - np.array([gt_x, gt_y]))
        errors.append(err)

    return float(np.mean(errors)) if errors else 0.0


def get_final_calibration_matrix(kpts_orig, kpts_gen, image_path):
    """
    量化测试专用版：极致刚体对齐 (强制躯干主导，带安全兜底)
    """
    # 1. 定义最高优先级的核心躯干点：5(左肩), 6(右肩), 11(左胯), 12(右胯)
    core_indices = [5, 6, 11, 12]

    valid_src = []
    valid_dst = []

    # 提取核心点 (置信度需 > 0.3)
    for idx in core_indices:
        if kpts_orig[idx][2] > 0.3 and kpts_gen[idx][2] > 0.3:
            valid_src.append(kpts_orig[idx, :2])
            valid_dst.append(kpts_gen[idx, :2])

    src = np.array(valid_src, dtype=np.float32)
    dst = np.array(valid_dst, dtype=np.float32)

    # ========================================================
    # 🌟 策略 A：如果核心点充足 (>=3)，使用极致刚体对齐
    # ========================================================
    if len(src) >= 3:
        print("📐 [Calibration] 使用【核心躯干刚体】进行极致对齐 (method=0)")
        M_calib, inliers = cv2.estimateAffinePartial2D(
            src, dst,
            method=0  # 🚨 强制最小二乘法，绝不抛弃这几个核心点
        )

    # ========================================================
    # ⚠️ 策略 B (兜底)：如果下半身被遮挡，退回不包含头部的全骨架 RANSAC 对齐
    # ========================================================
    else:
        print("⚠️ [Calibration] 核心点被严重遮挡，退回【剔除头部】的全身体对齐方案...")
        # 包含四肢：5 到 16 (坚决不要 0-4 的头部点)
        fallback_indices = [i for i in range(5, 17)]
        fb_src = []
        fb_dst = []
        for i in fallback_indices:
            if kpts_orig[i][2] > 0.3 and kpts_gen[i][2] > 0.3:
                fb_src.append(kpts_orig[i, :2])
                fb_dst.append(kpts_gen[i, :2])

        fb_src = np.array(fb_src, dtype=np.float32)
        fb_dst = np.array(fb_dst, dtype=np.float32)

        if len(fb_src) < 3:
            raise ValueError("严重警告：有效对齐锚点不足 3 个，仿射矩阵计算失败！")

        # 因为四肢游离性强，必须开启 RANSAC 剔除被大模型画飞的畸形手臂/腿
        M_calib, inliers = cv2.estimateAffinePartial2D(
            fb_src, fb_dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=20.0
        )

    if M_calib is None:
        raise ValueError("对齐完全失败！")

    return M_calib

def save_image_from_url(urls, source, save_dir):
    all_path = []
    os.makedirs(save_dir, exist_ok=True)
    for i, url in enumerate(urls):
        filename = f"{source}_{i}.jpg"

        save_path = os.path.join(save_dir, filename)

        try:
            print(f"⬇️ 正在下载图片: {url[:50]}...")
            response = requests.get(url, timeout=15)
            response.raise_for_status()  # 检查请求是否成功

            with open(save_path, 'wb') as file:
                file.write(response.content)

            print(f"💾 成功保存到本地: {save_path}")
            all_path.append(save_path)

        except requests.exceptions.RequestException as e:
            print(f"❌ 下载失败: {e}")
    return all_path

def draw_keypoints_cv2(image_path, keypoints, color=(0, 0, 255), radius=5, thickness=-1, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, font_color=(255, 0, 0), font_thickness=2):
    """
    使用 OpenCV 在图片上绘制关键点及其索引（适配中文说明）
    """
    # 1. 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️ 无法读取图片：{image_path}")
        return

    # 2. 遍历关键点并绘制
    for i, pt in enumerate(keypoints):
        x, y = int(pt[0]), int(pt[1])
        # 适配图像边界检查
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            # --- 适配绘制关键点圆圈 ---
            cv2.circle(img, (x, y), radius, color, thickness)
            # --- 适配绘制关键点索引文本 ---
            # 为了防止文本超出边界或遮挡关键点，可以根据坐标微调文本位置
            text_pos = (x + radius, y - radius)
            cv2.putText(img, str(i), text_pos, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        else:
            print(f"⚠️ 关键点索引 {i} ({x}, {y}) 超出图像范围 {img.shape[1]}x{img.shape[0]}")

    # 3. 显示图像 (可选)
    cv2.imshow('OpenCV 关键点与索引可视化示例', img)
    print("窗口已打开，按下任意键关闭...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 4. 保存图像 (可选)
    save_path = image_path.replace('.', '_kpts_idx_cv22.')
    cv2.imwrite(save_path, img)
    print(f"✅ 适配绘制后的图像已保存至：{save_path}")

def main(args):
    miou_total = 0
    mpjpe_intact_total = 0
    mpjpe_residual_total = 0
    success_count = 0
    pose_extractor = PoseExtractor(config_file=args.pose_config,
                                   checkpoint_file=args.pose_ckpt,
                                   device=args.device)

    reconstructor = ReconstructionEngine()
    geometric_refiner = GeometricRefinerAgent(pose_extractor)
    image_editor = AgenticImageEditor()
    sam_predictor = SAM2Predictor()

    image_dir = Path(args.img_dir)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [
        f for f in image_dir.rglob('*') if f.suffix.lower() in valid_extensions
    ]
    log_filename = Path(args.output_dir) / "log.txt"

    for img_path in image_files:
        current_output_dir = Path(args.output_dir) / img_path.stem
        current_output_dir.mkdir(parents=True, exist_ok=True)
        attempts = 0

        while attempts < 3:
            try:
                print(
                    f"\n[Main] 📥 正在处理第 {success_count + 1} 张成功样本 (总进度: {image_files.index(img_path) + 1}/{len(image_files)}): {img_path.name}")

                miou, intact, residual = predict(args, str(img_path), str(current_output_dir),
                                                 pose_extractor, reconstructor, geometric_refiner, image_editor, sam_predictor)


                # 累加分数
                miou_total += miou
                mpjpe_intact_total += intact
                mpjpe_residual_total += residual
                success_count += 1  # 🌟 只有 predict 成功运行后才加 1

                # 计算平均值 (使用 success_count 避免除以零)
                avg_miou = miou_total / success_count
                avg_intact = mpjpe_intact_total / success_count
                avg_residual = mpjpe_residual_total / success_count

                print(f"✅ {img_path.name} 处理完成")
                print(f"📊 当前平均 mIoU: {avg_miou:.4f}")
                print(f"📊 当前平均 MPJPE (Intact): {avg_intact:.2f} px")
                print(f"📊 当前平均 MPJPE (Residual): {avg_residual:.2f} px")
                with open(log_filename, 'a') as log_file:
                    log_file.write(f"{img_path.name} | mIoU: {miou:.4f} | Intact MPJPE: {intact:.2f} | Residual MPJPE: {residual:.2f}\n")
                    log_file.write(f"avg mIoU: {avg_miou:.4f} | avg Intact MPJPE: {avg_intact:.2f} | avg Residual MPJPE: {avg_residual:.2f}\n")
                    log_file.write(f"----------------------------------------\n")
                break

            except Exception as e:
                attempts += 1
                print(f"❌ 处理图片 {img_path.name} 时发生错误: {str(e)}")
                with open(Path(args.output_dir) / "error_log.txt", "a") as f:
                    f.write(f"{img_path.name}: {str(e)}\n")
                raise e

        # 循环结束后打印最终报告
    print("\n" + "=" * 30)
    print(f"🚀 全部处理完成！成功样本数: {success_count}")
    if success_count > 0:
        print(f"🏆 最终平均 mIoU: {miou_total / success_count:.4f}")
        print(f"🏆 最终平均 MPJPE (Intact): {mpjpe_intact_total / success_count:.2f}")
        print(f"🏆 最终平均 MPJPE (Residual): {mpjpe_residual_total / success_count:.2f}")
    print("=" * 30)

if __name__ == "__main__":
    args = parse_args()
    main(args)