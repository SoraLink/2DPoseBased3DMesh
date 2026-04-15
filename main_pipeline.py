import os

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
from sam2_coco import segment_subject

dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def parse_args():
    parser = argparse.ArgumentParser(description="🚀 Neuro-Symbolic Agentic 3D Pipeline")

    # 基础输入参数
    parser.add_argument('--img', type=str,
                        help='Path to the input image')

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
    parser.add_argument('--image_save_dir', type=str)
    parser.add_argument('--mesh_save_dir', type=str)

    return parser.parse_args()


def main(args):
    print("====== 🚀 Neuro-Symbolic Agentic 3D Pipeline ======")

    if not os.path.exists(args.img):
        print(f"❌ 找不到输入图片: {args.img}")
        return

    # 1. 初始化所有类 (传入 argparse 解析好的参数)
    print("\n[Init] 加载核心模块...")
    pose_extractor = PoseExtractor(config_file=args.pose_config,
                                   checkpoint_file=args.pose_ckpt,
                                   device=args.device)

    reconstructor = ReconstructionEngine()
    image_editor = AgenticImageEditor()
    geometric_refiner = GeometricRefinerAgent(pose_extractor)

    print(f"\n[Processing] 开始处理图像: {args.img}")
    kpts_orig, types_orig = read_kpts_annotation(args.img, args.annotation_file)
    sam2_img_path = segment_subject(args.img, kpts_orig)

    image_url = OSSProcessor().upload_and_get_url(local_file_path=sam2_img_path)

    # ==========================================
    # 4. Agentic Loop (生成 -> Critic -> 再生成)
    # ==========================================
    save_dir = args.image_save_dir
    if not isinstance(save_dir, str):
        save_dir = f"./workdir/output_{int(time.time())}"
    generated_image_urls = image_editor.run(image_url)
    last_generated_image_url = generated_image_urls[-1]
    save_image_from_url(generated_image_urls, "image_editor", save_dir)
    generated_image_urls = geometric_refiner.run(kpts_orig, last_generated_image_url)
    save_image_from_url(generated_image_urls, "geometric_refiner", save_dir)
    final_image_url = generated_image_urls[-1]

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
        # 注意: 这里的 args.img 可能依然是你最开始的本地路径，用于提取基础文件名
        base_name = os.path.basename(args.img).split('.')[0]
        mesh_save_path = os.path.join(args.mesh_save_dir, f"{base_name}_mesh.obj")

        mesh_save_path, pred_joints_3d = reconstructor.predict_mesh(img_pil, mesh_save_path)
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
        M_inv = get_final_calibration_matrix(kpts_orig, kpts_gen=kpts_gen)  # 这里暂时传 None，后续可以改成实际生成图的关键点
        # 遍历 METAINFO 中的最后 8 个残肢点 (ID 23 到 30)
        for i in range(23, 31):
            # 判断: 只有 type == 0 才是有效残肢点，且确保坐标数组够长
            if types_orig[i] == 0:
                res_name = METAINFO['keypoint_info'][i]['name']

                # 获取 2D 坐标 (假设 kpts_orig 的格式是 [x, y, conf])
                pt_2d = kpts_orig[i][0:2]
                # 查表找到对应的 3D 骨骼起点和终点
                if res_name in RES_BONE_MAPPING:
                    start_joint_name, end_joint_name = RES_BONE_MAPPING[res_name]

                    start_3d = pred_joints_3d[start_joint_name]
                    end_3d = pred_joints_3d[end_joint_name]

                    # 组装切割任务
                    cut_tasks.append({
                        'name': res_name,
                        'pt_2d': pt_2d,
                        'start_3d': start_3d,
                        'end_3d': end_3d
                    })

        # 如果收集到了切割任务，才去执行截断
        if cut_tasks:
            h, w = img_bgr.shape[:2]
            # 初始化截肢器 (焦距 5000 根据 HMR/CLIFF 默认设置)
            mesh_cutter = ResidualMeshCutter(focal_length=5000.0, img_center=(w / 2, h / 2))

            mesh = mesh_cutter.process_multiple_cuts(
                mesh_path=mesh_save_path,
                cut_tasks=cut_tasks,
                M_inv=M_inv,
            )
        else:
            raise ValueError("No residual bone cutting tasks found.")

        project_mesh_overlay(args.img, mesh, M_inv)  # 将最终 Mesh 投影回原图坐标系，生成 Overlay

    except Exception as e:
        raise e


def project_mesh_overlay(image_path, mesh, M_inv, focal_length=5000.0, hmr_size=(256, 256)):
    """
    将 3D Mesh (HMR space, e.g., SMPL) 精准投影回原始图片坐标系，生成绿色 Overlay。
    :param image_path: 原始图片路径 (incomplete)
    :param mesh: 切割完成后的 trimesh 对象
    :param M_inv: 2x3 矩阵 (原始坐标 -> HMR 256 坐标的校准矩阵)
    :param focal_length: HMR 内部焦距 (5000.0)
    :param hmr_size: HMR 内部画布 (256, 256)
    """
    # ================================================================
    # 第一步：计算反向仿射矩阵 (HMR 256 -> 原始坐标系)
    # ================================================================
    # 我们需要的是从 [生成图坐标] 映射回 [原始坐标] 的变换 (dst -> src)
    if M_inv.shape != (2, 3):
        raise ValueError("M_inv parameters must be a 2x3 matrix.")

    # 🌟 关键：对齐工程细节。将 2x3 扩充为 3x3，然后求逆，再取回 2x3
    M_inv_augmented = np.vstack([M_inv, [0, 0, 1]])  # 变为 3x3 齐次矩阵
    try:
        M_hmr_to_orig_3x3 = np.linalg.inv(M_inv_augmented)  # 完美求逆
        M_hmr_to_orig = M_hmr_to_orig_3x3[:2, :]  # 变回 2x3 用于 warpAffine
    except np.linalg.LinAlgError:
        raise ValueError("Matrix inversion failed. M_inv calculation is incorrect.")

    # ================================================================
    # 第二步：将 3D Mesh 投影到 HMR 虚拟画布 (3D -> 2D 256x256)
    # ================================================================
    print("[Visualization] 正在将 3D 拓扑投影至 HMR 画布...")
    fx, fy = focal_length, focal_length
    cx, cy = hmr_size[0] / 2.0, hmr_size[1] / 2.0

    vertices = mesh.vertices
    projected_vertices = np.zeros((len(vertices), 2), dtype=np.int32)
    # 默认设在画布外远端
    projected_vertices[:] = -1e6

    # 执行 3D -> 2D 投影，只处理相机前方的点
    for i, vert in enumerate(vertices):
        if vert[2] > 0.0:
            X, Y, Z = vert[0], vert[1], vert[2]
            projected_vertices[i, 0] = int(fx * (X / Z) + cx)
            projected_vertices[i, 1] = int(fy * (Y / Z) + cy)

    # 创建 HMR 画布的二值遮罩 (256x256)
    mask_256 = np.zeros(hmr_size, dtype=np.uint8)

    # 将 Faces 的顶点索引映射到投影后的 2D 坐标
    faces_int = mesh.faces.astype(np.int32)

    # 🌟 优化：只绘制那些三个顶点都在相机前方的面，防止虚边
    valid_faces_mask = (projected_vertices[faces_int] > -1e5).all(axis=2).all(axis=1)
    projected_faces = projected_vertices[faces_int[valid_faces_mask]]

    # 暴力填充面，生成完整的 segmentation mask
    cv2.fillPoly(mask_256, projected_faces, 255)
    cv2.imwrite("debug_1_mask_256.jpg", mask_256)
    print(f"DEBUG: 256掩码已保存，白色像素点数: {np.sum(mask_256 > 0)}")
    # ================================================================
    # 第三步：将 Mask 变形对齐原图，并进行 Alpha 混合
    # ================================================================
    print("[Visualization] 正在坐标反向补偿并混合 Overlay...")

    # 1. 读取原图
    image_data = np.fromfile(image_path, dtype=np.uint8)
    original_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    h_orig, w_orig = original_image.shape[:2]

    # 2. 🌟 关键：利用 M_hmr_to_orig 将 256 Mask 变形到原图尺寸和位置
    # 这一步将大模型生成的肢体误差完美消化。
    mask_orig = cv2.warpAffine(mask_256, M_hmr_to_orig, (w_orig, h_orig), flags=cv2.INTER_LINEAR)
    cv2.imwrite("debug_2_mask_orig.jpg", mask_orig)
    print(f"DEBUG: 变换后掩码已保存，白色像素点数: {np.sum(mask_orig > 0)}")
    # 3. 创建彩色 Overlay (绿色)
    overlay_color = np.zeros_like(original_image)
    overlay_color[:] = [0, 255, 0]  # 纯绿，可以改为你喜欢的任何颜色

    # 4. 混合 Overlay (Alpha 混合)
    output_image = original_image.copy()
    alpha = 0.5  # 透明度

    # 只有 mask 覆盖的区域进行混合
    mask_bool = (mask_orig > 127)
    output_image = cv2.addWeighted(overlay_color, alpha, output_image, 1 - alpha, 0, dst=output_image)
    # 将 mask 之外的区域变回原图
    output_image[~mask_bool] = original_image[~mask_bool]

    # ================================================================
    # 第四步：保存高质量结果
    # ================================================================
    output_path = image_path.replace(".jpg", "_projection.jpg").replace(".png", "_projection.jpg")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    is_success, im_buf_arr = cv2.imencode(".jpg", output_image, encode_param)

    if is_success:
        im_buf_arr.tofile(output_path)
        print(f"✅ [Projector] 几何闭环可视化已生成: {output_path}")
        return output_path
    else:
        raise ValueError("图像保存失败")

def get_final_calibration_matrix(kpts_orig, kpts_gen):
    """
    kpts_orig: 原始 1024 图像的关键点
    kpts_gen:  大模型生成 1024 图像后的关键点 (重新检测得到的)
    """
    # 选取肩膀和髋部点 (COCO: 5,6,11,12)
    indices = [5, 6, 11, 12]
    src = kpts_orig[indices, :2]
    dst = kpts_gen[indices, :2]

    # 1. 计算【原始 -> 生成图】的仿射校准
    M_calib, _ = cv2.estimateAffinePartial2D(src, dst)

    # 2. 叠加【生成图(比如1024) -> HMR(256)】的缩放
    scale_factor = 256.0 / 1024.0  # 假设你生成图是 1024
    S = np.array([[scale_factor, 0, 0],
                  [0, scale_factor, 0]], dtype=np.float32)

    # 最终矩阵: P_256 = S * [M_calib | 1] * P_orig
    # 这里简单处理：先做校准变换，再做缩放
    M_hmr_inv = S @ np.vstack([M_calib, [0, 0, 1]])

    return M_hmr_inv

def save_image_from_url(urls, source, save_dir):
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

        except requests.exceptions.RequestException as e:
            print(f"❌ 下载失败: {e}")

if __name__ == "__main__":
    args = parse_args()
    main(args)