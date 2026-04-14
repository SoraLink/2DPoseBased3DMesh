import os

import numpy as np
import requests
import time
import cv2
import argparse
from PIL import Image

from pose_extractor import PoseExtractor, read_kpts_annotation
from image_ops import ImageProcessor
from agentic_critic import AgenticImageEditor, GeometricRefinerAgent
from reconstruction_3d import ReconstructionEngine
from sam2_coco import segment_subject


def parse_args():
    parser = argparse.ArgumentParser(description="🚀 Neuro-Symbolic Agentic 3D Pipeline")

    # 基础输入参数
    parser.add_argument('--img', type=str,
                        help='Path to the input image')

    # PoseExtractor 参数 (必填或提供默认路径)
    parser.add_argument('--pose_config', type=str,
                        default='.models/pose/vit_config.py',
                        help='Path to the LDPose MMPose config file')
    parser.add_argument('--pose_ckpt', type=str,
                        default='.models/pose/epoch_1.pth',
                        help='Path to the LDPose checkpoint file')
    parser.add_argument('--device', type=str,
                        default='cuda:0',
                        help='Device to run pose extraction on (e.g., cuda:0 or cpu)')
    parser.add_argument('--annotation_file', type=str,
                        default='./data/ldpose_train_25kpts.json',
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
    kpts_orig = read_kpts_annotation(args.img, args.annotation_file)
    sam2_img_path = segment_subject(args.img, kpts_orig)

    base64_img = ImageProcessor.encode_to_base64(sam2_img_path)

    # ==========================================
    # 4. Agentic Loop (生成 -> Critic -> 再生成)
    # ==========================================
    save_dir = args.image_save_dir
    if not os.path.exists(save_dir):
        save_dir = time.time()
    generated_image_urls = image_editor.run(base64_img)
    last_generated_image_url = generated_image_urls[-1]
    save_image_from_url(generated_image_urls, "image_editor", save_dir)
    generated_image_urls = geometric_refiner.run(base64_img, last_generated_image_url)
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

        reconstructor.predict_mesh(img_pil, mesh_save_path)
        print(f"✅ 3D Mesh 已保存至: {mesh_save_path}")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ 错误: 无法下载网络图片。原因: {e}")
    except Exception as e:
        print(f"\n❌ 错误: 3D 恢复过程发生异常。原因: {e}")



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
            return save_path

        except requests.exceptions.RequestException as e:
            print(f"❌ 下载失败: {e}")
            return None

if __name__ == "__main__":
    args = parse_args()
    main(args)