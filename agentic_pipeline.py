import os
import time
import requests
import base64
import mimetypes
import cv2
import torch
import numpy as np
import trimesh
from pathlib import Path
from PIL import Image
from torchvision import transforms

# 阿里云 DashScope 官方 SDK
import dashscope
from dashscope import MultiModalConversation

# 4D-Humans (HMR 2.0) 核心导入
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT

# --- 0. 全局配置 ---
# 请确保在运行前已设置环境变量: export DASHSCOPE_API_KEY="your_api_key"
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
# 1. 备份原始的 torch.load
_original_load = torch.load

# 2. 写一个“狸猫换太子”的新函数，强行把 weights_only 设为 False
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

# 3. 全局替换！让后面所有第三方库调用的都是我们修改过的 load
torch.load = _patched_load
# ---------------------------------------------------------
# 辅助函数: 图像 Base64 编码
# ---------------------------------------------------------
def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"  # 默认回退类型

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except IOError as e:
        raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")


# ---------------------------------------------------------
# 1. 图像生成/编辑模块 (Aliyun Qwen-image-2.0-pro)
# ---------------------------------------------------------
def generate_able_bodied_image(input_image_path: str, save_path: str) -> str:
    print(f"\n[1/3 API] 正在读取并编码原图: {input_image_path}")

    try:
        base64_image = encode_file(input_image_path)
    except Exception as e:
        print(f"❌ 图片编码失败: {e}")
        return ""

    prompt_text = (
        "Inpaint and edit this image to create a realistic able-bodied person. "
        "Seamlessly generate the missing limb to match the person's pose, anatomy, "
        "clothing style, and lighting."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"image": base64_image},
                {"text": prompt_text}
            ]
        }
    ]

    print("[1/3 API] 正在请求 Aliyun DashScope (Qwen-image-2.0-pro) ...")
    try:
        response = MultiModalConversation.call(
            model="qwen-image-2.0-pro",
            messages=messages,
            stream=False,
            n=1,
            watermark=False,
            prompt_extend=True,
            size="1024*1024",  # 可根据需要修改分辨率
        )

        if response.status_code == 200:
            output_content = response.output.choices[0].message.content
            image_url = output_content[0]['image']
            print(f"✅ 云端生成成功！图片URL: {image_url}")

            print(f"[1/3 API] 正在下载图片至本地: {save_path} ...")
            img_data = requests.get(image_url).content
            with open(save_path, 'wb') as handler:
                handler.write(img_data)

            print(f"✅ 图片已成功保存。")
            return save_path
        else:
            print(f"❌ 接口请求失败！")
            print(f"错误码：{response.code} | 错误信息：{response.message}")
            return ""

    except Exception as e:
        print(f"❌ API 调用或网络错误：{e}")
        return ""


# ---------------------------------------------------------
# 2. 交互式裁剪模块 (OpenCV ROI)
# ---------------------------------------------------------
def get_human_crop(image_path: str, padding_ratio: float = 1.1):
    print(f"\n[2/3 UI] 请在弹出窗口中框选【完整人体】并按空格或回车确认")
    img_cv = cv2.imread(image_path)
    window_name = "4D-Humans: Select Human ROI"

    roi = cv2.selectROI(window_name, img_cv, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

    x, y, w, h = roi
    img_pil = Image.open(image_path).convert('RGB')

    if w == 0 or h == 0:
        print("⚠️ 未画框，自动取中心正方形")
        crop_size = min(img_pil.width, img_pil.height)
        left, top = (img_pil.width - crop_size) / 2, (img_pil.height - crop_size) / 2
        return img_pil.crop((left, top, left + crop_size, top + crop_size))

    center_x, center_y = x + w / 2.0, y + h / 2.0
    max_side = max(w, h) * padding_ratio

    left = max(0, int(center_x - max_side / 2.0))
    top = max(0, int(center_y - max_side / 2.0))
    right = min(img_pil.width, int(center_x + max_side / 2.0))
    bottom = min(img_pil.height, int(center_y + max_side / 2.0))

    img_cropped = img_pil.crop((left, top, right, bottom))
    print(f"✅ 裁剪完成！")
    return img_cropped


# ---------------------------------------------------------
# 3. 3D 重建模块 (4D-Humans 引擎)
# ---------------------------------------------------------
def run_3d_reconstruction(img_cropped: Image.Image, output_dir: str, original_filename: str):
    print(f"\n[3/3 3D] 启动 4D-Humans (HMR 2.0) 引擎...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f"[3/3 3D] 加载 ViT 模型权重到 {device} ...")
    model, _ = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    # HMR 2.0 标准图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_images = transform(img_cropped).unsqueeze(0).to(device)

    print("[3/3 3D] 执行前向推理计算 3D 拓扑...")
    with torch.no_grad():
        batch = {'img': batch_images}
        out = model(batch)
        vertices = out['pred_vertices'][0].cpu().numpy()
        faces = model.smpl.faces

    # 导出 3D 模型
    os.makedirs(output_dir, exist_ok=True)
    mesh_path = os.path.join(output_dir, f"{Path(original_filename).stem}_hmr2_mesh.obj")

    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(mesh_path)

    print(f"✨ 成功! 3D SMPL 模型已保存至: {mesh_path}")
    return mesh_path


# ---------------------------------------------------------
# 执行入口
# ---------------------------------------------------------
if __name__ == "__main__":
    # 配置你的测试文件路径 (请根据实际情况修改)
    RAW_IMG = "./data/residual_examples/000097_png_3.jpg"
    GEN_DIR = "./output/qwen_results"
    MESH_DIR = "./output/hmr2_meshes"

    os.makedirs(GEN_DIR, exist_ok=True)
    os.makedirs(MESH_DIR, exist_ok=True)

    print("=== Agentic 3D Pipeline 启动 ===")

    # 1. 检查环境变量
    if not dashscope.api_key:
        print("❌ 错误: 未检测到 DASHSCOPE_API_KEY，请先配置环境变量。")
        print("例如: export DASHSCOPE_API_KEY='sk-xxxxxxxxxx'")
        exit(1)

    # 2. Qwen 处理生成
    gen_filename = f"able_bodied_{int(time.time())}.jpg"
    target_path = os.path.join(GEN_DIR, gen_filename)

    ready_img_path = generate_able_bodied_image(RAW_IMG, target_path)

    # 3. 3D 重建流程
    if ready_img_path and os.path.exists(ready_img_path):
        # 交互式裁剪
        cropped_pil_image = get_human_crop(ready_img_path)

        # 传入 HMR 2.0 提取 Mesh
        run_3d_reconstruction(cropped_pil_image, MESH_DIR, gen_filename)
    else:
        print("\n❌ 错误: 未能获取到大模型生成的图像，Pipeline 在 3D 重建前终止。")

    print("\n=== Pipeline 结束 ===")