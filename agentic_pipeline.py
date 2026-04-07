import os
import time
import io
import mimetypes
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from google import genai
from google.genai import types

# === 配置区域 ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
PROJECT_ID = "project-fe4de98f-5478-4cee-b84"
LOCATION = "global"
MODEL_ID = 'gemini-2.5-flash-image'

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
import trimesh

def get_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type if mime_type else "image/jpeg"


# ---------------------------------------------------------
# 第一部分：大模型 API 交互模块 (整合 Vertex AI)
# ---------------------------------------------------------
def generate_able_bodied_image(client, input_image_path: str, save_path: str) -> str:
    """
    调用 Gemini API，将残疾人图片补全为健全人图片。
    包含完整的容错和重试机制。
    """
    print(f"\n[API] 正在读取原图: {input_image_path}")

    try:
        with open(input_image_path, 'rb') as f:
            source_bytes = f.read()
        source_mime = get_mime_type(input_image_path)
    except Exception as e:
        print(f"[API] ❌ 读取图片文件失败: {e}")
        return ""

    # 这里的 Prompt 已修改为适配“生成健全人”任务
    contents_list = [
        types.Part.from_bytes(data=source_bytes, mime_type=source_mime),
        "Inpaint and edit this image to create a realistic able-bodied person. "
        "Seamlessly generate the missing limb to match the person's pose, anatomy, clothing style, and lighting. "
        "Ensure the generated limb looks completely natural and anatomically correct."
    ]

    safety_config = [
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
    ]

    print("[API] 正在向 Gemini 发送图像生成请求...")
    attempt_count = 0
    while True:
        attempt_count += 1
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents_list,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    safety_settings=safety_config,
                    response_modalities=['IMAGE'],
                )
            )

            if response.candidates and response.candidates[0].content.parts:
                image_found = False
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        image_bytes = part.inline_data.data
                        image_io = io.BytesIO(image_bytes)
                        final_image = Image.open(image_io)

                        final_image.save(save_path, format="JPEG", quality=95)
                        print(f"[API] ✅ 生成成功! 健全人图片已保存至: {os.path.basename(save_path)}")

                        image_found = True
                        break

                if not image_found:
                    print(f"[API] ⚠️ 生成完成但无图片。Finish Reason: {response.candidates[0].finish_reason}")

                break  # 成功执行或明确无图，跳出重试循环

            else:
                print("[API] ❌ API 返回空内容。")
                break

        except Exception as e:
            print(f"[API] ❌ 发生错误: {e}")
            print(f"[API] ⚠️ 正在等待 10 秒后进行第 {attempt_count + 1} 次重试...")
            time.sleep(10)
            continue

    return save_path


# ---------------------------------------------------------
# 第二部分：UI 手动框选与预处理
# ---------------------------------------------------------
def manual_crop_to_square(image_path: str, padding_ratio: float = 1.1):
    print(f"\n[UI交互] 准备显示图像: {os.path.basename(image_path)}")
    print("👉 操作指南:")
    print("   1. 鼠标左键拖拽，框住【整个人体】")
    print("   2. 画错可以重新拖拽")
    print("   3. 框好后按【SPACE】或【ENTER】确认提取")

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError(f"OpenCV 无法读取图片: {image_path}")

    window_name = "Draw Bounding Box (Press SPACE/ENTER to confirm)"
    roi = cv2.selectROI(window_name, img_cv, showCrosshair=True, fromCenter=False)

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

    x, y, w, h = roi
    img_pil = Image.open(image_path).convert('RGB')

    if w == 0 or h == 0:
        print("[UI交互] ⚠️ 未检测到有效框！将默认裁剪全图中心。")
        crop_size = min(img_pil.width, img_pil.height)
        left = (img_pil.width - crop_size) / 2
        top = (img_pil.height - crop_size) / 2
        return img_pil.crop((left, top, left + crop_size, top + crop_size))

    center_x = x + w / 2.0
    center_y = y + h / 2.0
    max_side = max(w, h) * padding_ratio

    left = max(0, int(center_x - max_side / 2.0))
    top = max(0, int(center_y - max_side / 2.0))
    right = min(img_pil.width, int(center_x + max_side / 2.0))
    bottom = min(img_pil.height, int(center_y + max_side / 2.0))

    img_cropped = img_pil.crop((left, top, right, bottom))
    print(f"[UI交互] ✅ 裁剪完成！正方形区域: ({left}, {top}) -> ({right}, {bottom})")

    return img_cropped


# ---------------------------------------------------------
# 第三部分：3D 重建模块 (目前为调试模式，等待接入真实环境)
# ---------------------------------------------------------
def reconstruct_3d_mesh(image_path: str, output_dir: str):
    print(f"\n[3D] 开始处理补全后的图像: {os.path.basename(image_path)}")

    # 1. 获取裁剪后的图像
    img_cropped = manual_crop_to_square(image_path, padding_ratio=1.1)

    # 2. HMR 2.0 标准预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("[3D] 图像预处理完毕。")
    print("[3D] ⚠️ 当前处于调试模式，尚未执行真实 HMR 2.0 前向传播。")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()
    batch_images = transform(img_cropped).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(batch_images)
        pred_smpl_params = out['pred_smpl_params']
        pred_cam = out['pred_cam']
        vertices = out['pred_vertices'][0].cpu().numpy()

    print("[3D] SMPL 参数提取成功，正在生成 3D Mesh (.obj)...")
    faces = model.smpl.faces
    mesh_save_path = os.path.join(output_dir, "able_bodied_mesh.obj")
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(mesh_save_path)
    print(f"[3D] 🎉 3D Mesh 已成功保存至: {mesh_save_path}")
    return mesh_save_path


# ---------------------------------------------------------
# 主流程 Pipeline
# ---------------------------------------------------------
if __name__ == "__main__":
    # 配置你的测试文件路径
    INPUT_AMPUTEE_IMG = "./data/test_img.jpg"  # 替换为你的测试图片
    GENERATED_IMG_DIR = "./output/generated_images"
    MESH_OUTPUT_DIR = "./output/3d_meshes"

    os.makedirs(GENERATED_IMG_DIR, exist_ok=True)
    os.makedirs(MESH_OUTPUT_DIR, exist_ok=True)

    print("=== Pipeline 启动 ===")

    # 1. 初始化 Vertex AI Client
    try:
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    except Exception as e:
        print(f"Client 初始化失败: {e}")
        exit()

    # 2. 补全肢体生成图片
    temp_generated_path = os.path.join(GENERATED_IMG_DIR, "able_bodied_output.jpg")
    able_bodied_img_path = generate_able_bodied_image(
        client=client,
        input_image_path=INPUT_AMPUTEE_IMG,
        save_path=temp_generated_path
    )

    # 3. 进入手工裁剪与 3D 流程
    if able_bodied_img_path and os.path.exists(able_bodied_img_path):
        reconstruct_3d_mesh(
            image_path=able_bodied_img_path,
            output_dir=MESH_OUTPUT_DIR
        )
    else:
        print("错误: 未能获取到大模型生成的图像，Pipeline 终止。")

    print("=== Pipeline 结束 ===")