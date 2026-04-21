import numpy as np
from PIL import Image
from pathlib import Path


def create_green_overlay(image_dir: Path, mask_dir: Path, output_dir: Path, alpha: float = 0.5):
    """
    读取原图和 Mask，将 Mask 转化为透明绿色并叠加在原图上保存。

    :param image_dir: 填充后的原图目录 (eval_seg_padded)
    :param mask_dir: 生成的 Mask 目录 (inpaint_masks)
    :param output_dir: 可视化结果保存目录
    :param alpha: 绿色的不透明度 (0.0 到 1.0)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 遍历 mask 文件夹里的所有图片
    mask_files = list(mask_dir.glob("*_mask.png"))
    print(f"找到 {len(mask_files)} 个 Mask，开始生成叠加预览图...")

    for mask_path in mask_files:
        # 推导原图的文件名 (比如 0001_mask.png -> 0001.png)
        original_img_name = mask_path.name.replace("_mask", "")
        img_path = image_dir / original_img_name

        if not img_path.exists():
            print(f"找不到对应的原图: {original_img_name}，跳过。")
            continue

        # 1. 读取原图 (RGBA 格式)
        base_img = Image.open(img_path).convert("RGBA")

        # 2. 读取 Mask (转换为灰度图)
        mask_img = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_img)

        # 3. 创建一个全绿色的 RGBA 涂层，尺寸和原图一样
        green_layer_np = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)
        green_layer_np[:, :, 1] = 255  # 绿色通道拉满 (R=0, G=255, B=0)

        # 4. 根据 mask 和设置的透明度，计算 Alpha 通道
        # mask_np > 0 的地方赋予透明度，等于 0 的地方完全透明
        green_alpha = (mask_np > 0) * int(255 * alpha)
        green_layer_np[:, :, 3] = green_alpha

        green_overlay = Image.fromarray(green_layer_np)

        # 5. 将绿色图层叠加到原图上
        result_img = Image.alpha_composite(base_img, green_overlay)

        # 6. 保存结果
        save_path = output_dir / f"overlay_{original_img_name}"
        result_img.save(save_path, "PNG")

    print(f"\n✅ 预览图生成完毕！请去 {output_dir} 检查效果。")


# ==========================================
# 运行
# ==========================================
if __name__ == "__main__":
    IMAGE_DIR = Path('./data/eval_seg_padded')
    MASK_DIR = Path('./data/inpaint_masks')
    OVERLAY_DIR = Path('./data/inpaint_overlay_vis')  # 专门存叠加图的文件夹

    create_green_overlay(IMAGE_DIR, MASK_DIR, OVERLAY_DIR, alpha=0.5)