import json
from pathlib import Path
from PIL import Image


def pad_images_and_annotations(
        input_img_dir: Path,
        input_ann_path: Path,
        output_img_dir: Path,
        output_ann_path: Path,
        pad_ratio: float = 0.3
):
    """
    对图片进行 Padding 扩展，并同步更新 COCO 标注中的坐标数据。
    """
    # 确保输出目录存在
    output_img_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在加载筛选后的标注文件: {input_ann_path}...")
    with open(input_ann_path, 'r') as f:
        coco_data = json.load(f)

    # 用来记录每张图具体的 pad_x 和 pad_y，方便后面更新 Annotation
    img_id_to_pads = {}

    # ==========================================
    # 1. 处理 Images 并计算坐标偏移量
    # ==========================================
    print(f"开始处理图片，Padding 比例: {pad_ratio}...")
    valid_images = []

    for img_info in coco_data['images']:
        # 由于我们之前把所有图存为了 .png，所以需要把 COCO 里的文件名后缀强行转为 .png 去读取
        png_name = Path(img_info['file_name']).with_suffix('.png').name
        img_path = input_img_dir / png_name

        if not img_path.exists():
            print(f"警告: 找不到图片 {img_path.name}，跳过该图。")
            continue

        # 读取原图
        image = Image.open(img_path).convert("RGBA")
        orig_w, orig_h = image.size

        # 计算 Padding 像素数 (上下左右都加)
        pad_x = int(orig_w * pad_ratio)
        pad_y = int(orig_h * pad_ratio)

        # 计算新的画布大小
        new_w = orig_w + 2 * pad_x
        new_h = orig_h + 2 * pad_y

        # 创建新的全透明背景画布
        new_image = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
        # 将原图粘贴到中心 (带上 Alpha 通道遮罩)
        new_image.paste(image, (pad_x, pad_y), image)

        # 保存新的图片
        out_img_path = output_img_dir / png_name
        new_image.save(out_img_path, "PNG")

        # 更新 COCO 数据中该图片的信息
        img_info['width'] = new_w
        img_info['height'] = new_h
        # 如果你希望输出的文件名也保持 png，可以取消下面这行的注释
        # img_info['file_name'] = png_name

        valid_images.append(img_info)
        img_id_to_pads[img_info['id']] = (pad_x, pad_y)

    # 覆盖原来的 images 列表 (剔除可能因为文件丢失而没处理的图)
    coco_data['images'] = valid_images

    # ==========================================
    # 2. 同步处理 Annotations
    # ==========================================
    print("开始更新标注坐标 (Keypoints, Bbox, Segmentation)...")
    valid_annotations = []

    for ann in coco_data['annotations']:
        img_id = ann['image_id']

        # 如果这张图片刚才没被成功处理，就跳过它的标注
        if img_id not in img_id_to_pads:
            continue

        pad_x, pad_y = img_id_to_pads[img_id]

        # --- 更新 Keypoints [x, y, v, ...] ---
        if 'keypoints' in ann:
            kpts = ann['keypoints']
            for i in range(0, len(kpts), 3):
                # 只有当可见度 (v) 大于 0 时，才去加上偏移量。0 表示该点不存在。
                if kpts[i + 2] > 0:
                    kpts[i] += pad_x
                    kpts[i + 1] += pad_y

        # --- 更新 Bounding Box [x, y, width, height] ---
        # Bbox 的 x, y 是左上角坐标，所以也需要偏移，宽高不变
        if 'bbox' in ann:
            bbox = ann['bbox']
            bbox[0] += pad_x
            bbox[1] += pad_y

        # --- 更新 Segmentation (如果是多边形列表) ---
        if 'segmentation' in ann and isinstance(ann['segmentation'], list):
            for poly in ann['segmentation']:
                # poly 是 [x1, y1, x2, y2...] 的形式
                for i in range(0, len(poly), 2):
                    poly[i] += pad_x
                    poly[i + 1] += pad_y

        valid_annotations.append(ann)

    coco_data['annotations'] = valid_annotations

    # ==========================================
    # 3. 保存新的 COCO JSON
    # ==========================================
    with open(output_ann_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"\n✅ Padding 及坐标修正完成！")
    print(f"处理并保存了 {len(valid_images)} 张图片")
    print(f"修正了 {len(valid_annotations)} 条标注")
    print(f"新图片目录: {output_img_dir}")
    print(f"新标注文件: {output_ann_path}")


# ==========================================
# 运行
# ==========================================
if __name__ == "__main__":
    # 【输入】上一步处理完的成功数据
    INPUT_IMAGE_DIR = Path('./data/eval_seg')
    INPUT_ANN_PATH = Path('./data/filtered_annotations.json')

    # 【输出】经过 Padding 处理后的新数据
    OUTPUT_IMAGE_DIR = Path('./data/eval_seg_padded')
    OUTPUT_ANN_PATH = Path('./data/filtered_annotations_padded.json')

    if not INPUT_IMAGE_DIR.exists() or not INPUT_ANN_PATH.exists():
        print(f"错误: 找不到输入目录或标注文件，请检查路径。")
    else:
        # pad_ratio=0.3 意味着单边增加 30% 宽/高
        pad_images_and_annotations(
            input_img_dir=INPUT_IMAGE_DIR,
            input_ann_path=INPUT_ANN_PATH,
            output_img_dir=OUTPUT_IMAGE_DIR,
            output_ann_path=OUTPUT_ANN_PATH,
            pad_ratio=0.3
        )