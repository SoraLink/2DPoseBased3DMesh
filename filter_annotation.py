import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List


def check_person_match(image_path: Union[str, Path], keypoints: Union[List, np.ndarray], threshold: int = 9) -> bool:
    """
    判断关键点是否属于图片中被分割出的人。
    """
    try:
        image = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"无法读取图片 {image_path}: {e}")
        return False

    image_array = np.array(image)
    mask = image_array[:, :, 3] > 0
    height, width = mask.shape

    valid_count = 0

    # 转换 COCO 格式 [x,y,v, x,y,v...] 为 [[x,y,v], [x,y,v]...]
    if isinstance(keypoints, list) and len(keypoints) > 0 and not isinstance(keypoints[0], (list, np.ndarray)):
        keypoints = np.array(keypoints).reshape(-1, 3)

    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        conf = kp[2] if len(kp) > 2 else 1

        if conf == 0:  # 置信度为0表示点不存在
            continue

        if 0 <= x < width and 0 <= y < height:
            if mask[y, x]:
                valid_count += 1

    print(f"[{Path(image_path).name}] 匹配点数: {valid_count}/{len(keypoints)}")
    return valid_count >= threshold


def filter_coco_annotations(image_dir: Path, annotation_path: Path, output_path: Path):
    # 1. 加载 COCO 数据
    print(f"正在加载标注文件: {annotation_path}...")
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # 建立索引以加速查找
    filename_to_img = {img['file_name']: img for img in coco_data['images']}
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    new_images = []
    new_annotations = []

    # 2. 遍历分割后的图片目录
    image_files = list(image_dir.glob("*.png"))
    print(f"找到 {len(image_files)} 张分割图片，开始严格匹配...")

    for image_path in image_files:
        # 使用 pathlib 直接将后缀替换为 .jpg
        expected_jpg_name = image_path.with_suffix('.jpg').name

        target_img_info = None

        # 1. 优先精确匹配 .jpg 的文件名 (速度最快)
        if expected_jpg_name in filename_to_img:
            target_img_info = filename_to_img[expected_jpg_name]
        # 2. 防御性逻辑：万一 COCO 里面本身记录的就是 .png
        elif image_path.name in filename_to_img:
            target_img_info = filename_to_img[image_path.name]
        # 3. 兜底逻辑：处理其他未知后缀（比如 .jpeg 或 .JPG）
        else:
            for coco_fn, info in filename_to_img.items():
                if Path(coco_fn).stem == image_path.stem:
                    target_img_info = info
                    break

        if target_img_info is None:
            raise ValueError(f"数据缺失: 在 COCO 标注中完全找不到图片 {image_path.name} (或对应的jpg) 的记录！")

        image_id = target_img_info['id']
        anns = img_id_to_anns.get(image_id, [])

        # 临时存储当前图片匹配成功的标注
        matched_anns_for_this_image = []

        for ann in anns:
            kpts = ann.get('keypoints', [])
            if not kpts:
                continue

            if check_person_match(image_path, kpts, threshold=9):
                matched_anns_for_this_image.append(ann)

        # ==========================================
        # 严格校验匹配数量
        # ==========================================
        match_count = len(matched_anns_for_this_image)

        if match_count == 0:
            raise ValueError(f"匹配失败 [0个匹配]: 分割图 {image_path.name} 上的有效关键点少于阈值，未能对应任何标注！")
        elif match_count > 1:
            raise ValueError(
                f"匹配异常 [多个匹配]: 分割图 {image_path.name} 错误地同时包含了 {match_count} 个人物的关键点！")

        assert 0 in matched_anns_for_this_image[0]['keypoint_types'][
            23:31], f"错误: {image_path.name} 索引 23 到 30 之间没有包含 0"

        # 走到这里，说明 match_count == 1，符合绝对预期
        new_annotations.append(matched_anns_for_this_image[0])
        new_images.append(target_img_info)

    # 3. 构造并保存新的 COCO 文件
    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco_data.get('categories', []),
        "info": coco_data.get('info', {}),
        "licenses": coco_data.get('licenses', [])
    }

    with open(output_path, 'w') as f:
        json.dump(new_coco, f, indent=4)

    print(f"\n✅ 筛选与严格校验通过！")
    print(f"原始图片数: {len(coco_data['images'])} -> 筛选后: {len(new_images)}")
    print(f"原始标注数: {len(coco_data['annotations'])} -> 筛选后: {len(new_annotations)}")
    print(f"纯净的标注文件已保存至: {output_path}")

# ==========================================
# 运行
# ==========================================
if __name__ == "__main__":
    # 输入路径
    SEG_IMAGE_DIR = Path('./data/eval_seg')
    RAW_ANN_PATH = Path('./data/train_final.json')

    # 输出路径
    OUTPUT_ANN_PATH = Path('./data/filtered_annotations.json')

    if not SEG_IMAGE_DIR.exists():
        print(f"错误: 目录 {SEG_IMAGE_DIR} 不存在")
    else:
        filter_coco_annotations(SEG_IMAGE_DIR, RAW_ANN_PATH, OUTPUT_ANN_PATH)