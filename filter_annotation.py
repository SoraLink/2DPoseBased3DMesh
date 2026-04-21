import json
import shutil
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

    # 这里可以保留打印，或者注释掉以免刷屏
    # print(f"[{Path(image_path).name}] 匹配点数: {valid_count}/{len(keypoints)}")
    return valid_count >= threshold


def filter_coco_annotations(image_dir: Path, annotation_path: Path, output_path: Path, failed_dir: Path):
    # 确保失败文件夹存在
    failed_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"找到 {len(image_files)} 张分割图片，开始严格匹配与筛查...")

    for image_path in image_files:
        expected_jpg_name = image_path.with_suffix('.jpg').name
        target_img_info = None

        # 匹配逻辑
        if expected_jpg_name in filename_to_img:
            target_img_info = filename_to_img[expected_jpg_name]
        elif image_path.name in filename_to_img:
            target_img_info = filename_to_img[image_path.name]
        else:
            for coco_fn, info in filename_to_img.items():
                if Path(coco_fn).stem == image_path.stem:
                    target_img_info = info
                    break

        # 校验 1：数据缺失
        if target_img_info is None:
            print(f"[移除] 数据缺失: 在标注中找不到图片 {image_path.name}")
            shutil.move(str(image_path), str(failed_dir / image_path.name))
            continue

        image_id = target_img_info['id']
        anns = img_id_to_anns.get(image_id, [])
        matched_anns_for_this_image = []

        for ann in anns:
            kpts = ann.get('keypoints', [])
            if not kpts:
                continue

            if check_person_match(image_path, kpts, threshold=9):
                matched_anns_for_this_image.append(ann)

        match_count = len(matched_anns_for_this_image)

        # 校验 2 & 3：匹配数量异常 (0个或多个)
        if match_count == 0:
            print(f"[移除] 匹配失败: {image_path.name} 上的有效关键点少于阈值(0个匹配)")
            shutil.move(str(image_path), str(failed_dir / image_path.name))
            continue
        elif match_count > 1:
            print(f"[移除] 匹配异常: {image_path.name} 同时包含了 {match_count} 个人物(多个匹配)")
            shutil.move(str(image_path), str(failed_dir / image_path.name))
            continue

        # 校验 4：Keypoint Types 异常
        if 0 not in matched_anns_for_this_image[0]['keypoint_types'][23:31]:
            print(f"[移除] 类型异常: {image_path.name} 索引 23 到 30 之间没有包含 0")
            shutil.move(str(image_path), str(failed_dir / image_path.name))
            continue

        # 走到这里，说明所有校验通过，图片完美符合要求
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

    print(f"\n✅ 筛选完成！")
    print(f"总计检查图片: {len(image_files)}")
    print(f"成功筛选保留: {len(new_images)}")
    print(f"失败并被移走: {len(image_files) - len(new_images)}")
    print(f"纯净标注文件已保存至: {output_path}")
    print(f"异常图片已移动至: {failed_dir}")

# ==========================================
# 运行
# ==========================================
if __name__ == "__main__":
    # 输入路径
    SEG_IMAGE_DIR = Path('./data/eval_seg')
    RAW_ANN_PATH = Path('./data/train_final.json')

    # 输出路径
    OUTPUT_ANN_PATH = Path('./data/filtered_annotations.json')
    # 失败图片存放路径 (避免跟原目录冲突)
    FAILED_IMAGE_DIR = Path('./data/invalid_image')

    if not SEG_IMAGE_DIR.exists():
        print(f"错误: 目录 {SEG_IMAGE_DIR} 不存在")
    else:
        filter_coco_annotations(SEG_IMAGE_DIR, RAW_ANN_PATH, OUTPUT_ANN_PATH, FAILED_IMAGE_DIR)