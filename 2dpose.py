import os
import cv2
import numpy as np
from pathlib import Path

import torch

from pose_extractor import PoseExtractor, read_kpts_annotation

_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else _original_load(*args, **kwargs)

def main(ori_image_path, gen_image_path, pose_extractor, annotation_file=None):
    # 1. 读取原图 Ground Truth 标注
    try:
        kpts_orig, _, types_orig = read_kpts_annotation(ori_image_path, annotation_file)
    except Exception as e:
        print(f"读取标注失败: {e}")
        return 0, 0, 0

    # 2. 同步缩小 GT 坐标 (1/3)
    for i in range(len(kpts_orig)):
        kpts_orig[i][0] /= 3.0  # x 坐标缩小
        kpts_orig[i][1] /= 3.0  # y 坐标缩小

    # 3. 读取并缩小原图
    img_ori_temp = cv2.imread(ori_image_path)
    if img_ori_temp is None:
        return 0, 0, 0
    target_h, target_w = img_ori_temp.shape[:2]
    target_h, target_w = int(target_h / 3), int(target_w / 3)

    image_ori_temp = cv2.resize(img_ori_temp, (target_w, target_h), interpolation=cv2.INTER_AREA)
    base_name, ext = os.path.splitext(ori_image_path)
    temp_ori_path = f"{base_name}_resized{ext}"
    cv2.imwrite(temp_ori_path, image_ori_temp)

    # 4. 🌟 直接对缩放后的【原图】进行预测
    try:
        kpts_pred, types_pred = pose_extractor.extract_keypoints(temp_ori_path)
    except Exception as e:
        print(f"Pose extraction failed: {e}")
        return 0, 0, 0

    # 5. 🌟 计算 2D MPJPE (按类型严格过滤)
    intact_errors = []
    residual_errors = []

    # Part A: 计算 0-16 (即 COCO 17 个正常肢体点)
    for i in range(17):
        if types_orig[i] == 0:  # 只有 type=0 才是需要统计的有效点
            gt_pt = np.array(kpts_orig[i][:2])
            pred_pt = np.array(kpts_pred[i][:2])
            dist = np.linalg.norm(gt_pt - pred_pt)
            intact_errors.append(dist)

    # Part B: 计算 23-30 (残肢点)
    for i in range(23, 31):
        if types_orig[i] == 0:  # 只有 type=0 才是有效的残肢标注
            gt_pt = np.array(kpts_orig[i][:2])
            pred_pt = np.array(kpts_pred[i][:2])
            dist = np.linalg.norm(gt_pt - pred_pt)
            residual_errors.append(dist)

    # 计算平均值 (防空列表报错)
    mpjpe_intact = float(np.mean(intact_errors)) if intact_errors else 0.0
    mpjpe_residual = float(np.mean(residual_errors)) if residual_errors else 0.0

    print(f"      -> 完整关节 2D MPJPE: {mpjpe_intact:.2f} px")
    print(f"      -> 残肢端点 2D MPJPE: {mpjpe_residual:.2f} px")

    # 6. 返回结果 (用 0.0 占位 miou，保证外层拆包不报错)
    return 0.0, mpjpe_intact, mpjpe_residual


if __name__ == "__main__":
    pose_extractor = PoseExtractor(
        config_file='./models/pose/vit_config.py',
        checkpoint_file='./models/pose/epoch_1.pth',
        device='cuda:0'
    )
    workdir = Path('./workdir1')

    # 提前转为 list
    dirs = list(workdir.glob('*'))

    miou = 0
    mpjpe_intact = 0
    mpjpe_residual = 0
    valid_count = 0  # 🌟 必须加上有效计数器
    bad_images = []

    for dir_path in dirs:
        image_folder = Path(dir_path)

        all_files = [str(p) for p in image_folder.iterdir() if p.is_file() and p.name != 'final.png']

        if not all_files:
            continue

        all_files.sort()
        gen_image_path = all_files[-1]
        ori_image_path = f'./data/eval_seg_padded/{dir_path.name}.png'
        print(f'start to analyse image {gen_image_path}')

        result = main(ori_image_path, gen_image_path, pose_extractor,
                      annotation_file='./data/filtered_annotations_padded_png.json')