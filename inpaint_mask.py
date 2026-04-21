import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image

# ==========================================
# 1. 定义残肢点及其拓扑关系 (Kinematic Tree)
# ==========================================
# 字典结构: 残肢点ID : {'parent': 上游关节ID, 'downstream': [下游所有可能的关节ID]}
RESIDUAL_KINEMATICS = {
    # --- 上肢 ---
    23: {'parent': 5, 'downstream': [7, 9, 17]},  # L-Elbow-Above (左大臂残)
    24: {'parent': 6, 'downstream': [8, 10, 18]},  # R-Elbow-Above (右大臂残)
    25: {'parent': 7, 'downstream': [9, 17]},  # L-Elbow-Below (左小臂残)
    26: {'parent': 8, 'downstream': [10, 18]},  # R-Elbow-Below (右小臂残)

    # --- 下肢 ---
    27: {'parent': 11, 'downstream': [13, 15, 19, 21]},  # L-Knee-Above (左大腿残)
    28: {'parent': 12, 'downstream': [14, 16, 20, 22]},  # R-Knee-Above (右大腿残)
    29: {'parent': 13, 'downstream': [15, 19, 21]},  # L-Knee-Below (左小腿残)
    30: {'parent': 14, 'downstream': [16, 20, 22]},  # R-Knee-Below (右小腿残)
}


def generate_inpaint_mask(image_shape, keypoints, keypoint_types, bbox):
    """
    生成用于 Inpainting 的 Mask
    :param image_shape: (height, width)
    :param keypoints: 扁平化的关键点列表 [x1,y1,v1, x2,y2,v2...]
    :param keypoint_types: 标注类型列表，1 表示假肢，0 表示正常/残肢，2 表示忽略
    :param bbox: [x, y, width, height] 用于动态计算肢体粗细
    :return: Numpy 数组的 Mask (0为背景，255为需要Inpaint的区域)
    """
    height, width = image_shape
    # 创建纯黑背景 Mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # 格式化 kpts
    kpts = np.array(keypoints).reshape(-1, 3)

    # 动态计算肢体宽度基准
    box_w, box_h = bbox[2], bbox[3]
    limb_width = int(max(box_w, box_h) * 0.08)
    limb_width = max(limb_width, 10)

    # 遍历所有可能的残肢点
    for res_idx, info in RESIDUAL_KINEMATICS.items():
        rx, ry, rv = kpts[res_idx]

        # ==========================================
        # [修改 1]：如果残肢点本身不可见，或者 type 为 2 (忽略)，直接跳过不画
        # ==========================================
        if rv == 0 or keypoint_types[res_idx] == 2:
            continue

        rx, ry = int(rx), int(ry)

        # 检查其下游是否连接了“假肢”
        prosthetic_points = []
        for down_idx in info['downstream']:
            dx, dy, dv = kpts[down_idx]
            # 这里原本的逻辑是严格等于 1 (假肢) 才画，所以 type 为 2 的天然就被挡在外面了
            if dv > 0 and keypoint_types[down_idx] == 1:
                prosthetic_points.append((int(dx), int(dy)))

        # 逻辑分支 1：存在假肢
        if len(prosthetic_points) > 0:
            path_points = [(rx, ry)] + prosthetic_points

            for i in range(len(path_points) - 1):
                p1, p2 = path_points[i], path_points[i + 1]
                cv2.line(mask, p1, p2, color=255, thickness=limb_width)
                cv2.circle(mask, p1, limb_width // 2, 255, -1)
                cv2.circle(mask, p2, limb_width // 2, 255, -1)

        # 逻辑分支 2：没有假肢 (纯残肢肉包)
        else:
            parent_idx = info['parent']
            px, py, pv = kpts[parent_idx]

            # ==========================================
            # [修改 2]：如果父节点可见，且 type 不为 2，才用它计算延伸方向！
            # 之前那条长到天际的绿线，大概率是因为用了一个错误的 parent 坐标算向量
            # ==========================================
            if pv > 0 and keypoint_types[parent_idx] != 2:
                # 获取从父节点指向残肢点的向量
                vx, vy = rx - px, ry - py
                length = np.sqrt(vx ** 2 + vy ** 2)

                if length > 0:
                    # 向量归一化
                    vx, vy = vx / length, vy / length
                    # 沿着方向向外延伸
                    extend_x = int(rx + vx * limb_width)
                    extend_y = int(ry + vy * limb_width)

                    # 画向外延伸的长条形肉包
                    cv2.line(mask, (rx, ry), (extend_x, extend_y), color=255, thickness=limb_width)
                    cv2.circle(mask, (rx, ry), limb_width // 2, 255, -1)
                    cv2.circle(mask, (extend_x, extend_y), int(limb_width * 0.6), 255, -1)
            else:
                # 兜底情况：父节点无效或是被标注为 2，不计算延伸，直接在残肢点原地画个肉包
                cv2.circle(mask, (rx, ry), int(limb_width * 0.8), 255, -1)

    return mask


# ==========================================
# 测试 / 批处理入口
# ==========================================
def process_masks(image_dir: Path, annotation_path: Path, output_mask_dir: Path):
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # 建立 image_id -> image_info 的映射
    id_to_img = {img['id']: img for img in coco_data['images']}

    for ann in coco_data['annotations']:
        img_info = id_to_img.get(ann['image_id'])
        if not img_info: continue

        image_name = Path(img_info['file_name']).name
        image_path = image_dir / image_name

        if not image_path.exists():
            continue

        # 提取关键信息
        kpts = ann.get('keypoints', [])
        kpt_types = ann.get('keypoint_types', [])
        bbox = ann.get('bbox', [0, 0, 100, 100])  # 兜底 bbox

        # 确保关键点类型数据存在
        if not kpts or not kpt_types:
            continue

        # 获取图片尺寸
        height, width = img_info['height'], img_info['width']

        # 生成 Mask
        mask_array = generate_inpaint_mask((height, width), kpts, kpt_types, bbox)

        # 如果这张图有被遮盖的残肢/假肢 (即 mask 不是全黑的)
        if np.any(mask_array > 0):
            # 将 numpy array 转为 Image 并保存
            mask_img = Image.fromarray(mask_array)
            # 保存名字，例如 原图名_mask.png
            mask_save_path = output_mask_dir / f"{Path(image_name).stem}_mask.png"
            mask_img.save(mask_save_path)
            print(f"成功生成 Mask: {mask_save_path.name}")


if __name__ == "__main__":
    process_masks(
        image_dir=Path('./data/eval_seg_padded'),
        annotation_path=Path('./data/filtered_annotations_padded_png.json'),
        output_mask_dir=Path('./data/inpaint_masks')
    )