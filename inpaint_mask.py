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
    :param keypoint_types: 标注类型列表，1 表示假肢，0 表示正常/残肢
    :param bbox: [x, y, width, height] 用于动态计算肢体粗细
    :return: Numpy 数组的 Mask (0为背景，255为需要Inpaint的区域)
    """
    height, width = image_shape
    # 创建纯黑背景 Mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # 格式化 kpts
    kpts = np.array(keypoints).reshape(-1, 3)

    # 动态计算肢体宽度基准 (按 Bounding Box 最大边的 8% 估算，可根据实际效果微调)
    box_w, box_h = bbox[2], bbox[3]
    limb_width = int(max(box_w, box_h) * 0.08)
    # 确保宽度至少有几个像素，避免太细
    limb_width = max(limb_width, 10)

    # 遍历所有可能的残肢点
    for res_idx, info in RESIDUAL_KINEMATICS.items():
        rx, ry, rv = kpts[res_idx]

        # 如果该残肢点不存在/不可见，跳过
        if rv == 0:
            continue

        rx, ry = int(rx), int(ry)

        # 检查其下游是否连接了“假肢”
        prosthetic_points = []
        for down_idx in info['downstream']:
            dx, dy, dv = kpts[down_idx]
            # 如果点存在 且 类型是假肢 (1)
            if dv > 0 and keypoint_types[down_idx] == 1:
                prosthetic_points.append((int(dx), int(dy)))

        # ==========================================
        # 逻辑分支 1：存在假肢
        # ==========================================
        if len(prosthetic_points) > 0:
            # 路线：从残肢点开始，连接所有下游假肢点
            path_points = [(rx, ry)] + prosthetic_points

            for i in range(len(path_points) - 1):
                p1, p2 = path_points[i], path_points[i + 1]
                # 画粗线连接
                cv2.line(mask, p1, p2, color=255, thickness=limb_width)
                # 在关节点画圆，确保连接处圆滑（类似圆角线帽）
                cv2.circle(mask, p1, limb_width // 2, 255, -1)
                cv2.circle(mask, p2, limb_width // 2, 255, -1)

        # ==========================================
        # 逻辑分支 2：没有假肢 (纯残肢肉包)
        # ==========================================
        else:
            parent_idx = info['parent']
            px, py, pv = kpts[parent_idx]

            if pv > 0:
                # 获取从父节点指向残肢点的向量
                vx, vy = rx - px, ry - py
                length = np.sqrt(vx ** 2 + vy ** 2)

                if length > 0:
                    # 向量归一化
                    vx, vy = vx / length, vy / length
                    # 沿着方向向外延伸一个 limb_width 的距离
                    extend_x = int(rx + vx * limb_width)
                    extend_y = int(ry + vy * limb_width)

                    # 画一条向外延伸的粗线（形似长条形肉包）
                    cv2.line(mask, (rx, ry), (extend_x, extend_y), color=255, thickness=limb_width)
                    # 两端画圆角
                    cv2.circle(mask, (rx, ry), limb_width // 2, 255, -1)
                    cv2.circle(mask, (extend_x, extend_y), int(limb_width * 0.6), 255, -1)  # 顶端稍微膨胀一点
            else:
                # 兜底情况：如果连父节点(如肩膀/大腿根)都被遮挡没标注
                # 就直接在残肢点画一个稍大的圆形肉包
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
        annotation_path=Path('./data/filtered_annotations_padded.json'),
        output_mask_dir=Path('./data/inpaint_masks')
    )