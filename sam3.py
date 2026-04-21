import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from pose_extractor import PoseExtractor, read_kpts_annotation


def main()
    # 1. 初始化模型和 Processor
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    pose_extractor = PoseExtractor(config_file='./models/pose/vit_config.py',
                                   checkpoint_file='./models/pose/epoch_1.pth',
                                   device='cuda:0')


    # 2. 加载你的图片并初始化推理状态
    image_path = "./eval/baidu_残疾人跑步_21.jpg"
    annotation_path = "./data/train_final.json"
    filtered_kpts, ori_kpts, types = read_kpts_annotation(image_path, annotation_path)

    image = Image.open(image_path).convert("RGB")
    inference_state = processor.set_image(image)

    output_text = processor.set_text_prompt(
        state=inference_state,
        prompt="person"
    )


def segment_and_draw_mask(processor, inference_state, image, point_coords, point_labels, text):
    """
    接受最多 31 个关键点，使用 SAM 3 进行分割，并将 Mask 和关键点绘制在图像上。

    参数:
    - processor: 初始化的 Sam3Processor
    - inference_state: 图像推理状态
    - image: PIL.Image 格式的原始图像
    - point_coords: 列表或 numpy 数组，格式为 [[x1, y1], [x2, y2], ...]
    - point_labels: 列表或 numpy 数组，1 表示正向点，0 表示负向点
    """

    points = np.array(point_coords)
    labels = np.array(point_labels)

    processor.set_text_prompt(
        state=inference_state,
        prompt="person"
    )

    # 2. 调用 SAM 3 进行推理
    output = processor.set_point_prompt(
        state=inference_state,
        point_coords=points,
        point_labels=labels
    )

    # SAM 通常会返回 3 个不同层级（整体、部分、细节）的 Mask，我们需要获取置信度最高的那一个
    masks = output["masks"]  # 形状通常为 (N, H, W)
    scores = output["scores"]  # 形状为 (N,)
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]

    # 3. 开始可视化绘制
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # 叠加 Mask 与 关键点
    show_mask(best_mask, plt.gca())
    show_points(points, labels, plt.gca())

    plt.title(f"SAM 3 Segmentation (Score: {scores[best_idx]:.3f})", fontsize=14)
    plt.axis('off')
    plt.show()

    # 返回最佳的 Mask 数组供后续处理 (例如保存或裁剪)
    return best_mask

def show_mask(mask, ax, random_color=False):
    """在当前画布上叠加半透明的 Mask"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # 默认使用半透明的道奇蓝 (Dodger Blue)
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    # 将 boolean mask 转换为带颜色的 RGBA 图片层
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """绘制正向点和负向点，使用不同颜色和形状区分"""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    # 正向点：绿色五角星
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
                   s=marker_size, edgecolor='white', linewidth=1.25)

    # 负向点：红色叉号
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='X',
                   s=marker_size, edgecolor='white', linewidth=1.25)

if __name__ == "__main__":
    main()

