import json
import os
from pathlib import Path

import cv2
import time
import numpy as np
import torch
from PIL import Image
from dashscope import MultiModalConversation

from image_ops import ImageProcessor
from pose_extractor import PoseExtractor

_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else _original_load(*args, **kwargs)

class LimbCompositingAgent:
    def __init__(self, pose_extractor, edit_model='qwen-image-2.0-pro'):
        self.edit_model = edit_model
        self.pose_extractor = pose_extractor

        self.generation_instruction = """
        [Task: Character Completion]
        Please completely reconstruct and generate the missing limbs of the person in the image.
        Ensure the generated limbs look highly realistic and match the person's skin tone, lighting, and proportions perfectly.
        Keep the background exactly the same white.
        """

    def _get_compositing_rules(self, keypoint_types):
        """
        找出原图的残肢点，并映射出在生成图中需要连线提取的正常关节点。
        返回规则: anchor(锚点), orig_res(原图残肢端点), downstream(生成图里需要提取的下游关节)
        """
        mapping_dict = {
            23: (5, 23, [7, 9, 17]),  # 左肩残 -> 提取生成图的: 左肩(5)->左肘(7)->左腕(9)
            24: (6, 24, [8, 10, 18]),  # 右肩残 -> 提取生成图的: 右肩(6)->右肘(8)->右腕(10)
            25: (7, 25, [9, 17]),  # 左肘残 -> 提取生成图的: 左肘(7)->左腕(9)
            26: (8, 26, [10, 18]),  # 右肘残 -> 提取生成图的: 右肘(8)->右腕(10)
            27: (11, 27, [13, 15, 19, 21]),  # 左胯残 -> 提取生成图的: 左胯(11)->左膝(13)->左踝(15)
            28: (12, 28, [14, 16, 20, 22]),  # 右胯残 -> 提取生成图的: 右胯(12)->右膝(14)->右踝(16)
            29: (13, 29, [15, 19, 21]),  # 左膝残 -> 提取生成图的: 左膝(13)->左踝(15)
            30: (14, 30, [16, 20, 22])  # 右膝残 -> 提取生成图的: 右膝(14)->右踝(16)
        }

        rules = []
        for res_idx, rule in mapping_dict.items():
            if len(keypoint_types) > res_idx and keypoint_types[res_idx] in [0, 1]:
                rules.append({
                    "res_idx": res_idx,
                    "orig_anchor": rule[0],
                    "orig_res": rule[1],
                    "downstream": rule[2]
                })
        return rules

    def generate_pure_extraction_mask(self, image_shape, kpts_gen, anchor_idx, downstream_indices, bbox):
        """
        极简版提取 Mask：不看任何 type，直接把锚点和所有下游点连成一条粗线。
        """
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        # 动态计算粗细
        box_w, box_h = bbox[2], bbox[3]
        limb_width = int(max(box_w, box_h) * 0.08)
        limb_width = max(limb_width, 10)

        # 把锚点和下游点串成一条路径 (例如：[右肩, 右肘, 右腕])
        path_indices = [anchor_idx] + downstream_indices

        # 收集这些点在生成图中的真实坐标
        path_points = []
        for idx in path_indices:
            # 确保大模型确实生成了这个点 (置信度 > 0.1)
            if kpts_gen[idx][2] > 0.1:
                path_points.append((int(kpts_gen[idx][0]), int(kpts_gen[idx][1])))

        # 顺着路径画线和圆圈
        if len(path_points) >= 2:
            for i in range(len(path_points) - 1):
                p1, p2 = path_points[i], path_points[i + 1]
                cv2.line(mask, p1, p2, color=255, thickness=limb_width)
                cv2.circle(mask, p1, limb_width // 2, 255, -1)
                cv2.circle(mask, p2, limb_width // 2, 255, -1)
        else:
            print("⚠️ 警告：大模型生成的肢体关键点丢失，无法画出完整的提取 Mask。")

        return mask

    def align_and_blend(self, orig_bgr, gen_bgr, mask_uint8, pt_orig_anchor, pt_orig_res, pt_gen_anchor, pt_gen_end):
        """核心仿射变换与泊松融合"""
        # 1. 计算夹角差
        vec_orig = pt_orig_res - pt_orig_anchor
        vec_gen = pt_gen_end - pt_gen_anchor

        angle_orig = np.arctan2(vec_orig[1], vec_orig[0])
        angle_gen = np.arctan2(vec_gen[1], vec_gen[0])
        angle_diff = np.degrees(angle_orig - angle_gen)

        # 2. 计算平移差
        tx = pt_orig_anchor[0] - pt_gen_anchor[0]
        ty = pt_orig_anchor[1] - pt_gen_anchor[1]

        # 3. 构建仿射矩阵并变换
        center_pt = (float(pt_gen_anchor[0]), float(pt_gen_anchor[1]))
        M = cv2.getRotationMatrix2D(center_pt, angle_diff, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty

        h, w = orig_bgr.shape[:2]
        warped_gen = cv2.warpAffine(gen_bgr, M, (w, h))
        warped_mask = cv2.warpAffine(mask_uint8, M, (w, h))

        # 4. 泊松融合
        contours, _ = cv2.findContours(warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return orig_bgr

        c = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(c)
        center = (int(x + w_box / 2), int(y + h_box / 2))

        try:
            return cv2.seamlessClone(warped_gen, orig_bgr, warped_mask, center, cv2.NORMAL_CLONE)
        except Exception:
            mask_3d = warped_mask[:, :, None] / 255.0
            return (warped_gen * mask_3d + orig_bgr * (1 - mask_3d)).astype(np.uint8)

    def run(self, image_path, base_output_dir, original_annotation):
        print("\n" + "=" * 50)
        print(f"🔪 启动骨架刻刀再植流水线 (Clean Logic)")
        print("=" * 50)

        orig_bgr = cv2.imread(image_path)
        keypoint_types = original_annotation.get("keypoint_types", [])
        kpts_orig = np.array(original_annotation["keypoints"])

        # 获取切图规则
        rules = self._get_compositing_rules(keypoint_types)

        if not rules:
            raise ValueError('No rules found!')

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_dir = os.path.join(base_output_dir, base_name)
        os.makedirs(save_dir, exist_ok=True)

        gen_image_path = self.generate_full_image(image_path, save_dir)

        # 提取生成图关键点和 bbox
        kpts_gen = self.pose_extractor.extract_31_keypoints(gen_image_path)
        gen_bgr = cv2.imread(gen_image_path)

        valid_kpts = kpts_gen[kpts_gen[:, 2] > 0.1]
        min_x, min_y = np.min(valid_kpts[:, :2], axis=0)
        max_x, max_y = np.max(valid_kpts[:, :2], axis=0)
        bbox_gen = [min_x, min_y, max_x - min_x, max_y - min_y]

        current_canvas = orig_bgr.copy()

        # 开始切割缝合
        for rule in rules:
            print(f"✂️ 直接沿正常关节提取肢体...")

            # 极简提取法：直接传入生成的关键点、锚点和下游点列表
            limb_mask = self.generate_pure_extraction_mask(
                image_shape=gen_bgr.shape,
                kpts_gen=kpts_gen,
                anchor_idx=rule["orig_anchor"],
                downstream_indices=rule["downstream"],
                bbox=bbox_gen
            )

            # 获取物理坐标进行对齐
            pt_orig_anchor = kpts_orig[rule['orig_anchor'], :2]
            pt_orig_res = kpts_orig[rule['orig_res'], :2]

            pt_gen_anchor = kpts_gen[rule['orig_anchor'], :2]
            # 计算方向所用的终点，取最末端的关节
            pt_gen_end = kpts_gen[rule['downstream'][0], :2]

            # 缝合
            current_canvas = self.align_and_blend(
                current_canvas, gen_bgr, limb_mask,
                pt_orig_anchor, pt_orig_res, pt_gen_anchor, pt_gen_end
            )
        final_path = os.path.join(save_dir, 'final.png')
        cv2.imwrite(final_path, current_canvas)
        return final_path

    def generate_full_image(self, image_path, save_dir):
        """
        调用 Qwen 大模型，让它自由发挥补全肢体（生成素材库）。
        """
        print(f"\n🎨 [生成] 调用 {self.edit_model} 生成完整肢体素材...")

        # 1. 编码原图 (沿用你之前的工具类)
        image_encoded = ImageProcessor.encode_file(image_path)

        # 2. 组装最干净的 Payload：只有原图和补全指令
        content_list = [
            {"image": image_encoded},
            {"text": self.generation_instruction.strip()}
        ]
        messages = [{"role": "user", "content": content_list}]

        # 3. 稳健的网络请求重试机制
        api_attempt = 0
        while api_attempt < 3:
            try:
                # 沿用你之前跑通的 API 调用参数
                response = MultiModalConversation.call(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    model=self.edit_model,
                    messages=messages,
                    stream=False,
                    n=1,
                    seed=42,
                    guidance_scale=7.0,
                    watermark=False,
                    negative_prompt="shifting torso, changing existing joints, changing clothes",
                    prompt_extend=False
                )

                if response.status_code == 200:
                    for content in response.output.choices[0].message.content:
                        if 'image' in content:
                            print(f"✅ 素材库生成成功！")
                            # 下载并保存图片 (这里传 attempt_suffix="raw_gen" 以示区分)
                            path = ImageProcessor.save_image_from_url(
                                content['image'],
                                'compositing_material',
                                'raw_gen',
                                save_dir
                            )
                            return path
                    raise RuntimeError("❌ API 返回了 200，但未找到图片链接。")
                else:
                    raise RuntimeError(f"❌ 图像生成失败: HTTP {response.status_code}, {response.message}")

            except Exception as e:
                api_attempt += 1
                time.sleep(3)
                print(f"❌ API 调用崩溃 (网络重试 {api_attempt}/3): {str(e)}")

        return None


if __name__ == "__main__":
    image_dir = Path('./data/eval_seg_padded')
    save_dir = Path('./workdir1')

    pose_extractor = PoseExtractor(
        config_file='./models/pose/vit_config.py',
        checkpoint_file='./models/pose/epoch_1.pth',
        device='cuda:0'
    )
    agent = LimbCompositingAgent(pose_extractor)

    # 1. 一次性读取 COCO 格式的 JSON
    annotation_path = Path('./data/train_final.json')
    with open(annotation_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 2. 解析 images 列表，建立 image_id -> file_name 的快速映射
    image_id_to_name = {}
    for img_info in coco_data.get('images', []):
        # 假设 JSON 里存的是 "baidu_残疾人跑步_21.png" 或者带路径的相对地址
        # 用 Path(xxx).name 只取纯文件名，方便后续和本地文件完美匹配
        pure_name = Path(img_info['file_name']).name
        image_id_to_name[img_info['id']] = pure_name

    # 3. 解析 annotations 列表，建立 file_name -> annotation 的终极映射字典
    anno_dict = {}
    for anno in coco_data.get('annotations', []):
        image_id = anno.get('image_id')
        if image_id in image_id_to_name:
            file_name = image_id_to_name[image_id]
            # 假设每张图只有一个核心标注目标
            # 注意：COCO 格式中关键点通常存在 anno['keypoints']，如果有你自定义的 'keypoint_types' 也会在这里面
            anno_dict[file_name] = anno

    # 4. 开始遍历本地文件夹里的图片
    for i, image_path in enumerate(image_dir.glob('*.png')):
        img_name = image_path.name  # 取纯文件名

        # 5. 精准匹配
        if img_name in anno_dict:
            current_image_annotation = anno_dict[img_name]

            print(f"\n[{i + 1}] 正在处理: {img_name}")

            # 传给 agent！
            agent.run(str(image_path), str(save_dir), current_image_annotation)
        else:
            print(f"⚠️ 警告: JSON 中没有找到图片 {img_name} 的关联标注，跳过该图。")