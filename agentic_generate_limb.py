import json
import os
import re
from pathlib import Path

import cv2
import time

import dashscope
import numpy as np
import torch
from PIL import Image
from dashscope import MultiModalConversation

from image_ops import ImageProcessor
from pose_extractor import PoseExtractor

_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else _original_load(*args, **kwargs)

dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

class LimbCompositingAgent:
    def __init__(self, pose_extractor, edit_model='qwen-image-2.0-pro', eval_model='qwen3.6-plus'):
        self.edit_model = edit_model
        self.pose_extractor = pose_extractor
        self.eval_model = eval_model
        self.generation_instruction = """
        [任务: 健全肢体生成]
        图中是一个确实肢体的残疾人。有可能带假肢有可能不带。确认残疾肢体位置，如果是带假肢的人将假肢替换成健全肢体。如果没带假肢请把他的肢体沿着
        残肢的方向补全。确保背景是白色不变。
        """
        self.eval_instruction = """
        [任务: 确认肢体健全]
        观察图片是否是一个四肢健全的人。如果发现肢体缺失或者带有假肢都算图片不合格。可以接受某些肢体被身体遮挡看不见，但是不能有可见的残肢端点
        
        输出按照JSON格式:
        {{
            "passed": true/false,
            "reason": 描述不合格原因比如有假肢或者某个肢体存在残肢。
        }}
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

    def evaluate_image(self, image_path):
        generated_image = ImageProcessor.encode_file(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"imaeg": generated_image},
                {"text": self.eval_instruction},
            ]
        }]
        resp = MultiModalConversation.call(
            model=self.eval_model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"❌ VLM 评估失败: {resp.message}")

        content = resp.output.choices[0].message.content

        if isinstance(content, list):
            text_parts = [item['text'] for item in content if 'text' in item]
            content = "".join(text_parts)

        # 清洗可能混入的思考标签或 Markdown
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"passed": False, "reason": "JSON_PARSE_ERROR"}

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
        """放弃泊松融合，改用清晰度更高的 Alpha 混合"""
        # 1. 计算仿射变换矩阵 (保持不变)
        vec_orig = pt_orig_res - pt_orig_anchor
        vec_gen = pt_gen_end - pt_gen_anchor
        angle_diff = np.degrees(np.arctan2(vec_orig[1], vec_orig[0]) - np.arctan2(vec_gen[1], vec_gen[0]))
        tx, ty = pt_orig_anchor[0] - pt_gen_anchor[0], pt_orig_anchor[1] - pt_gen_anchor[1]

        center_pt = (float(pt_gen_anchor[0]), float(pt_gen_anchor[1]))
        M = cv2.getRotationMatrix2D(center_pt, angle_diff, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty

        # 2. 变换素材和遮罩
        h, w = orig_bgr.shape[:2]
        # 背景统一用白色填充
        warped_gen = cv2.warpAffine(gen_bgr, M, (w, h), borderValue=(255, 255, 255))
        warped_mask = cv2.warpAffine(mask_uint8, M, (w, h), borderValue=0)

        # 3. 【关键】对 Mask 进行羽化，防止边缘锯齿
        # 使用较小的高斯模糊 (如 5x5 或 7x7)，既能软化边缘，又不会像泊松那样弄花整条腿
        feathered_mask = cv2.GaussianBlur(warped_mask, (7, 7), 0)
        alpha = feathered_mask.astype(float) / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])  # 转为3通道

        # 4. 直接合成 (Final = 素材 * alpha + 原图 * (1-alpha))
        foreground = warped_gen.astype(float)
        background = orig_bgr.astype(float)

        final_img = (foreground * alpha + background * (1.0 - alpha)).astype(np.uint8)

        return final_img

    def run(self, image_path, base_output_dir, original_annotation, max_attempts=3):
        print("\n" + "=" * 50)
        print(f"🔪 启动骨架刻刀再植流水线 (Clean Logic)")
        print("=" * 50)

        img_with_alpha = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if img_with_alpha is not None and img_with_alpha.shape[2] == 4:
            # 分离通道
            b, g, r, a = cv2.split(img_with_alpha)
            # 创建纯白背景 (与原图等大)
            white_bg = np.ones_like(img_with_alpha[:, :, :3]) * 255
            # 将 alpha 归一化到 0-1
            alpha_factor = a.astype(float) / 255.0
            alpha_factor = cv2.merge([alpha_factor, alpha_factor, alpha_factor])
            # 前景 (人)
            foreground = cv2.merge([b, g, r]).astype(float)
            # 背景 (白)
            background = white_bg.astype(float)
            # 线性插值合成: Final = Foreground * Alpha + Background * (1 - Alpha)
            orig_bgr = cv2.add(cv2.multiply(foreground, alpha_factor),
                               cv2.multiply(background, 1.0 - alpha_factor)).astype(np.uint8)
        else:
            # 如果没有 Alpha 通道或者是普通图片，直接读取 BGR
            orig_bgr = cv2.imread(image_path)
        # ----------------------------------------------
        keypoint_types = original_annotation.get("keypoint_types", [])
        kpts_orig = np.array(original_annotation["keypoints"]).reshape(-1, 3)

        # 获取切图规则
        rules = self._get_compositing_rules(keypoint_types)

        if not rules:
            raise ValueError('No rules found!')

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_dir = os.path.join(base_output_dir, base_name)
        os.makedirs(save_dir, exist_ok=True)

        for i in range(max_attempts):
            gen_image_path = self.generate_full_image(image_path, save_dir)
            eval_res = self.evaluate_image(gen_image_path)
            print(f"📊 评估: {'✅ 通过' if eval_res['passed'] else '❌ 未通过'}")
            print(f"📝 原因: {eval_res.get('reason', '-')}")
        # gen_image_path='./workdir1/bing_義足のランナー_6068/compositing_material_raw_gen.jpg'
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
                print(os.getenv("DASHSCOPE_API_KEY"))
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
    annotation_path = Path('./data/filtered_annotations_padded_png.json')
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
        if i > 0: break
        img_name = image_path.name  # 取纯文件名

        # 5. 精准匹配
        if img_name in anno_dict:
            current_image_annotation = anno_dict[img_name]

            print(f"\n[{i + 1}] 正在处理: {img_name}")

            # 传给 agent！
            agent.run(str(image_path), str(save_dir), current_image_annotation)
        else:
            print(f"⚠️ 警告: JSON 中没有找到图片 {img_name} 的关联标注，跳过该图。")