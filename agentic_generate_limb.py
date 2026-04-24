import json
import os
import re
from pathlib import Path

import cv2
import time

import dashscope
import numpy as np
import torch
from anyio import sleep
from dashscope import MultiModalConversation
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message

from image_ops import ImageProcessor

from pose_extractor import PoseExtractor

_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else _original_load(*args, **kwargs)

dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

class LimbCompositingAgent:
    def __init__(self, pose_extractor, edit_model='wan2.7-image-pro', eval_model='qwen3.6-plus'):
        self.edit_model = edit_model
        self.pose_extractor = pose_extractor
        self.eval_model = eval_model
        self.generation_instruction = """
        如果图中有假肢/义肢就替换图中的假肢/义肢为正常肢体，其他部位不要有任何改动。如果有残肢就沿着残肢生成合适的正常肢体确认双手双脚都有。
        """
        self.eval_instruction = """
        [任务: 确认肢体健全]
        观察图片是否是一个四肢健全的人，有完好的双手双脚。如果发现肢体缺失或者带有假肢都算图片不合格。可以接受某些肢体被身体遮挡看不见，但是不能有可见的残肢端点
        
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

    def generate_prompt(self, image_path, kpts, kpt_types):
        """
        评估图片是否存在残肢。
        基于确定的缺陷类型，同时生成：
        1. 给修复模型的精准 Master Prompt
        2. 给复查裁判的专属 VLM Instruction (绝不含糊其辞)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片用于 Eval: {image_path}")
        height, width = img.shape[:2]
        fallback_center_x = width / 2.0

        def get_valid_x(kpt_id):
            if kpt_id < len(kpt_types) and kpts[kpt_id][2] > 0:
                return kpts[kpt_id][0]
            return None

        # 计算身体中线用于判断左右
        ls_x, rs_x = get_valid_x(5), get_valid_x(6)
        upper_xs = [x for x in (ls_x, rs_x) if x is not None]
        upper_center_x = sum(upper_xs) / len(upper_xs) if upper_xs else fallback_center_x

        lh_x, rh_x = get_valid_x(11), get_valid_x(12)
        lower_xs = [x for x in (lh_x, rh_x) if x is not None]
        lower_center_x = sum(lower_xs) / len(lower_xs) if lower_xs else fallback_center_x

        res_mapping = {
            23: (17, '手臂', 'upper'),
            24: (18, '手臂', 'upper'),
            27: (21, '腿部', 'lower'),
            28: (22, '腿部', 'lower'),
            25: (17, '手臂', 'upper'),
            26: (18, '手臂', 'upper'),
            29: (21, '腿部', 'lower'),
            30: (22, '腿部', 'lower')
        }

        defects_found = []
        vlm_checks = []  # 专门存放精准的验收标准
        defect_parts_set = set()

        for res_id, (downstream_id, limb_name, body_part_type) in res_mapping.items():
            if res_id < len(kpt_types) and kpt_types[res_id] == 0:
                res_x = kpts[res_id][0]
                ref_center_x = upper_center_x if body_part_type == 'upper' else lower_center_x
                screen_side = "画面左侧" if res_x < ref_center_x else "画面右侧"
                defect_key = f"{screen_side}的{limb_name}"

                if defect_key not in defect_parts_set:
                    defect_parts_set.add(defect_key)

                    downstream_type = kpt_types[downstream_id] if downstream_id < len(kpt_types) else 2

                    # 动态明确末端名称，杜绝“手/脚”
                    end_part = "手" if limb_name == "手臂" else "脚"

                    if downstream_type == 1:
                        # 明确已知是假肢
                        defects_found.append(
                            f"将{defect_key}的假肢/义肢完全替换为具有完美解剖结构的真实正常肢体（包含末端的{end_part}）"
                        )
                        vlm_checks.append(
                            f"- 【{defect_key}】：必须确认原有的假肢或义肢已被彻底移除，并替换为真实的血肉肢体，且包含完整的{end_part}。如果该部位依然能看到任何机械、碳纤维等假肢痕迹，则算作不合格。"
                        )
                    else:
                        # 明确已知是缺失断肢
                        defects_found.append(
                            f"沿着{defect_key}的残肢端点方向自然延伸，补全为具有完美解剖结构的完整{limb_name}（包含末端的{end_part}）。"
                            f"【强烈注意】：绝对不要对{defect_key}现有的残肢部分进行任何像素改变、不要改变其原有的位置和方向"
                        )
                        vlm_checks.append(
                            f"- 【{defect_key}】：必须确认残缺部分已经被自然延伸并补全，且末端包含完整的{end_part}。如果该部位依然呈现为断肢、或者延伸部分缺失了{end_part}，则算作不合格。"
                        )

        if not defects_found:
            return {
                "passed": True,
                "reason": "未检测到生物性残肢断点，无需补全。",
                "prompt": "",
                "eval_instruction": ""
            }
        else:
            defect_desc = "；并且，".join(defects_found)
            repair_prompt = (
                f"当前图片存在需要修改的肢体。请严格执行以下局部修改指令：\n{defect_desc}。\n"
                f"【全局绝对指令】：只允许对上述明确指定的缺陷部位进行处理！绝对不要移动、修改或重绘任何现有的健全肢体、躯干、脸部以及背景环境！必须保持原图人物主体动作和画风完全不变。"
            )

            # 将精准的验收标准组装给大模型
            vlm_checks_str = "\n".join(vlm_checks)
            eval_instruction = (
                f"[任务: 确认特定缺陷部位是否完美修复]\n"
                f"请严格按照以下标准，检查图片中的特定部位：\n"
                f"{vlm_checks_str}\n\n"
                f"输出必须严格按照JSON格式:\n"
                f"{{\n"
                f"    \"passed\": true/false,\n"
                f"    \"reason\": \"详细描述你观察到的上述被检部位的当前状态，明确说明其是否满足了上述通过标准。\"\n"
                f"}}"
            )

            return {
                "passed": False,
                "reason": f"检测到异常部位：{'、'.join(defect_parts_set)}",
                "prompt": repair_prompt,
                "eval_instruction": eval_instruction
            }

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

    def evaluate_image(self, image_path, prompt):
        print(f"👁️ [复查] 召唤视觉大模型 {self.eval_model} 担任裁判...")

        # 1. 编码图片 (由于 DashScope 多模态支持 file:// 协议，如果 ImageProcessor 返回的是 base64 也可以直接用)
        image_encoded = ImageProcessor.encode_file(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_encoded},
                    {"text": prompt}
                ]
            }
        ]

        api_attempt = 0
        while api_attempt < 3:
            try:
                # 调用多模态视觉模型
                response = MultiModalConversation.call(
                    model=self.eval_model,
                    messages=messages,
                )

                if response.status_code == 200:
                    # DashScope 多模态返回的 content 通常是一个包含 dict 的 list
                    raw_content = response.output.choices[0].message.content
                    text_result = ""
                    if isinstance(raw_content, list):
                        for item in raw_content:
                            if 'text' in item:
                                text_result += item['text']
                    else:
                        text_result = str(raw_content)

                    # 鲁棒解析 JSON (防止大模型加上 ```json 等 markdown 标记)
                    json_match = re.search(r'\{.*\}', text_result, re.DOTALL)
                    if json_match:
                        result_dict = json.loads(json_match.group())
                        # 确保 key 存在
                        if "passed" in result_dict:
                            return result_dict

                    print(f"⚠️ VLM 裁判返回格式异常，未找到合法的 JSON。原始输出: {text_result}")
                    raise ValueError("JSON parse failed")

                else:
                    sleep(3)
                    raise RuntimeError(f"VLM API 调用失败: {response.code} - {response.message}")

            except Exception as e:
                api_attempt += 1
                print(f"⚠️ VLM 裁判开小差了 (重试 {api_attempt}/3): {str(e)}")
                time.sleep(2)

        # 兜底：如果 API 彻底挂了，默认让它重画
        return {"passed": False, "reason": "VLM 裁判多次调用失败，默认复查不通过。"}

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
        kpt_types_orig = original_annotation.get("keypoint_types", [])
        kpts_orig = np.array(original_annotation["keypoints"]).reshape(-1, 3)

        res = self.generate_prompt(image_path, kpts_orig, kpt_types_orig)

        master_prompt = res['prompt']
        print(f"📋 初诊发现缺陷，生成全局修改指令: {res['reason']}")
        print(f"📝 Master Prompt: {master_prompt}")

        # 获取切图规则
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_dir = os.path.join(base_output_dir, base_name)
        os.makedirs(save_dir, exist_ok=True)

        # for attempt in range(max_attempts):
        #     print(f"\n🔄 尝试修复 ({attempt + 1}/{max_attempts})...")
        #
        #     # =========================================================
        #     # 核心改动 2：无论第几次尝试，永远传入【原图】 + 【Master Prompt】
        #     # =========================================================
        #     gen_image_path = self.generate_full_image(image_path, save_dir, master_prompt, str(attempt))
        #
        #     if not gen_image_path:
        #         print("❌ 图片生成失败，跳过本次重试。")
        #         continue
        #
        #     # =========================================================
        #     # 核心改动 3：对【生成的新图】进行姿态提取和“复查”
        #     # =========================================================
        #     # 注意：这里必须提取生成图的 kpts！不能用原图的！
        #
        #     gen_eval_res = self.evaluate_image(gen_image_path, res['eval_instruction'])
        #
        #     print(f"📊 复查评估: {'✅ 通过' if gen_eval_res['passed'] else '❌ 未通过'}")
        #
        #     if gen_eval_res['passed']:
        #         print("🎉 肢体补全成功！")
        #         # 走你后面的对齐、抠图缝合逻辑...
        #         break
        #     else:
        #         print(f"⚠️ 生成图依然存在问题: {gen_eval_res['reason']}")

        rules = self._get_compositing_rules(kpt_types_orig)

        if not rules:
            raise ValueError('No rules found!')
        # gen_image_path='./workdir1/bing_義足のランナー_6068/compositing_material_raw_gen.jpg'
        image_folder = Path(save_dir)

        # --- 极简逻辑 ---
        # 1. 拿到目录下所有文件，排除 final.png
        # 只要是文件就全收进来，不管它是 .png, .jpg 还是其他
        all_files = [str(p) for p in image_folder.iterdir() if p.is_file() and p.name != 'final.png']

        if not all_files:
            raise FileNotFoundError(f"目录 {save_dir} 是空的，没找到素材。")

        # 2. 字母序排序
        all_files.sort()
        # 3. 取最后一张
        gen_image_path = all_files[-1]

        print(f"🎯 选定融合素材: {gen_image_path}")
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

    def generate_full_image(self, image_path, save_dir, prompt, iter):
        """
        调用 Qwen 大模型，让它自由发挥补全肢体（生成素材库）。
        """
        print(f"\n🎨 [生成] 调用 {self.edit_model} 生成完整肢体素材...")

        # 1. 编码原图 (沿用你之前的工具类)
        image_encoded = ImageProcessor.encode_file(image_path)

        # 2. 组装最干净的 Payload：只有原图和补全指令
        content_list = [
            {"image": image_encoded},
            {"text": prompt.strip()}
        ]
        messages = Message(
            role="user",
            content= content_list
        )

        # 3. 稳健的网络请求重试机制
        api_attempt = 0
        while api_attempt < 3:
            try:
                print(os.getenv("DASHSCOPE_API_KEY"))
                # 沿用你之前跑通的 API 调用参数
                response = ImageGeneration.call(
                    model=self.edit_model,
                    messages=[messages],
                    n=1,
                )

                if response.status_code == 200:
                    for i, choice in enumerate(response.output.choices):
                        for j, content in enumerate(choice["message"]["content"]):
                            if content.get("type") == "image":
                                print(f"✅ 素材库生成成功！")
                                # 下载并保存图片 (这里传 attempt_suffix="raw_gen" 以示区分)
                                path = ImageProcessor.save_image_from_url(
                                    content['image'],
                                    'compositing_material',
                                    iter,
                                    save_dir
                                )
                                return path
                    raise RuntimeError("❌ API 返回了 200，但未找到图片链接。")
                else:
                    sleep(5)
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
        # if i <= 10: continue
        img_name = image_path.name  # 取纯文件名

        # 5. 精准匹配
        if img_name in anno_dict:
            current_image_annotation = anno_dict[img_name]

            print(f"\n[{i + 1}] 正在处理: {img_name}")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            target_dir = os.path.join(save_dir, base_name)
            # if os.path.exists(target_dir):
            #     print(f"⚠️ 目录 {target_dir} 已存在，跳过该图以避免覆盖。")
            #     continue
            # 传给 agent！
            agent.run(str(image_path), str(save_dir), current_image_annotation)
            break
        else:
            print(f"⚠️ 警告: JSON 中没有找到图片 {img_name} 的关联标注，跳过该图。")