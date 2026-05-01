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

    def _get_residual_eval_rules(self, keypoint_types):
        """
        Build residual-limb evaluation rules for the pose-based geometric critic.

        Each rule defines:
        - res_idx: residual endpoint in the original image
        - anchor_idx: upstream anatomical joint
        - downstream_idx: first downstream joint in the generated proxy image

        This is used only for geometric validation, not for compositing.
        """
        mapping_dict = {
            # Upper limbs
            23: (5, 7),  # left upper-arm residual: left shoulder -> left elbow
            24: (6, 8),  # right upper-arm residual: right shoulder -> right elbow
            25: (7, 9),  # left forearm residual: left elbow -> left wrist
            26: (8, 10),  # right forearm residual: right elbow -> right wrist

            # Lower limbs
            27: (11, 13),  # left thigh residual: left hip -> left knee
            28: (12, 14),  # right thigh residual: right hip -> right knee
            29: (13, 15),  # left lower-leg residual: left knee -> left ankle
            30: (14, 16),  # right lower-leg residual: right knee -> right ankle
        }

        rules = []
        for res_idx, (anchor_idx, downstream_idx) in mapping_dict.items():
            if res_idx < len(keypoint_types) and keypoint_types[res_idx] in [0, 1]:
                rules.append({
                    "res_idx": res_idx,
                    "anchor_idx": anchor_idx,
                    "downstream_idx": downstream_idx,
                })

        return rules

    def _bbox_scale_from_keypoints(self, kpts, conf_thr=None):
        """
        Compute subject scale as sqrt(bounding-box area) from visible keypoints.
        This matches the scale-normalization idea used in the paper.
        """
        if conf_thr is None:
            conf_thr = getattr(self, "conf_thr", 0.10)

        kpts = np.asarray(kpts)
        visible = kpts[kpts[:, 2] > conf_thr]

        if len(visible) < 2:
            return 1.0

        min_xy = np.min(visible[:, :2], axis=0)
        max_xy = np.max(visible[:, :2], axis=0)

        w, h = max_xy - min_xy
        area = max(float(w * h), 1.0)

        return float(np.sqrt(area))

    def _safe_get_point(self, kpts, idx, conf_thr=None):
        """
        Safely fetch a visible 2D keypoint.
        Return None if the keypoint is missing or low-confidence.
        """
        if conf_thr is None:
            conf_thr = getattr(self, "conf_thr", 0.10)

        if idx >= len(kpts):
            return None

        if kpts[idx][2] <= conf_thr:
            return None

        return kpts[idx, :2].astype(np.float32)

    def evaluate_pose_geometric(self, kpts_orig, kpts_gen, kpt_types_orig):
        """
        Pose-based geometric critic.

        It validates two properties:

        1. Body preservation:
           Non-target visible keypoints should not move too much after image editing.

        2. Residual-limb direction/path consistency:
           The generated limb segment adjacent to the residual region should extend
           along the original residual-limb direction. Instead of using angular error,
           we compute the perpendicular deviation from the original residual-limb ray.

        Args:
            kpts_orig: np.ndarray of shape (N, 3), original keypoints.
            kpts_gen: np.ndarray of shape (N, 3), generated proxy keypoints.
            kpt_types_orig: list or array, original keypoint types.

        Returns:
            dict containing:
            - passed: bool
            - reason: str
            - E_pose: float
            - pose_ok: bool
            - direction_ok: bool
            - direction_checks: list of per-limb checking results
        """
        pose_tau = getattr(self, "pose_tau", 0.045)
        perp_tau = getattr(self, "perp_tau", 0.060)
        conf_thr = getattr(self, "conf_thr", 0.10)

        kpts_orig = np.asarray(kpts_orig, dtype=np.float32)
        kpts_gen = np.asarray(kpts_gen, dtype=np.float32)

        rules = self._get_residual_eval_rules(kpt_types_orig)

        if not rules:
            return {
                "passed": True,
                "reason": "No residual/prosthetic target found for pose evaluation.",
                "E_pose": 0.0,
                "pose_ok": True,
                "direction_ok": True,
                "direction_checks": []
            }

        scale = self._bbox_scale_from_keypoints(kpts_orig, conf_thr=conf_thr)

        # ------------------------------------------------------------------
        # 1. Body-preservation error
        # ------------------------------------------------------------------
        # These keypoints are expected to change or may not exist in the original image.
        # We exclude residual endpoints and first generated downstream joints.
        # Anchor joints are NOT excluded, because they should remain stable.
        target_indices = set()
        for r in rules:
            target_indices.add(r["res_idx"])
            target_indices.add(r["downstream_idx"])

        keep_errors = []

        num_kpts = min(len(kpts_orig), len(kpts_gen))
        for i in range(num_kpts):
            if i in target_indices:
                continue
            if kpt_types_orig[i] == 2:
                continue

            p_orig = self._safe_get_point(kpts_orig, i, conf_thr=conf_thr)
            p_gen = self._safe_get_point(kpts_gen, i, conf_thr=conf_thr)

            if p_orig is None or p_gen is None:
                continue

            dist = np.linalg.norm(p_gen - p_orig)
            keep_errors.append(dist / scale)

        E_pose = float(np.mean(keep_errors)) if keep_errors else 0.0
        pose_ok = E_pose <= pose_tau

        # ------------------------------------------------------------------
        # 2. Residual-limb direction/path consistency
        # ------------------------------------------------------------------
        direction_checks = []
        direction_ok = True

        for r in rules:
            res_idx = r["res_idx"]
            anchor_idx = r["anchor_idx"]
            downstream_idx = r["downstream_idx"]

            p_a = self._safe_get_point(kpts_orig, anchor_idx, conf_thr=conf_thr)
            p_r = self._safe_get_point(kpts_orig, res_idx, conf_thr=conf_thr)

            p_a_hat = self._safe_get_point(kpts_gen, anchor_idx, conf_thr=conf_thr)
            p_d_hat = self._safe_get_point(kpts_gen, downstream_idx, conf_thr=conf_thr)

            check = {
                "res_idx": res_idx,
                "anchor_idx": anchor_idx,
                "downstream_idx": downstream_idx,
                "passed": False,
                "alpha": None,
                "d_perp": None,
                "reason": ""
            }

            if p_a is None:
                check["reason"] = f"Original anchor keypoint {anchor_idx} is missing."
                direction_ok = False
                direction_checks.append(check)
                continue

            if p_r is None:
                check["reason"] = f"Original residual endpoint {res_idx} is missing."
                direction_ok = False
                direction_checks.append(check)
                continue

            if p_a_hat is None:
                check["reason"] = f"Generated anchor keypoint {anchor_idx} is missing."
                direction_ok = False
                direction_checks.append(check)
                continue

            if p_d_hat is None:
                check["reason"] = f"Generated downstream keypoint {downstream_idx} is missing."
                direction_ok = False
                direction_checks.append(check)
                continue

            # Original residual-limb vector:
            # original anchor joint -> original residual endpoint
            v_res = p_r - p_a

            # Generated adjacent limb segment:
            # generated anchor joint -> generated first downstream joint
            q_gen = p_d_hat - p_a_hat

            denom = float(np.dot(v_res, v_res))
            if denom < 1e-6:
                check["reason"] = "Original residual-limb vector is too short or degenerate."
                direction_ok = False
                direction_checks.append(check)
                continue

            # Projection coefficient of q_gen on the original residual-limb direction.
            alpha = float(np.dot(q_gen, v_res) / denom)

            # Perpendicular deviation from the original residual-limb ray.
            perp_vec = q_gen - alpha * v_res
            d_perp = float(np.linalg.norm(perp_vec) / scale)

            check["alpha"] = alpha
            check["d_perp"] = d_perp

            if alpha > 0 and d_perp < perp_tau:
                check["passed"] = True
                check["reason"] = (
                    f"Passed: alpha={alpha:.4f}, d_perp={d_perp:.4f}."
                )
            else:
                check["passed"] = False
                check["reason"] = (
                    f"Failed: alpha={alpha:.4f}, d_perp={d_perp:.4f}, "
                    f"required alpha > 0 and d_perp < {perp_tau:.4f}."
                )
                direction_ok = False

            direction_checks.append(check)

        passed = pose_ok and direction_ok

        return {
            "passed": passed,
            "reason": (
                f"E_pose={E_pose:.4f}, pose_ok={pose_ok}, "
                f"direction_ok={direction_ok}"
            ),
            "E_pose": E_pose,
            "pose_ok": pose_ok,
            "direction_ok": direction_ok,
            "direction_checks": direction_checks
        }

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
                            f"- 【{defect_key}】：只评估该目标区域生成出的生物肢体是否视觉合理。"
                            f"重点检查生成肢体的长度是否合理、解剖结构是否自然、材质/纹理是否与人体或衣物外观一致，"
                            f"以及是否存在明显畸形、扭曲、断裂、不合理比例或不自然连接。"
                        )
                    else:
                        # 明确已知是缺失断肢
                        defects_found.append(
                            f"沿着{defect_key}的残肢端点方向自然延伸，补全为具有完美解剖结构的完整{limb_name}（包含末端的{end_part}）。"
                            f"【强烈注意】：绝对不要对{defect_key}现有的残肢部分进行任何像素改变、不要改变其原有的位置和方向"
                        )
                        vlm_checks.append(
                            f"- 【{defect_key}】：只评估沿残肢补全出的生物肢体是否视觉合理。"
                            f"重点检查生成肢体的长度是否合理、解剖结构是否自然、材质/纹理是否与人体或衣物外观一致，"
                            f"以及是否存在明显畸形、扭曲、断裂、不合理比例或不自然连接。"
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
                f"[任务: 生成肢体视觉质量评估]\n"
                f"请严格按照以下标准，只检查图片中指定目标肢体区域的视觉质量：\n"
                f"{vlm_checks_str}\n\n"
                f"输出必须严格按照JSON格式:\n"
                f"{{\n"
                f"    \"passed\": true/false,\n"
                f"    \"reason\": \"简要说明目标生成肢体在长度、结构、材质/纹理或比例上是否合理。\"\n"
                f"}}"
            )

            return {
                "passed": False,
                "reason": f"检测到异常部位：{'、'.join(defect_parts_set)}",
                "prompt": repair_prompt,
                "eval_instruction": eval_instruction
            }

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
        print("🚀 AgentHMR-Res proxy generation with dual-critic validation")
        print("=" * 50)

        kpt_types_orig = original_annotation.get("keypoint_types", [])
        kpts_orig = np.array(original_annotation["keypoints"]).reshape(-1, 3)

        res = self.generate_prompt(image_path, kpts_orig, kpt_types_orig)

        if res["passed"]:
            print(f"✅ {res['reason']}")
            return image_path

        master_prompt = res["prompt"]
        print(f"📋 初诊发现缺陷，生成局部修改指令: {res['reason']}")
        print(f"📝 Master Prompt: {master_prompt}")

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_dir = os.path.join(base_output_dir, base_name)
        os.makedirs(save_dir, exist_ok=True)

        for attempt in range(max_attempts):
            print(f"\n🔄 尝试生成 proxy image ({attempt + 1}/{max_attempts})...")

            gen_image_path = self.generate_full_image(
                image_path,
                save_dir,
                master_prompt,
                str(attempt)
            )

            if not gen_image_path:
                print("❌ 图片生成失败，跳过本次重试。")
                continue

            # 1. Pose-based geometric critic
            try:
                kpts_gen = self.pose_extractor.extract_31_keypoints(gen_image_path)
            except Exception as e:
                print(f"❌ 生成图 pose extraction 失败: {e}")
                continue

            pose_eval = self.evaluate_pose_geometric(
                kpts_orig=kpts_orig,
                kpts_gen=kpts_gen,
                kpt_types_orig=kpt_types_orig,
            )

            print(f"📐 Pose critic: {'✅ 通过' if pose_eval['passed'] else '❌ 未通过'}")
            print(f"   {pose_eval['reason']}")

            if not pose_eval["passed"]:
                print("⚠️ Pose critic failed, regenerate from original image.")
                continue

            # 2. VLM-based visual-quality critic
            gen_eval_res = self.evaluate_image(gen_image_path, res["eval_instruction"])

            print(f"👁️ VLM critic: {'✅ 通过' if gen_eval_res['passed'] else '❌ 未通过'}")
            print(f"   {gen_eval_res.get('reason', '')}")

            if gen_eval_res["passed"]:
                print("🎉 Proxy image passed dual-critic validation!")
                return gen_image_path

            print("⚠️ VLM critic failed, regenerate from original image.")

        print("❌ No proxy image passed dual-critic validation.")
        return None

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
        else:
            print(f"⚠️ 警告: JSON 中没有找到图片 {img_name} 的关联标注，跳过该图。")