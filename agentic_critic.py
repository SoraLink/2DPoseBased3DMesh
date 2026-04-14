import json
import math
import os
import re
import time

import requests
import dashscope
from dashscope import MultiModalConversation
import numpy as np

from auto_param_builder import AutoParamBuilder
from pose_extractor import PoseExtractor


class PoseGeometricEvaluator:
    def __init__(self, displacement_threshold=15.0, angle_threshold_deg=10.0):
        self.disp_thresh = displacement_threshold
        self.angle_thresh = angle_threshold_deg

    def _get_intersection(self, p1, p2, p3, p4):
        """计算躯体对角线交点"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0: return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return np.array([px, py])

    def align_poses(self, kpts_orig, kpts_gen, torso_indices):
        """对齐原图与生成图的躯干中心"""
        l_sho = kpts_orig[torso_indices['L_sho']]
        r_hip = kpts_orig[torso_indices['R_hip']]
        r_sho = kpts_orig[torso_indices['R_sho']]
        l_hip = kpts_orig[torso_indices['L_hip']]
        center_orig = self._get_intersection(l_sho, r_hip, r_sho, l_hip)

        l_sho_g = kpts_gen[torso_indices['L_sho']]
        r_hip_g = kpts_gen[torso_indices['R_hip']]
        r_sho_g = kpts_gen[torso_indices['R_sho']]
        l_hip_g = kpts_gen[torso_indices['L_hip']]
        center_gen = self._get_intersection(l_sho_g, r_hip_g, r_sho_g, l_hip_g)

        if center_orig is None or center_gen is None:
            raise ValueError("无法计算躯干中心点，请检查关键点提取是否完整。")

        translation_vector = center_orig - center_gen
        kpts_aligned = {k: v + translation_vector for k, v in kpts_gen.items()}
        return kpts_aligned, translation_vector

    def calculate_vector_angle(self, v1, v2):
        """计算向量夹角"""
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        return math.degrees(math.acos(dot_product))

    def evaluate(self, kpts_orig, kpts_gen, stable_keys, residual_vecs_list, generated_vecs_list, torso_indices):
        kpts_aligned, _ = self.align_poses(kpts_orig, kpts_gen, torso_indices)

        error_reasons = []
        correction_steps = []

        # ---------------------------------------------------------
        # 1. 检查静止点位移 (收集所有超标的关节点)
        # ---------------------------------------------------------
        displaced_joints = []
        for key in stable_keys:
            # 判断在原图和生成图中，该点是否都有效 (置信度 > 0)
            if kpts_orig[key, 2] > 0 and kpts_aligned[key, 2] > 0:
                # [注意] 必须切片 [:2] 只取 X 和 Y 计算欧氏距离
                disp = np.linalg.norm(kpts_orig[key][:2] - kpts_aligned[key][:2])
                if disp > self.disp_thresh:
                    displaced_joints.append((key, disp))

        # 如果有静止点发生偏移
        if displaced_joints:
            # 按偏移量从大到小排序，让模型优先关注偏得最厉害的点
            displaced_joints.sort(key=lambda x: x[1], reverse=True)

            # [注意] key 现在是数字(int)，拼接字符串时需要用 str(k)
            joint_details = ", ".join([f"'Joint {k}' ({d:.1f}px)" for k, d in displaced_joints])
            joint_names = ", ".join([str(k) for k, d in displaced_joints])

            error_reasons.append(
                f"[Displacement] The following joints moved beyond the {self.disp_thresh}px threshold: {joint_details}.")
            correction_steps.append(
                f"Lock the exact positions of these joints: {joint_names}. DO NOT shift the torso or original body parts.")

        # ---------------------------------------------------------
        # 2. 检查残肢生成角度 (带相对方向提示)
        # ---------------------------------------------------------
        for res_vec_keys, gen_vec_keys in zip(residual_vecs_list, generated_vecs_list):
            # [注意] 同样必须切片 [:2] 提取二维向量，抛弃第三维度的置信度
            v_res = kpts_orig[res_vec_keys[1]][:2] - kpts_orig[res_vec_keys[0]][:2]
            v_gen = kpts_aligned[gen_vec_keys[1]][:2] - kpts_aligned[gen_vec_keys[0]][:2]

            angle_diff = self.calculate_vector_angle(v_res, v_gen)

            if angle_diff > self.angle_thresh:
                # 向量叉乘计算相对旋转方向
                cross_product = v_res[0] * v_gen[1] - v_res[1] * v_gen[0]
                direction_hint = "counter-clockwise" if cross_product > 0 else "clockwise"

                error_reasons.append(
                    f"[Angle Error on Joints {gen_vec_keys}] Generated limb deviated by {angle_diff:.1f} degrees.")
                correction_steps.append(
                    f"The limb connected to joint {gen_vec_keys[0]} is pointing incorrectly. Rotate it slightly {direction_hint}.")

        # ---------------------------------------------------------
        # 3. 结果汇总打包
        # ---------------------------------------------------------
        if not error_reasons:
            return {"passed": True, "reason": "Geometric alignment perfect.", "correction": ""}
        else:
            formatted_reasons = "\n".join([f"- {r}" for r in error_reasons])
            formatted_corrections = "\n".join([f"Task {i + 1}: {c}" for i, c in enumerate(correction_steps)])
            return {
                "passed": False,
                "reason": f"Multiple geometric errors detected:\n{formatted_reasons}",
                "correction": f"Please fix ALL the following issues simultaneously:\n{formatted_corrections}"
            }

# ==========================================
# 2. 精细校准 Agent (专职处理第二阶段)
# ==========================================
class GeometricRefinerAgent:
    def __init__(self, pose_extractor: PoseExtractor, edit_model='qwen-image-2.0-pro', disp_thresh=15.0, angle_thresh=10.0, max_iterations=3):
        self.edit_model = edit_model
        self.max_iterations = max_iterations
        self.evaluator = PoseGeometricEvaluator(disp_thresh, angle_thresh)

        self.pose_extractor = pose_extractor
        self.auto_param_builder = AutoParamBuilder()


        # 精细微调的专用基础指令（更强调几何校准）
        self.refine_instruction = """
        [Task: Geometric Limb Calibration]
        Objective: Micro-adjust the posture of the newly generated limb in the provided image.

        [Strict Rules]
        1. DO NOT touch, move, or redraw any original body parts (torso, head, intact limbs). Lock the torso completely.
        2. Adjust ONLY the angle or position of the limb according to the specific [Correction Directive] below.
        3. Maintain photorealistic skin texture and clothing continuity.
        """

    def edit_image(self, image_url, prompt, mask_url=None):
        print(f"\n🎨 [生成] 调用 {self.edit_model} (同步对话模式)...")

        # 1. 组装新版 API 要求的 messages 结构
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_url},
                    {"text": prompt.strip()}
                ]
            }
        ]

        # 如果有 mask_url，按顺序插入到 text 前面
        if mask_url:
            messages[0]["content"].insert(1, {"image": mask_url})

        try:
            # 2. 发起同步调用 (自动等待直到图片生成完毕)
            response = MultiModalConversation.call(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model=self.edit_model,
                messages=messages,
                stream=False,
                n=1,
                watermark=False,
                # 完整保留你精简版的负向提示词
                negative_prompt="shifting torso, changing joint angles, redrawing background, missing limbs",
                # 强烈建议设为 False，防止大模型魔改你的 prompt
                prompt_extend=False
            )

            # 3. 提取返回的干净 URL
            if response.status_code == 200:
                for content in response.output.choices[0].message.content:
                    if 'image' in content:
                        result_url = content['image']
                        return result_url
                raise RuntimeError("❌ API 返回了 200，但未找到图片链接。")
            else:
                error_msg = f"HTTP返回码：{response.status_code}, 错误信息：{response.message}"
                raise RuntimeError(f"❌ 图像生成失败: {error_msg}")

        except Exception as e:
            raise RuntimeError(f"❌ 大模型 API 调用崩溃: {str(e)}")

    def run(self, original_url, initial_gen_url, mask_url=None):
        """
        :param original_url: 原始残肢图片 (Ground Truth 几何参考)
        :param initial_gen_url: 第一步 Agent 生成的初版图片
        """
        print("\n" + "=" * 50)
        print(f"🔬 启动第二阶段: 几何精细校准 Agent")
        print("=" * 50)

        # 1. 提取原图的绝对基准点 (Ground Truth)
        kpts_orig = self.pose_extractor.extract_31_keypoints(original_url)

        current_url = initial_gen_url
        generated_image_urls = [current_url]

        eval_params = self.auto_param_builder.infer_params(kpts_orig)

        for i in range(1, self.max_iterations + 1):
            print(f"\n⚙️ 几何校准轮次 {i}/{self.max_iterations}")

            # 2. 提取当前生成图的位姿
            kpts_gen = self.pose_extractor.extract_31_keypoints(current_url)

            # 3. 严格数学计算
            eval_res = self.evaluator.evaluate(
                kpts_orig=kpts_orig,
                kpts_gen=kpts_gen,
                stable_keys=eval_params["stable_keys"],
                residual_vecs_list=eval_params["residual_vecs_list"],
                generated_vecs_list=eval_params["generated_vecs_list"],
                torso_indices=eval_params["torso_indices"]
            )

            if eval_res["passed"]:
                print("🎯 [校准通过] 所有几何和位移误差均小于阈值！")
                return generated_image_urls
            else:
                print(f"⚠️ [校准未达标] {eval_res['reason']}")

            if i < self.max_iterations:
                print("🔧 生成纠偏指令，提交重新编辑...")
                # 将数学偏差转化为具体指令
                current_prompt = f"{self.refine_instruction}\n\n[Correction Directive]: {eval_res['correction']}"
                try:
                    # 基于第一步的生成图进行微调编辑
                    current_url = self.edit_image(current_url, current_prompt, mask_url)
                    generated_image_urls.append(current_url)
                except Exception as e:
                    print(f"❌ 微调中断: {e}")
                    break
            else:
                print("🛑 已达最大校准次数，返回当前最优微调结果。")

        return generated_image_urls


class AgenticImageEditor:
    def __init__(self, edit_model='qwen-image-2.0-pro', eval_model='qwen3.6-plus'):
        self.edit_model = edit_model
        self.eval_model = eval_model
        self.max_iterations = 3
        self.base_instruction = """
        [Task: High-Precision Anatomical Inpainting]
        Objective: Restore missing limbs to create a realistic able-bodied person.

        [Hard Constraints - MUST FOLLOW]
        - Preserve Original: Keep existing torso, head, and limbs 100% unchanged.
        - Anatomical Alignment: Extend the new limb strictly along the direction of the residual limb.
        - Joint Integrity: DO NOT alter any original joint angles or body posture.
        - Completeness: Ensure the final output has two arms and two legs with visible hands/feet.

        [Visual Consistency]
        - Seamlessly match original skin texture, lighting, and clothing style.
        """

    def edit_image(self, image_url, prompt, mask_url=None):
        print(f"\n🎨 [生成] 调用 {self.edit_model} (同步对话模式)...")

        # 1. 组装符合新版 API 要求的 messages 结构
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_url},
                    {"text": prompt.strip()}
                ]
            }
        ]

        # 如果有 mask_url，也塞进 content 列表里
        if mask_url:
            messages[0]["content"].insert(1, {"image": mask_url})

        try:
            # 2. 发起同步调用 (程序会在这里耐心等待，直到阿里云把图画完)
            response = MultiModalConversation.call(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model=self.edit_model,
                messages=messages,
                stream=False,
                n=1,
                watermark=False,
                # 你的负向提示词完整保留
                negative_prompt="deformed torso, changing existing limb angles, distorted joints, redrawing entire person, extra fingers, anatomical nonsense, missing limbs",
                # 强烈建议设为 False，防止模型自己乱加词破坏你的严苛医学约束
                prompt_extend=False
            )

            # 3. 解析并返回干净的 URL
            if response.status_code == 200:
                for content in response.output.choices[0].message.content:
                    if 'image' in content:
                        result_url = content['image']
                        print(f"✅ 生成成功，新图像 URL: {result_url}")
                        return result_url
                raise RuntimeError("❌ API 返回了 200 成功，但内容里没有图片链接。")
            else:
                error_msg = f"HTTP返回码：{response.status_code}, 错误信息：{response.message}"
                raise RuntimeError(f"❌ 图像生成失败: {error_msg}")

        except Exception as e:
            # 捕获底层的网络断连、超时等异常，直接抛出清晰的报错
            raise RuntimeError(f"❌ 大模型 API 调用崩溃: {str(e)}")

    def evaluate_image(self, original_url, current_url, original_prompt):
        print(f"\n🔍 [评估] 调用 {self.eval_model} 进行视觉审视...")

        eval_prompt = f"""You are a strict Image Quality Auditor. Compare the ORIGINAL and EDITED images.
        Verify if the output meets these requirements: {original_prompt}

        CRITICAL INSPECTION POINTS:
        1. Are all 4 limbs (2 arms, 2 legs) present and complete?
        2. Is the original torso and existing limbs UNCHANGED?
        3. Does the new limb follow the natural direction of the stump without changing joint angles?

        Output ONLY a valid JSON object:
        {{
          "passed": true/false,
          "reason": "Detailed explanation of violations in English (e.g., 'torso distorted', 'limb angle mismatch')",
          "suggestion": "Specific correction command for the next iteration in English"
        }}"""

        resp = MultiModalConversation.call(
            model=self.eval_model,
            messages=[{
                "role": "user",
                "content": [
                    {"image": original_url},
                    {"image": current_url},
                    {"text": eval_prompt}
                ]
            }],
            response_format={'type': 'json_object'}
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
            return {"passed": False, "reason": "JSON_PARSE_ERROR",
                    "suggestion": "Re-generate following the original constraints strictly."}

    def run(self, image_url, mask_url=None):
        generated_image_urls = []
        current_url = image_url
        current_prompt = self.base_instruction
        print(f"🚀 启动 Agentic 编辑流程 | 模型: {self.edit_model} + {self.eval_model}")

        for i in range(1, self.max_iterations + 1):
            print(f"\n{'=' * 40} 第 {i}/{self.max_iterations} 轮 {'=' * 40}")

            # 1. 执行编辑
            current_url = self.edit_image(current_url, current_prompt, mask_url)
            generated_image_urls.append(current_url)


            # 2. 自我审视（最后一轮直接返回）
            if i < self.max_iterations:
                eval_res = self.evaluate_image(image_url, current_url, self.base_instruction)
                print(f"📊 评估: {'✅ 通过' if eval_res['passed'] else '❌ 未通过'}")
                print(f"📝 原因: {eval_res.get('reason', '-')}")

                if eval_res["passed"]:
                    print("🎉 约束全部满足，提前结束迭代！")
                    return generated_image_urls

                # 3. 动态注入修正指令
                print("🛠️ 注入修正指令，准备下一轮生成...")
                current_prompt = f"{self.base_instruction}\n\n[Correction Directive]: {eval_res['suggestion']}"
            else:
                print("⚠️ 已达最大迭代次数，返回当前最佳结果。")

        return generated_image_urls