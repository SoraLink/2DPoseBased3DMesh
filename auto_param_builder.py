import numpy as np


class AutoParamBuilder:
    def __init__(self, score_threshold=0.3):
        self.score_threshold = score_threshold

        # 躯干基准点ID: 5(L_sho), 6(R_sho), 11(L_hip), 12(R_hip)
        self.torso_indices = {'L_sho': 5, 'R_sho': 6, 'L_hip': 11, 'R_hip': 12}

        # 定义核心17个原生点作为默认静止候选点
        self.standard_keys = list(range(17))

        # 残肢判定规则库:
        # ID: [大臂/大腿残端, 小臂/小腿残端]
        # regen_ids: 需要模型重新生成的关节点ID (将从静止点中剔除)
        self.amputation_rules = {
            23: {"name": "左臂-肘上截肢", "proximal": 5, "mid": 7, "regen_ids": [7, 9, 17]},
            25: {"name": "左臂-肘下截肢", "proximal": 7, "mid": 9, "regen_ids": [9, 17]},
            24: {"name": "右臂-肘上截肢", "proximal": 6, "mid": 8, "regen_ids": [8, 10, 18]},
            26: {"name": "右臂-肘下截肢", "proximal": 8, "mid": 10, "regen_ids": [10, 18]},
            27: {"name": "左腿-膝上截肢", "proximal": 11, "mid": 13, "regen_ids": [13, 15, 19, 21]},
            29: {"name": "左腿-膝下截肢", "proximal": 13, "mid": 15, "regen_ids": [15, 19, 21]},
            28: {"name": "右腿-膝上截肢", "proximal": 12, "mid": 14, "regen_ids": [14, 16, 20, 22]},
            30: {"name": "右腿-膝下截肢", "proximal": 14, "mid": 16, "regen_ids": [16, 20, 22]},
        }

    def infer_params(self, kpts_orig: np.ndarray):
        """
        根据原图关键点，推断评价所需的四个参数（支持多处截肢同时处理）
        """
        valid_residual_ids = []

        # 1. 找出【所有】置信度大于阈值的残肢点
        for res_id in range(23, 31):
            if kpts_orig[res_id, 2] > self.score_threshold:
                valid_residual_ids.append(res_id)

        if not valid_residual_ids:
            raise ValueError("⚠️ 无法在原图中检测到有效的残肢点 (ID 23-30)，请检查图片或降低置信度阈值。")

        residual_vecs = []
        generated_vecs = []
        all_regen_ids = set()  # 用集合来取并集，防止重复
        detected_types = []

        # 2. 遍历所有检测到的截肢部位
        for res_id in valid_residual_ids:
            rule = self.amputation_rules[res_id]
            detected_types.append(rule['name'])

            # 将每一处截肢的向量对加入列表 (注意这里变成了复数列表)
            residual_vecs.append((rule['proximal'], res_id))
            generated_vecs.append((rule['proximal'], rule['mid']))

            # 合并所有需要重绘的关节点
            all_regen_ids.update(rule['regen_ids'])

        print(f"🦴 检测到截肢部位: {', '.join(detected_types)}")

        # 3. 构造静止点：所有标准点 减去 所有需要重绘的点
        stable_keys = [k for k in self.standard_keys if k not in all_regen_ids]

        return {
            "torso_indices": self.torso_indices,
            "stable_keys": stable_keys,
            "residual_vecs_list": residual_vecs,  # [注意] 变成了 List
            "generated_vecs_list": generated_vecs  # [注意] 变成了 List
        }