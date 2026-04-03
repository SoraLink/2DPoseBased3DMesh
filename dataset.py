import os
import random

import cv2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torch
import numpy as np

class AmputeeCOCODataset(Dataset):
    def __init__(self, data_root='./data', is_train=True):
        self.data_root = data_root
        self.is_train = is_train
        # 这里假设你已经加载了 COCO 的 annotations 列表
        self.annotations = [...]

    def __len__(self):
        return len(self.annotations)

    def _simulate_amputation(self, image, densepose_mask, kp_t, kp_a, cut_ratio):
        """
        核心向量裁剪魔法：根据有符号距离 (Signed Distance) 像素级切断肢体
        kp_t: 近端点 (例如膝盖), kp_a: 远端点 (例如脚踝)
        """
        vec_bone = np.array(kp_a) - np.array(kp_t)
        # 计算 2D 物理断点 R_r
        R_r = np.array(kp_t) + cut_ratio * vec_bone

        H, W = image.shape[:2]
        Y, X = np.indices((H, W))

        # 矩阵化暴算全图像素到断点 R_r 的向量
        vec_P_x = X - R_r[0]
        vec_P_y = Y - R_r[1]

        # 点乘计算方向
        dot_product = vec_P_x * vec_bone[0] + vec_P_y * vec_bone[1]

        # 假设 DensePose 中左小腿的 ID 是 13 (这里作为示例)
        is_lower_leg = (densepose_mask == 13)
        # 如果是小腿像素，且在断点靠脚踝的一侧 (>0)
        to_remove = is_lower_leg & (dot_product > 0)

        # 涂黑图像
        image[to_remove] = 0
        return image, R_r

    def __getitem__(self, idx):
        # 1. 读取原图和 DensePose Mask
        # image = cv2.imread(...)
        # densepose_mask = np.load(...)

        # 模拟数据初始化 (25个关键点坐标 [x, y], 默认全是 -1)
        coords = np.zeros((25, 2), dtype=np.float32)
        # 状态标签：0=可见(正常点或Rr), 1=MASK(被切掉的Ja), 2=NULL(健全人不存在的Rr)
        status = np.full(25, 2, dtype=np.int64)

        # 2. 模拟抛硬币决定这张图是否生成截肢
        is_amputee = np.random.rand() < 0.5

        if is_amputee:
            # 随机在 30% 到 80% 的位置下刀
            cut_ratio = np.random.uniform(0.3, 0.8)
            # image, R_r = self._simulate_amputation(image, densepose_mask, kp_knee, kp_ankle, cut_ratio)

            # --- 给 Token 贴状态标签 ---
            # 假设 index 13 是左膝盖 (原远端点 Ja)，被截断了
            coords[13] = [0, 0]  # 坐标无所谓，反正会被替换
            status[13] = 1  # [MASK] 标记，等待模型预测

            # 假设 index 20 是新增的左小腿残肢点 (Rr)
            coords[20] = R_r  # 赋予真实的截断坐标
            status[20] = 0  # 设为可见，作为模型的先验提示

        else:
            # 健全人
            # 左膝盖正常可见
            # coords[13] = kp_knee
            status[13] = 0

            # 残肢点根本不存在
            coords[20] = [0, 0]
            status[20] = 2  # [NULL] 标记，要求模型忽略

        # 注意：你需要把坐标除以图片宽高，归一化到 [-1, 1]
        # coords = (coords / [W, H]) * 2.0 - 1.0

        # 将 image 转为 Tensor (C, H, W)
        return {"image": image_tensor, "coords": torch.tensor(coords), "status": torch.tensor(status)}


class DeterministicAmputeeEvalDataset(Dataset):
    def __init__(self, data_root='./data'):
        self.data_root = data_root
        # 假设这里加载了 COCO Validation 集的 annotations
        self.annotations = [1, 2, 3, 4, 5]  # 占位符，代表你的真实数据列表

        # =======================================================
        # 核心魔法：预生成并“冻结”验证集的随机参数
        # =======================================================
        self.fixed_params = []

        # 极其重要：声明一个局部的、固定 Seed 的随机数生成器
        # 只要 Seed 是 42，无论你在哪台电脑、哪年哪月跑这段代码，
        # 下面生成的 5000 张图的截肢位置绝对一模一样！
        rng = np.random.RandomState(42)

        for _ in range(len(self.annotations)):
            # 为每一张图提前决定命运
            is_amputee = rng.rand() < 0.5
            # 即使不是残疾，也先摇一个比例占位，保证随机数序列的严密性
            cut_ratio = rng.uniform(0.3, 0.8)

            self.fixed_params.append({
                'is_amputee': is_amputee,
                'cut_ratio': cut_ratio
            })

        print(f"✅ Eval Dataset 已锁定，共 {len(self.annotations)} 条固定的测试配置。")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 1. 加载图像 (伪代码)
        # image = cv2.imread(...)
        # densepose_mask = np.load(...)

        # 2. 读取这张图“命中注定”的参数
        params = self.fixed_params[idx]
        is_amputee = params['is_amputee']
        cut_ratio = params['cut_ratio']

        # 初始化坐标和状态
        coords = np.zeros((25, 2), dtype=np.float32)
        status = np.full(25, 2, dtype=np.int64)

        # 3. 严格按照 fixed_params 执行切割
        if is_amputee:
            # 执行你的向量裁剪代码 (同 Training，但传入固定的 cut_ratio)
            # image, R_r = self._simulate_amputation(image, densepose_mask, kp_knee, kp_ankle, cut_ratio)

            # coords[13] = [0, 0] # J_a 被切掉
            status[13] = 1  # [MASK]
            # coords[20] = R_r    # R_r 是通过固定的 cut_ratio 算出来的，绝对一致
            status[20] = 0  # [VISIBLE]
        else:
            # 健全人处理
            status[13] = 0  # [VISIBLE]
            status[20] = 2  # [NULL]

        # 返回固定的测试张量
        return {"image": torch.randn(3, 224, 224), "coords": torch.tensor(coords), "status": torch.tensor(status)}


class RealBaselineHeatmapDataset(Dataset):
    def __init__(self, coco_ann_file, dense_pose_ann_file, img_size=(256, 192)):
        self.img_size = img_size
        self.coco = COCO(coco_ann_file)
        self.dense_pose_ann_file = dense_pose_ann_file
        self.amputation_types = [
            {'name': 'Above Left Elbow', 'limb_id': 'left_arm', 'p': 5, 'd': 7, 'r': 17, 'lost_pts': [7, 9],
             'dp_parts': [15, 17]},
            {'name': 'Below Left Elbow', 'limb_id': 'left_arm', 'p': 7, 'd': 9, 'r': 19, 'lost_pts': [9],
             'dp_parts': [19, 21]},
            {'name': 'Above Right Elbow', 'limb_id': 'right_arm', 'p': 6, 'd': 8, 'r': 18, 'lost_pts': [8, 10],
             'dp_parts': [16, 18]},
            {'name': 'Below Right Elbow', 'limb_id': 'right_arm', 'p': 8, 'd': 10, 'r': 20, 'lost_pts': [10],
             'dp_parts': [20, 22]},
            {'name': 'Above Left Knee', 'limb_id': 'left_leg', 'p': 11, 'd': 13, 'r': 21, 'lost_pts': [13, 15],
             'dp_parts': [8, 10]},
            {'name': 'Below Left Knee', 'limb_id': 'left_leg', 'p': 13, 'd': 15, 'r': 23, 'lost_pts': [15],
             'dp_parts': [12, 14]},
            {'name': 'Above Right Knee', 'limb_id': 'right_leg', 'p': 12, 'd': 14, 'r': 22, 'lost_pts': [14, 16],
             'dp_parts': [7, 9]},
            {'name': 'Below Right Knee', 'limb_id': 'right_leg', 'p': 14, 'd': 16, 'r': 24, 'lost_pts': [16],
             'dp_parts': [11, 13]},
        ]

        ann_ids = self.coco.getAnnIds(iscrowd=False)
        self.valid_anns = []

        for ann_id in ann_ids:
            ann = self.coco.loadAnns(ann_id)[0]
            if 'keypoints' in ann and ann['num_keypoints'] > 5:
                self.valid_anns.append(ann)
        print(f"✅ Finished loading {len(self.valid_anns)} valid annotations.")

    def __len__(self):
        return len(self.valid_anns)

    def __getitem__(self, idx):
        ann = self.valid_anns[idx]
        kpts = ann['keypoints']
        ann_id = ann['id']
        image_id = ann['image_id']

        x_min, y_min, w, h = ann['bbox']
        if w < 10 or h < 10:
            w, h = 10, 10

        H_net, W_net = self.img_size
        coords = np.zeros((25, 2), dtype=np.float32)
        status = np.full(25, 2, dtype=np.int64)

        for i in range(17):
            x_abs, y_abs, v = kpts[i * 3], kpts[i * 3 + 1], kpts[i * 3 + 2]
            if v > 0:
                x_rel = x_abs - x_min
                y_rel = y_abs - y_min
                x_net = x_rel * (W_net / w)
                y_net = y_rel * (H_net / h)
                coords[i] = [x_net, y_net]
                status[i] = 0
            else:
                coords[i] = [0, 0]

        amputated_limbs = set()

        indices = list(range(len(self.amputation_types)))
        random.shuffle(indices)

        for idx in indices:
            amputation = self.amputation_types[idx]
            limb_id = amputation['limb_id']

            if limb_id in amputated_limbs:
                continue

            if random.random() < 0.5:
                p, d, r = amputation['p'], amputation['d'], amputation['r']
                lost_pts, dp_parts = amputation['lost_pts'], amputation['dp_parts']

                if status[p] == 0 and status[d] == 0:
                    limb_mask = self._load_densepose_mask(image_id, ann_id, dp_parts, ann['bbox'])

                    cut_ratio = random.uniform(0.1, 0.9)
                    R_r = self._sample_residual_point_with_mask(coords[p], coords[d], cut_ratio, limb_mask)

                    for pt in lost_pts:
                        status[pt] = 2

                    status[d] = 1
                    coords[r] = R_r
                    status[r] = 0

                    amputated_limbs.add(limb_id)

        input_heatmaps = np.zeros((25, self.img_size[0], self.img_size[1]), dtype=np.float32)
        for i in range(25):
            if status[i] == 0:
                input_heatmaps[i] = self._generate_gaussian_heatmap_optimized(coords[i], self.img_size)

        gt_heatmaps = np.zeros((8, self.img_size[0], self.img_size[1]), dtype=np.float32)
        target_weights = np.zeros((8,), dtype=np.float32)

        for i, amputation in enumerate(self.amputation_types):
            target_d = amputation['d']
            if status[target_d] == 1:
                gt_heatmaps[i] = self._generate_gaussian_heatmap_optimized(coords[target_d], self.img_size)
                target_weights[i] = 1.0

        return {
            "input_heatmaps": torch.from_numpy(input_heatmaps),
            "gt_heatmaps": torch.from_numpy(gt_heatmaps),
            "target_weights": torch.from_numpy(target_weights)
        }

    def _load_densepose_mask(self, image_id, ann_id, dp_parts, bbox):
        H_net, W_net = self.img_size
        mask_filename = f"{image_id:012d}_{ann_id}.png"
        mask_path = os.path.join(self.dense_pose_ann_file, mask_filename)

        if not os.path.exists(mask_path):
            raise ValueError('Requested mask file does not exist: ' + mask_path)

        person_isolated_matrix = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        limb_mask_full_res = np.isin(person_isolated_matrix, dp_parts).astype(np.uint8)

        x, y, w, h = [int(v) for v in bbox]
        H_orig, W_orig = limb_mask_full_res.shape

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W_orig, x + w), min(H_orig, y + h)

        if x2 <= x1 or y2 <= y1:
            raise ValueError('Requested mask is out of bounds: ' + mask_path)

        patch_mask = limb_mask_full_res[y1:y2, x1:x2]

        limb_mask_resized = cv2.resize(
            patch_mask,
            (W_net, H_net),
            interpolation=cv2.INTER_NEAREST
        )

        return limb_mask_resized

    def _sample_residual_point_with_mask(self, coords_p, coords_d, cut_ratio, limb_mask):

        vec_bone = coords_d - coords_p
        R_base = coords_p + cut_ratio * vec_bone # Cut Point

        bone_length = np.linalg.norm(vec_bone)
        if bone_length < 1e-5:
            raise ValueError('Bone length is too small: ' + str(bone_length))

        unit_v = vec_bone / bone_length
        unit_u = np.array([-unit_v[1], unit_v[0]]) # vertical unit vector

        H, W = limb_mask.shape
        x_base, y_base = int(R_base[0]), int(R_base[1])

        if not (0 <= x_base < W and 0 <= y_base < H):
            raise ValueError('Residual point is out of bounds: ' + str(R_base))

        def cast_ray(direction_sign):
            dist = 0
            max_dist = max(W, H)
            while dist < max_dist:
                cur_x = int(R_base[0] + dist * unit_u[0] * direction_sign)
                cur_y = int(R_base[1] + dist * unit_u[1] * direction_sign)
                if cur_x < 0 or cur_x >= W or cur_y < 0 or cur_y >= H:
                    break
                if limb_mask[cur_y, cur_x] == 0:
                    break
                dist += 1
            return dist

        dist_pos = cast_ray(1)
        dist_neg = cast_ray(-1)
        total_width = dist_pos + dist_neg

        if total_width < 2:
            raise ValueError('Ray width is too small: ' + str(total_width))

        sigma = max(total_width / 6.0, 1.0) # 3 Times Sigma Rule

        max_tries = 10
        for _ in range(max_tries):
            offset = np.random.normal(loc=0.0, scale=sigma)

            if -dist_neg <= offset <= dist_pos:
                R_r_new = R_base + offset * unit_u
                r_x, r_y = int(R_r_new[0]), int(R_r_new[1])

                if 0 <= r_x < W and 0 <= r_y < H and limb_mask[r_y, r_x] > 0:
                    return R_r_new

        return R_base

    def _generate_gaussian_heatmap_optimized(self, center, img_size=(256, 192), sigma=2.0):
        H, W = img_size
        heatmap = np.zeros((H, W), dtype=np.float32)
        x_c, y_c = int(center[0]), int(center[1])

        if x_c < 0 or x_c >= W or y_c < 0 or y_c >= H:
            raise ValueError('Center is out of bounds: ' + str(center))

        radius = int(3 * sigma)
        x1 = max(0, x_c - radius)
        y1 = max(0, y_c - radius)
        x2 = min(W, x_c + radius + 1)
        y2 = min(H, y_c + radius + 1)

        local_size = 2 * radius + 1
        X_local, Y_local = np.meshgrid(np.arange(local_size), np.arange(local_size))

        dist_sq = (X_local - radius) ** 2 + (Y_local - radius) ** 2
        gaussian_patch = np.exp(-dist_sq / (2 * sigma ** 2))

        patch_x1 = x1 - (x_c - radius)
        patch_y1 = y1 - (y_c - radius)
        patch_x2 = patch_x1 + (x2 - x1)
        patch_y2 = patch_y1 + (y2 - y1)

        heatmap[y1:y2, x1:x2] = gaussian_patch[patch_y1:patch_y2, patch_x1:patch_x2]

        return heatmap