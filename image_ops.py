import cv2
import numpy as np
import base64
import mimetypes

from dotenv import load_dotenv


class ImageProcessor:
    @staticmethod
    def encode_to_base64(file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "image/jpeg"
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"

    @staticmethod
    def generate_safe_mask(img_shape, stump_kpts, intact_kpts=None, extension_length=400) -> np.ndarray:
        """生成自动向下延伸且避开好腿的掩码"""
        mask_shape = img_shape[:2]
        joint_upper, joint_stump = stump_kpts

        # 1. 基础残肢延伸 Mask (攻击区)
        stump_mask = np.zeros(mask_shape, dtype=np.uint8)
        dx = joint_stump[0] - joint_upper[0]
        dy = max(1, joint_stump[1] - joint_upper[1])  # 防止倒立出错

        length = np.hypot(dx, dy)
        dir_x, dir_y = dx / length, dy / length

        end_pt = (int(joint_stump[0] + dir_x * extension_length),
                  int(joint_stump[1] + dir_y * extension_length))
        stump_pt = (int(joint_stump[0]), int(joint_stump[1]))

        cv2.line(stump_mask, stump_pt, end_pt, 255, thickness=150)
        cv2.circle(stump_mask, stump_pt, radius=75, color=255, thickness=-1)

        # 2. 生成保护区 (如果提供了好腿的关键点)
        if intact_kpts and len(intact_kpts) >= 2:
            protection_mask = np.zeros(mask_shape, dtype=np.uint8)
            for i in range(len(intact_kpts) - 1):
                pt1 = (int(intact_kpts[i][0]), int(intact_kpts[i][1]))
                pt2 = (int(intact_kpts[i + 1][0]), int(intact_kpts[i + 1][1]))
                cv2.line(protection_mask, pt1, pt2, 255, thickness=180)

            kernel = np.ones((15, 15), np.uint8)
            protection_mask = cv2.dilate(protection_mask, kernel, iterations=1)
            # 相减抠除
            stump_mask = cv2.bitwise_and(stump_mask, cv2.bitwise_not(protection_mask))

        # 3. 边缘羽化
        safe_mask = cv2.GaussianBlur(stump_mask, (41, 41), 0)
        _, safe_mask = cv2.threshold(safe_mask, 127, 255, cv2.THRESH_BINARY)
        return safe_mask

    @staticmethod
    def kinematic_late_fusion(orig_path: str, gen_path: str, mask: np.ndarray, save_path: str) -> str:
        """Alpha Blending：保证未截肢区域 100% 像素保真"""
        print("[Fusion] 执行生成图与原图的像素级融合...")
        orig_img = cv2.imread(orig_path)
        gen_img = cv2.imread(gen_path)

        if orig_img.shape != gen_img.shape:
            gen_img = cv2.resize(gen_img, (orig_img.shape[1], orig_img.shape[0]))

        # 对 Mask 进行高阶高斯模糊，实现自然羽化过渡
        blurred_mask = cv2.GaussianBlur(mask, (51, 51), 0)
        alpha = blurred_mask.astype(float) / 255.0
        alpha = np.expand_dims(alpha, axis=-1)

        # 融合: 掩码区用生成图，背景用原图
        blended_float = gen_img.astype(float) * alpha + orig_img.astype(float) * (1.0 - alpha)
        blended = np.clip(blended_float, 0, 255).astype(np.uint8)

        cv2.imwrite(save_path, blended)
        return save_path


import oss2
import os
import uuid
load_dotenv()

class OSSProcessor:
    def __init__(self):
        # 强烈建议将这些敏感信息配置在操作系统的环境变量中
        self.access_key_id = os.getenv('OSS_ACCESS_KEY_ID', '你的AccessKeyId')
        self.access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET', '你的AccessKeySecret')
        # Endpoint 示例：'oss-cn-beijing.aliyuncs.com'
        self.endpoint = os.getenv('OSS_ENDPOINT', '你的Endpoint')
        self.bucket_name = os.getenv('OSS_BUCKET_NAME', '你的Bucket名称')

        # 校验配置
        if not all([self.access_key_id, self.access_key_secret, self.endpoint, self.bucket_name]):
            raise ValueError("❌ OSS 环境变量未配置完整，请检查。")

        # 初始化认证和 Bucket 实例
        self.auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)

    def upload_and_get_url(self, local_file_path, folder="agent_images"):
        """
        上传本地文件到 OSS 并返回带签名的临时访问 URL。
        """
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"❌ 找不到要上传的文件: {local_file_path}")

        # 1. 生成云端唯一的文件名 (防止同名文件覆盖)
        file_extension = os.path.splitext(local_file_path)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        object_name = f"{folder}/{unique_filename}"

        # 2. 执行上传
        print(f"☁️ 正在将图像上传至 OSS: {object_name} ...")
        self.bucket.put_object_from_file(object_name, local_file_path)

        # 3. 生成带签名的访问 URL (有效时间 3600 秒 = 1小时)
        # 这样做的好处是：你的 Bucket 可以保持"私有"读写权限，极其安全，
        # 同时大模型 API 也能通过这个临时 URL 顺利拉取到图片。
        signed_url = self.bucket.sign_url('GET', object_name, 3600)

        print(f"✅ 上传成功！获取临时访问链接。")
        return signed_url