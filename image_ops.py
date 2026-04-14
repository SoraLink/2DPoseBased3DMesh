import cv2
import numpy as np
import base64
import mimetypes

from dotenv import load_dotenv
from tqdm import tqdm


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


class OSSProgressTracker:
    """用于 OSS 上传的终端进度条回调类"""

    def __init__(self, description="上传中"):
        self.pbar = None
        self.description = description

    def __call__(self, consumed_bytes, total_bytes):
        if total_bytes:
            if self.pbar is None:
                self.pbar = tqdm(total=total_bytes, unit='B', unit_scale=True, desc=self.description)
            self.pbar.update(consumed_bytes - self.pbar.n)
            if consumed_bytes == total_bytes:
                self.pbar.close()


class OSSProcessor:
    def __init__(self):
        self.access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
        self.access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET')
        self.endpoint = os.getenv('OSS_ENDPOINT')  # 如: oss-cn-beijing.aliyuncs.com
        self.bucket_name = os.getenv('OSS_BUCKET_NAME')

        if not all([self.access_key_id, self.access_key_secret, self.endpoint, self.bucket_name]):
            raise ValueError("❌ OSS 环境变量未配置完整，请检查。")

        self.auth = oss2.Auth(self.access_key_id, self.access_key_secret)

        # 全局超时 & 重试配置
        oss2.defaults.connect_timeout = 60
        oss2.defaults.socket_timeout = 120  # 🔑 新增：读写超时
        oss2.defaults.request_retries = 5

        session = oss2.Session(pool_size=10)
        self.bucket = oss2.Bucket(
            self.auth,
            self.endpoint,
            self.bucket_name,
            session=session
        )

    def _build_public_url(self, object_name):
        """🔑 核心：构造公开可读的永久链接"""
        # endpoint 示例: oss-cn-beijing.aliyuncs.com
        # 目标格式: https://{bucket}.oss-cn-beijing.aliyuncs.com/{object_name}

        # 提取 region 部分（兼容不同 endpoint 格式）
        if self.endpoint.startswith('http'):
            domain = self.endpoint.split('://', 1)[1]
        else:
            domain = self.endpoint

        # 确保 bucket 在域名最前面
        if self.bucket_name in domain:
            # 已经是 {bucket}.oss-xxx 格式，直接使用
            public_domain = domain
        else:
            # 标准格式：bucket.oss-region.aliyuncs.com
            public_domain = f"{self.bucket_name}.{domain}"

        return f"https://{public_domain}/{object_name}"

    def upload_and_get_url(self, local_file_path, folder="agent_images", make_public=True):
        """
        上传文件到 OSS 并返回 URL

        Args:
            local_file_path: 本地文件路径
            folder: OSS 中的文件夹路径
            make_public: 🔑 是否设置为公共读（默认 True，供 DashScope 访问）
        """
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"❌ 找不到要上传的文件: {local_file_path}")

        file_extension = os.path.splitext(local_file_path)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        object_name = f"{folder}/{unique_filename}"

        print(f"☁️ 准备上传至 OSS: {object_name}")
        tracker = OSSProgressTracker(description=f"上传 {os.path.basename(local_file_path)}")

        try:
            # 分片断点续传上传
            oss2.resumable_upload(
                self.bucket,
                object_name,
                local_file_path,
                multipart_threshold=100 * 1024,
                part_size=100 * 1024,
                num_threads=3,
                progress_callback=tracker
            )
            print(f"\n✅ 文件上传成功: {object_name}")

            # 🔑 关键 1: 设置为公共读权限（让 DashScope 能直接访问）
            if make_public:
                self.bucket.put_object_acl(object_name, oss2.OBJECT_ACL_PUBLIC_READ)
                print("🔓 已设置对象为公共读权限")

            # 🔑 关键 2: 返回公开永久链接（非签名链接）
            public_url = self._build_public_url(object_name)
            print(f"🌐 公开访问链接: {public_url}")

            return public_url

        except oss2.exceptions.RequestError as e:
            raise RuntimeError(f"❌ 上传网络异常: {e}")
        except oss2.exceptions.OssError as e:
            raise RuntimeError(f"❌ OSS 服务错误: {e.code} - {e.message}")