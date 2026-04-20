import tempfile

import cv2
import numpy as np
import base64
import mimetypes

import requests
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm


class ImageProcessor:
    @staticmethod
    def encode_file(file_path):
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError("不支持或无法识别的图像格式")

        try:
            with open(file_path, "rb") as image_file:
                encoded_string = base64.b64encode(
                    image_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_string}"
        except IOError as e:
            raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")

    @staticmethod
    def save_image_from_url(url, source, iter, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{source}_{iter}.jpg"

        save_path = os.path.join(save_dir, filename)

        try:
            print(f"⬇️ 正在下载图片: {url[:50]}...")
            response = requests.get(url, timeout=15)
            response.raise_for_status()  # 检查请求是否成功

            with open(save_path, 'wb') as file:
                file.write(response.content)

            print(f"💾 成功保存到本地: {save_path}")
        except requests.exceptions.RequestException as e:
            print(f"❌ 下载失败: {e}")
        return save_path

    @staticmethod
    def enforce_pure_black_background(image_path, sam2_predictor, kpts_gen, types_orig):
        """
        强制洗图：用 SAM2 重新提取人物轮廓，把背景一切非人物像素暴力置为绝对纯黑 (0,0,0)
        并另存为带有 '_black_bg' 后缀的新文件，避免覆盖原图。
        """
        print("🧹 [洗图] 正在清除大模型产生的背景噪声...")

        # 1. 重新提取当前图的 Mask
        mask = sam2_predictor.get_solid_mask(
            image_path,
            kpts_gen,
            types_orig
        )

        if mask is None:
            print("⚠️ 无法获取 Mask，跳过背景清洗。")
            return image_path

        # 2. 读取原图
        img_bgr = cv2.imread(image_path)

        # 3. 强制黑底 (Mask 为 0 的地方全部赋值为 [0, 0, 0])
        # 注意：mask 是 255 (前景) 和 0 (背景)
        img_bgr[mask < 127] = [0, 0, 0]

        # 4. 构建新的保存路径 (image name + _black_bg)
        dirname = os.path.dirname(image_path)
        basename = os.path.basename(image_path)
        name, ext = os.path.splitext(basename)
        new_filename = f"{name}_black_bg{ext}"
        new_image_path = os.path.join(dirname, new_filename)

        # 5. 保存到新路径
        cv2.imwrite(new_image_path, img_bgr)
        print(f"✨ 背景已恢复至绝对纯黑！已保存为: {new_image_path}")

        return new_image_path


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
        将图片转换为 JPG 后上传文件到 OSS 并返回 URL

        Args:
            local_file_path: 本地文件路径
            folder: OSS 中的文件夹路径
            make_public: 🔑 是否设置为公共读（默认 True，供 DashScope 访问）
        """
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"❌ 找不到要上传的文件: {local_file_path}")

        # 获取原始后缀
        original_extension = os.path.splitext(local_file_path)[1].lower()

        file_to_upload = local_file_path
        file_extension = original_extension
        temp_jpg_path = None

        # === 新增：如果不是 JPG/JPEG，则尝试转换为 JPG ===
        if original_extension not in ['.jpg', '.jpeg']:
            try:
                print(f"🔄 正在将图片格式 {original_extension} 转换为 JPG...")
                with Image.open(local_file_path) as img:
                    # 处理带透明通道的图片(PNG等)，防止转换为JPEG时背景变黑/报错
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        # 如果有透明通道，将原图粘贴到白色背景上
                        if img.mode == 'RGBA':
                            background.paste(img, mask=img.split()[3])
                        else:
                            background.paste(img)
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')

                    # 生成临时文件路径
                    fd, temp_jpg_path = tempfile.mkstemp(suffix='.jpg')
                    os.close(fd)  # 关闭底层文件描述符

                    # 保存为 JPG，quality=95 保证图片质量
                    img.save(temp_jpg_path, format='JPEG', quality=95)

                    # 更新接下来要上传的文件路径和后缀
                    file_to_upload = temp_jpg_path
                    file_extension = '.jpg'
            except Exception as e:
                # 如果转换失败（比如传入的根本不是图片文件而是txt），则继续上传原文件
                print(f"⚠️ 图片转 JPG 失败（可能非图片文件），将原样上传。错误信息: {e}")
                if temp_jpg_path and os.path.exists(temp_jpg_path):
                    os.remove(temp_jpg_path)
                    temp_jpg_path = None

        # 生成唯一的 OSS 对象名称
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        object_name = f"{folder}/{unique_filename}"

        print(f"☁️ 准备上传至 OSS: {object_name}")
        tracker = OSSProgressTracker(description=f"上传 {unique_filename}")

        try:
            # 分片断点续传上传（注意这里使用的是 file_to_upload）
            oss2.resumable_upload(
                self.bucket,
                object_name,
                file_to_upload,
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

        finally:
            # === 新增：无论上传成功与否，都要清理产生的本地临时 JPG 文件 ===
            if temp_jpg_path and os.path.exists(temp_jpg_path):
                try:
                    os.remove(temp_jpg_path)
                    # print("🧹 临时 JPG 文件已清理")
                except Exception as e:
                    print(f"⚠️ 清理临时文件失败: {e}")