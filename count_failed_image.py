from pathlib import Path

# 修改成你的根目录
ROOT_DIR = Path("./workdir3")

# 常见图片后缀
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

more_than_one = []
only_one = []
zero_image = []

for subdir in ROOT_DIR.iterdir():
    if not subdir.is_dir():
        continue

    images = [
        p for p in subdir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    count = len(images)

    if count > 1:
        more_than_one.append((subdir.name, count))
    elif count == 1:
        only_one.append((subdir.name, count))
    else:
        zero_image.append((subdir.name, count))

print("====== 统计结果 ======")
print(f"多于 1 张图片的文件夹数量: {len(more_than_one)}")
print(f"只有 1 张图片的文件夹数量: {len(only_one)}")
print(f"没有图片的文件夹数量: {len(zero_image)}")

print("\n====== 多于 1 张图片的文件夹 ======")
for name, count in more_than_one:
    print(f"{name}: {count}")

print("\n====== 只有 1 张图片的文件夹 ======")
for name, count in only_one:
    print(f"{name}: {count}")