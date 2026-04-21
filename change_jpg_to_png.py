import json
from pathlib import Path


def change_extension_to_png(input_json_path: Path, output_json_path: Path):
    """
    读取 COCO 标注文件，将其中的 image file_name 后缀全部改为 .png
    """
    print(f"正在读取标注文件: {input_json_path} ...")

    # 1. 加载 JSON 数据
    with open(input_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    if 'images' not in coco_data:
        print("错误：JSON 文件中找不到 'images' 字段，请检查文件格式。")
        return

    # 2. 遍历并修改后缀
    modified_count = 0
    for img_info in coco_data['images']:
        original_name = img_info.get('file_name', '')

        if not original_name:
            continue

        # 使用 pathlib 强行将任何后缀替换为 .png
        new_name = Path(original_name).with_suffix('.png').name

        if original_name != new_name:
            img_info['file_name'] = new_name
            modified_count += 1

    # 3. 保存新的 JSON 文件
    # 如果输出路径的文件夹不存在，自动创建
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)

    print(f"\n✅ 修改完成！")
    print(f"共计修改了 {modified_count} 个图片名称的后缀。")
    print(f"新标注文件已保存至: {output_json_path}")


# ==========================================
# 运行配置
# ==========================================
if __name__ == "__main__":
    # 【输入】你需要修改的 JSON 文件路径
    # 请根据你实际的文件名进行修改，这里假设是你之前 pad 过的那个文件
    INPUT_JSON = Path('./data/filtered_annotations_padded.json')

    # 【输出】修改后缀后的新 JSON 文件路径
    OUTPUT_JSON = Path('./data/filtered_annotations_padded_png.json')

    if not INPUT_JSON.exists():
        print(f"找不到输入文件: {INPUT_JSON}")
    else:
        change_extension_to_png(INPUT_JSON, OUTPUT_JSON)