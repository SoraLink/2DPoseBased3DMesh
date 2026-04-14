import json
import os
import numpy as np
import cv2

# OpenPose BODY_25 骨架连线定义 (点索引对)
BODY_25_PAIRS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
    (11, 24), (11, 22), (22, 23), (14, 21), (14, 19), (19, 20)
]


def convert_and_visualize(anno_path, img_dir, image_name, output_dir):
    # 1. 加载标注和图片
    with open(anno_path, 'r') as f:
        data = json.load(f)

    img_path = os.path.join(img_dir, image_name)
    canvas = cv2.imread(img_path)
    if canvas is None:
        print(f"Error: 找不到图片 {img_path}")
        return

    # 2. 查找对应的第一个 annotation
    target_img_id = next((img['id'] for img in data['images'] if img['file_name'] == image_name), None)
    anno = next((a for a in data['annotations'] if a['image_id'] == target_img_id), None)

    if not anno:
        print(f"Error: 标注匹配失败")
        return

    # 3. 核心转换逻辑 (同前)
    kpts = np.array(anno['keypoints']).reshape(-1, 3)
    c17 = kpts[:17]
    op_body25 = np.zeros((25, 3))

    mapping = {0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 9: 12, 10: 14, 11: 16, 12: 11, 13: 13, 14: 15, 15: 2, 16: 1,
               17: 4, 18: 3}
    for op_idx, c_idx in mapping.items():
        if c17[c_idx][2] > 0: op_body25[op_idx] = c17[c_idx]

    # 计算 Neck 和 MidHip
    if c17[5][2] > 0 and c17[6][2] > 0:
        op_body25[1][:2] = (c17[5][:2] + c17[6][:2]) / 2
        op_body25[1][2] = (c17[5][2] + c17[6][2]) / 2
    if c17[11][2] > 0 and c17[12][2] > 0:
        op_body25[8][:2] = (c17[11][:2] + c17[12][:2]) / 2
        op_body25[8][2] = (c17[11][2] + c17[12][2]) / 2

    # --- 增加可视化绘图逻辑 ---
    # 画线 (Skeleton)
    for pair in BODY_25_PAIRS:
        p1, p2 = int(pair[0]), int(pair[1])
        if op_body25[p1][2] > 0 and op_body25[p2][2] > 0:
            pt1 = tuple(op_body25[p1][:2].astype(int))
            pt2 = tuple(op_body25[p2][:2].astype(int))
            cv2.line(canvas, pt1, pt2, (255, 255, 0), 2)  # 青色线

    # 画点 (Joints)
    for i, (x, y, conf) in enumerate(op_body25):
        if conf > 0:
            # 正常存在的点画绿色
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.putText(canvas, str(i), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            # 你提到的残疾/缺失点不画，或者你想看位置可以画红色小点
            pass

    # 4. 保存 JSON 和 预览图
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    file_basename = os.path.splitext(image_name)[0]

    # 保存 JSON
    out_json = os.path.join(output_dir, f"{file_basename}_keypoints.json")
    output_data = {"version": 1.3, "people": [{"pose_keypoints_2d": op_body25.flatten().tolist(),
                                               "face_keypoints_2d": [0] * 210, "hand_left_keypoints_2d": [0] * 63,
                                               "hand_right_keypoints_2d": [0] * 63}]}
    with open(out_json, 'w') as f:
        json.dump(output_data, f)

    # 保存预览图
    out_img = os.path.join(output_dir, f"{file_basename}_check.jpg")
    cv2.imwrite(out_img, canvas)
    print(f"验证图已保存至: {out_img}")


# 调用方法
# convert_and_visualize('LDpose.json', './images', 'test.png', './output')

# 使用示例
convert_and_visualize('./data/ldpose_train_25kpts.json', './data/residual_examples', 'baidu_残疾运动员_841.jpg', './data/keypoints')