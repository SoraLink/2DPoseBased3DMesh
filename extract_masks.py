import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from tqdm import tqdm

def extract_densepose_to_instance_png(json_path, output_dir):
    print(f"📦 Loading DensePose JSON: {json_path}")
    coco = COCO(json_path)
    os.makedirs(output_dir, exist_ok=True)

    img_ids = coco.getImgIds()
    print(f"🚀 Start extracting DensePose masks to instance PNGs, total {len(img_ids)} images...")

    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        original_file_name = img_info['file_name']
        H, W = img_info['height'], img_info['width']

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if 'dp_masks' not in ann or 'dp_I' not in ann:
                raise ValueError(f"Missing dp_masks or dp_I in ann: {ann}")

            instance_canvas = np.zeros((H, W), dtype=np.uint8)

            x_min, y_min, w, h = ann['bbox']
            x, y, w, h = int(x_min), int(y_min), int(w), int(h)

            part_ids = ann['dp_I']
            rle_masks = ann['dp_masks']

            for part_id, rle in zip(part_ids, rle_masks):
                patch_mask = maskUtils.decode(rle)

                if patch_mask.size == 0:
                    continue

                if w <= 0 or h <= 0:
                    raise ValueError(f"Invalid mask size: {patch_mask.size}, {w}, {h}")

                patch_mask_resized = cv2.resize(
                    patch_mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(W, x + w), min(H, y + h)

                if x1 >= x2 or y1 >= y2:
                    raise ValueError(f"Invalid bbox: {x1}, {y1}, {x2}, {y2}")

                px1, py1 = x1 - x, y1 - y
                px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)

                valid_patch = patch_mask_resized[py1:py2, px1:px2]
                instance_canvas[y1:y2, x1:x2][valid_patch > 0] = 255

            ann_id = ann['id']
            save_path = os.path.join(output_dir, f"{img_id:012d}_{ann_id}.png")
            cv2.imwrite(save_path, instance_canvas)

    print(f"✅ Successfully extracted DensePose masks to instance PNGs. Saved in: {output_dir}")

if __name__ == "__main__":
    json_file = "./data/densepose_coco_2014_train.json"
    out_folder = "./data/densepose_masks_instance/"
    extract_densepose_to_instance_png(json_file, out_folder)