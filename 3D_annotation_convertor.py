import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image


# =========================
# 1. Keypoint definitions: final output = 31 points
# =========================

COCO17 = [
    ("Nose", "nose"),
    ("L_Eye", "left_eye"),
    ("R_Eye", "right_eye"),
    ("L_Ear", "left_ear"),
    ("R_Ear", "right_ear"),
    ("L_Shoulder", "left_shoulder"),
    ("R_Shoulder", "right_shoulder"),
    ("L_Elbow", "left_elbow"),
    ("R_Elbow", "right_elbow"),
    ("L_Wrist", "left_wrist"),
    ("R_Wrist", "right_wrist"),
    ("L_Hip", "left_hip"),
    ("R_Hip", "right_hip"),
    ("L_Knee", "left_knee"),
    ("R_Knee", "right_knee"),
    ("L_Ankle", "left_ankle"),
    ("R_Ankle", "right_ankle"),
]

AUX_POINTS = [
    ("L_Finger", "L_Middle_Tip"),
    ("R_Finger", "R_Middle_Tip"),
    ("L_Heel", "L_Heel"),
    ("R_Heel", "R_Heel"),
    ("L_Toe", "L_Toe_Tip"),
    ("R_Toe", "R_Toe_Tip"),
]

# raw front/back residual pairs -> final Propose/METAINFO residual point name
RESIDUAL_PAIR_TO_OUTPUT = [
    (
        "Residual_L_Upperarm_Front",
        "Residual_L_Upperarm_Back",
        "L-Elbow-Res-Above",
    ),
    (
        "Residual_R_Upperarm_Front",
        "Residual_R_Upperarm_Back",
        "R-Elbow-Res-Above",
    ),
    (
        "Residual_L_Forearm_Front",
        "Residual_L_Forearm_Back",
        "L-Elbow-Res-Below",
    ),
    (
        "Residual_R_Forearm_Front",
        "Residual_R_Forearm_Back",
        "R-Elbow-Res-Below",
    ),
    (
        "Residual_L_Tigh_Front",
        "Residual_L_Tigh_Back",
        "L-Knee-Res-Above",
    ),
    (
        "Residual_R_Tigh_Front",
        "Residual_R_Tigh_Back",
        "R-Knee-Res-Above",
    ),
    (
        "Residual_L_Calf_Front",
        "Residual_L_Calf_Back",
        "L-Knee-Res-Below",
    ),
    (
        "Residual_R_Calf_Front",
        "Residual_R_Calf_Back",
        "R-Knee-Res-Below",
    ),
]


SKELETON_LINKS_BY_NAME = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),

    ("left_wrist", "L_Middle_Tip"),
    ("right_wrist", "R_Middle_Tip"),
    ("left_ankle", "L_Heel"),
    ("left_ankle", "L_Toe_Tip"),
    ("right_ankle", "R_Heel"),
    ("right_ankle", "R_Toe_Tip"),

    ("left_shoulder", "L-Elbow-Res-Above"),
    ("right_shoulder", "R-Elbow-Res-Above"),
    ("left_elbow", "L-Elbow-Res-Below"),
    ("right_elbow", "R-Elbow-Res-Below"),
    ("left_hip", "L-Knee-Res-Above"),
    ("right_hip", "R-Knee-Res-Above"),
    ("left_knee", "L-Knee-Res-Below"),
    ("right_knee", "R-Knee-Res-Below"),
]


# residual 以下的 terminal / standard distal points are prosthetic
DISTAL_KEYPOINTS = {
    "L-Elbow-Res-Above": ["left_elbow", "left_wrist", "L_Middle_Tip"],
    "R-Elbow-Res-Above": ["right_elbow", "right_wrist", "R_Middle_Tip"],

    "L-Elbow-Res-Below": ["left_wrist", "L_Middle_Tip"],
    "R-Elbow-Res-Below": ["right_wrist", "R_Middle_Tip"],

    "L-Knee-Res-Above": ["left_knee", "left_ankle", "L_Heel", "L_Toe_Tip"],
    "R-Knee-Res-Above": ["right_knee", "right_ankle", "R_Heel", "R_Toe_Tip"],

    "L-Knee-Res-Below": ["left_ankle", "L_Heel", "L_Toe_Tip"],
    "R-Knee-Res-Below": ["right_ankle", "R_Heel", "R_Toe_Tip"],
}


def build_keypoint_specs() -> List[Tuple[str, str]]:
    """
    Output order:
        17 COCO + 6 terminal points + 8 averaged residual points = 31.
    For normal points:
        tuple = (raw_name, output_name)
    For averaged residual points:
        raw_name is a special key created after averaging.
    """
    specs = []
    specs.extend(COCO17)
    specs.extend(AUX_POINTS)

    for _, _, output_name in RESIDUAL_PAIR_TO_OUTPUT:
        specs.append((output_name, output_name))

    return specs


def build_skeleton(keypoint_names: List[str]) -> List[List[int]]:
    name_to_idx_1based = {name: i + 1 for i, name in enumerate(keypoint_names)}
    skeleton = []

    for a, b in SKELETON_LINKS_BY_NAME:
        if a in name_to_idx_1based and b in name_to_idx_1based:
            skeleton.append([name_to_idx_1based[a], name_to_idx_1based[b]])

    return skeleton


# =========================
# 2. Residual averaging and type inference
# =========================

def is_valid_point(value) -> bool:
    if not isinstance(value, list) or len(value) < 2:
        return False

    x, y = value[0], value[1]
    if x is None or y is None:
        return False

    try:
        x = float(x)
        y = float(y)
    except Exception:
        return False

    return math.isfinite(x) and math.isfinite(y)


def average_points(p1, p2) -> Optional[List[float]]:
    """
    Average x, y, depth from front/back residual points.
    If one side is missing, return the valid one.
    If both missing, return None.
    """
    valid1 = is_valid_point(p1)
    valid2 = is_valid_point(p2)

    if not valid1 and not valid2:
        return None

    if valid1 and not valid2:
        x = float(p1[0])
        y = float(p1[1])
        d = float(p1[2]) if len(p1) >= 3 and p1[2] is not None else 0.0
        return [x, y, d]

    if valid2 and not valid1:
        x = float(p2[0])
        y = float(p2[1])
        d = float(p2[2]) if len(p2) >= 3 and p2[2] is not None else 0.0
        return [x, y, d]

    x = (float(p1[0]) + float(p2[0])) / 2.0
    y = (float(p1[1]) + float(p2[1])) / 2.0

    d1 = float(p1[2]) if len(p1) >= 3 and p1[2] is not None else 0.0
    d2 = float(p2[2]) if len(p2) >= 3 and p2[2] is not None else 0.0
    d = (d1 + d2) / 2.0

    return [x, y, d]


def add_averaged_residual_points(frame_kps: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """
    Add 8 Propose residual points into frame_kps by averaging front/back raw points.
    """
    out = dict(frame_kps)

    for front_name, back_name, output_name in RESIDUAL_PAIR_TO_OUTPUT:
        avg = average_points(frame_kps.get(front_name), frame_kps.get(back_name))
        if avg is not None:
            out[output_name] = avg

    return out


def infer_keypoint_types(
    frame_kps_with_avg_res: Dict[str, List[float]],
    output_names: List[str],
) -> List[str]:
    """
    Default type is normal.
    If an averaged residual point exists, downstream keypoints are marked as prosthetic.
    The residual endpoint itself remains normal.
    """
    keypoint_types = [0] * len(output_names)
    name_to_idx = {name: i for i, name in enumerate(output_names)}

    for res_name in DISTAL_KEYPOINTS.keys():
        if not is_valid_point(frame_kps_with_avg_res.get(res_name)):
            continue

        for kp_name in DISTAL_KEYPOINTS[res_name]:
            if kp_name in name_to_idx:
                keypoint_types[name_to_idx[kp_name]] = 1

    return keypoint_types


# =========================
# 3. COCO conversion utils
# =========================

def convert_frame_keypoints(
    frame_kps: Dict[str, List[float]],
    keypoint_specs: List[Tuple[str, str]],
) -> Tuple[List[float], List[float], int, List[Tuple[float, float]]]:
    keypoints = []
    depths = []
    visible_xy = []
    num_visible = 0

    for raw_name, _out_name in keypoint_specs:
        value = frame_kps.get(raw_name)

        if is_valid_point(value):
            x = float(value[0])
            y = float(value[1])
            d = float(value[2]) if len(value) >= 3 and value[2] is not None else 0.0

            keypoints.extend([x, y, 2])
            depths.append(d)
            visible_xy.append((x, y))
            num_visible += 1
        else:
            keypoints.extend([0.0, 0.0, 0])
            depths.append(0.0)

    return keypoints, depths, num_visible, visible_xy


def bbox_from_points(points: List[Tuple[float, float]], image_w: int, image_h: int):
    if not points:
        return [0.0, 0.0, 0.0, 0.0], 0.0

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(float(image_w - 1), max(xs))
    y_max = min(float(image_h - 1), max(ys))

    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    area = w * h

    return [x_min, y_min, w, h], area


def find_json_file(camera_dir: Path) -> Path:
    expected = camera_dir / f"{camera_dir.name}_2d_keypoints.json"
    if expected.exists():
        return expected

    candidates = sorted(camera_dir.glob("*_2d_keypoints.json"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No *_2d_keypoints.json found in {camera_dir}")


def frame_id_from_image_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("frame_"):
        return stem.replace("frame_", "")
    return stem


def select_frame_ids(frame_ids: List[str], n: int, strategy: str) -> List[str]:
    frame_ids = sorted(frame_ids)

    if len(frame_ids) <= n:
        return frame_ids

    if strategy == "first":
        return frame_ids[:n]

    if strategy == "last":
        return frame_ids[-n:]

    if strategy == "even":
        if n == 1:
            return [frame_ids[len(frame_ids) // 2]]
        indices = [round(i * (len(frame_ids) - 1) / (n - 1)) for i in range(n)]
        return [frame_ids[i] for i in indices]

    raise ValueError(f"Unknown sample strategy: {strategy}")


# =========================
# 4. Main converter
# =========================

def convert_dataset(
    root_dir: Path,
    output_dir: Path,
    frames_per_view: int = 2,
    sample_strategy: str = "even",
):
    keypoint_specs = build_keypoint_specs()
    output_keypoint_names = [out_name for _, out_name in keypoint_specs]
    skeleton = build_skeleton(output_keypoint_names)

    assert len(output_keypoint_names) == 31, f"Expected 31 keypoints, got {len(output_keypoint_names)}"

    output_image_dir = output_dir / "images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {
            "description": "Converted synthetic residual/prosthetic pose dataset",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": output_keypoint_names,
                "skeleton": skeleton,
                "num_keypoints": len(output_keypoint_names),
            }
        ],
        "keypoint_type_categories": {
            "normal": 0,
            "prosthetic": 1,
        },
    }

    camera_dirs = sorted([p for p in root_dir.glob("Camera_View_*") if p.is_dir()])

    image_id = 1
    ann_id = 1

    for camera_dir in camera_dirs:
        camera_name = camera_dir.name
        frames_dir = camera_dir / "frames"

        if not frames_dir.exists():
            print(f"Skip {camera_name}: no frames directory")
            continue

        json_path = find_json_file(camera_dir)

        with open(json_path, "r", encoding="utf-8") as f:
            keypoint_data = json.load(f)

        image_paths = sorted(frames_dir.glob("frame_*.png"))
        image_frame_ids = {frame_id_from_image_path(p): p for p in image_paths}

        valid_frame_ids = [
            fid for fid in keypoint_data.keys()
            if fid in image_frame_ids
        ]

        selected_ids = sorted(valid_frame_ids)
        print(f"{camera_name}: use all {len(selected_ids)} frames")
        for fid in selected_ids:
            src_img_path = image_frame_ids[fid]
            out_file_name = f"{camera_name}_frame_{fid}.png"
            dst_img_path = output_image_dir / out_file_name

            shutil.copy2(src_img_path, dst_img_path)

            with Image.open(src_img_path) as img:
                width, height = img.size

            raw_frame_kps = keypoint_data[fid]

            # Add 8 averaged residual endpoints.
            frame_kps = add_averaged_residual_points(raw_frame_kps)

            keypoints, depths, num_visible, visible_xy = convert_frame_keypoints(
                frame_kps=frame_kps,
                keypoint_specs=keypoint_specs,
            )

            keypoint_types = infer_keypoint_types(
                frame_kps_with_avg_res=frame_kps,
                output_names=output_keypoint_names,
            )

            bbox, area = bbox_from_points(visible_xy, width, height)

            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": out_file_name,
                    "width": width,
                    "height": height,
                    "camera_view": camera_name,
                    "frame_id": fid,
                    "source_file": str(src_img_path.relative_to(root_dir)),
                }
            )

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "iscrowd": 0,
                    "bbox": bbox,
                    "area": area,
                    "num_keypoints": num_visible,
                    "keypoints": keypoints,
                    "keypoint_types": keypoint_types,
                    "keypoint_depths": depths,
                    "segmentation": [],
                }
            )

            image_id += 1
            ann_id += 1

    output_ann_path = output_dir / "annotations_propose_coco.json"
    with open(output_ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print("\nDone.")
    print(f"Output images: {output_image_dir}")
    print(f"Output annotation: {output_ann_path}")
    print(f"num_keypoints = {len(output_keypoint_names)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="./3D_data/demo16",
        help="Root directory containing Camera_View_xx folders",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./3D_data",
        help="Output directory",
    )
    parser.add_argument(
        "--frames-per-view",
        type=int,
        default=2,
        help="Number of frames selected from each Camera_View folder",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default="even",
        choices=["even", "first", "last"],
        help="Frame sampling strategy",
    )

    args = parser.parse_args()

    convert_dataset(
        root_dir=Path(args.root),
        output_dir=Path(args.out),
        frames_per_view=args.frames_per_view,
        sample_strategy=args.sample,
    )