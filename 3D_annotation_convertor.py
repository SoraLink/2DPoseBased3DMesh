import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image


# ============================================================
# 1. Keypoint definitions: final output = 31 points
# ============================================================

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

RESIDUAL_PAIR_TO_OUTPUT = [
    ("Residual_L_Upperarm_Front", "Residual_L_Upperarm_Back", "L-Elbow-Res-Above"),
    ("Residual_R_Upperarm_Front", "Residual_R_Upperarm_Back", "R-Elbow-Res-Above"),
    ("Residual_L_Forearm_Front", "Residual_L_Forearm_Back", "L-Elbow-Res-Below"),
    ("Residual_R_Forearm_Front", "Residual_R_Forearm_Back", "R-Elbow-Res-Below"),
    ("Residual_L_Tigh_Front", "Residual_L_Tigh_Back", "L-Knee-Res-Above"),
    ("Residual_R_Tigh_Front", "Residual_R_Tigh_Back", "R-Knee-Res-Above"),
    ("Residual_L_Calf_Front", "Residual_L_Calf_Back", "L-Knee-Res-Below"),
    ("Residual_R_Calf_Front", "Residual_R_Calf_Back", "R-Knee-Res-Below"),
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

# If a residual endpoint exists, downstream complete-limb keypoints are prosthetic.
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


# ============================================================
# 2. Basic keypoint utilities
# ============================================================

def is_valid_point(value) -> bool:
    """
    For 2D/depth keypoints: [x, y, depth]
    For 3D keypoints: [x, y, z]
    This checks at least x/y are valid.
    """
    if not isinstance(value, list) or len(value) < 2:
        return False

    try:
        x = float(value[0])
        y = float(value[1])
    except Exception:
        return False

    return math.isfinite(x) and math.isfinite(y)


def is_valid_3d_point(value) -> bool:
    if not isinstance(value, list) or len(value) < 3:
        return False

    try:
        x = float(value[0])
        y = float(value[1])
        z = float(value[2])
    except Exception:
        return False

    return math.isfinite(x) and math.isfinite(y) and math.isfinite(z)


def get_third(value) -> float:
    if isinstance(value, list) and len(value) >= 3 and value[2] is not None:
        try:
            v = float(value[2])
            if math.isfinite(v):
                return v
        except Exception:
            pass
    return 0.0


def average_points(p1, p2) -> Optional[List[float]]:
    """
    Average x, y, third value.
    For 2D camera annotations, third value is depth.
    For global 3D annotations, third value is z.
    """
    valid1 = is_valid_point(p1)
    valid2 = is_valid_point(p2)

    if not valid1 and not valid2:
        return None

    if valid1 and not valid2:
        return [float(p1[0]), float(p1[1]), get_third(p1)]

    if valid2 and not valid1:
        return [float(p2[0]), float(p2[1]), get_third(p2)]

    x = (float(p1[0]) + float(p2[0])) / 2.0
    y = (float(p1[1]) + float(p2[1])) / 2.0
    z_or_depth = (get_third(p1) + get_third(p2)) / 2.0

    return [x, y, z_or_depth]


def add_averaged_residual_points(frame_kps: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """
    Add 8 final residual endpoints by averaging front/back residual annotations.
    """
    out = dict(frame_kps)

    for front_name, back_name, output_name in RESIDUAL_PAIR_TO_OUTPUT:
        avg = average_points(frame_kps.get(front_name), frame_kps.get(back_name))
        if avg is not None:
            out[output_name] = avg

    return out


def infer_keypoint_types(
    frame_kps_with_avg_res: Dict[str, List[float]],
    keypoint_specs: List[Tuple[str, str]],
) -> List[int]:
    """
    keypoint_types:
        0 = normal
        1 = prosthetic
        2 = absent

    Logic:
    - If a keypoint does not exist, type = 2.
    - Existing keypoints are normal by default, type = 0.
    - If a residual endpoint exists, downstream existing points are prosthetic, type = 1.
    - The residual endpoint itself remains normal, type = 0.
    """
    keypoint_types = []
    name_to_idx = {}

    for idx, (raw_name, out_name) in enumerate(keypoint_specs):
        name_to_idx[out_name] = idx

        if is_valid_point(frame_kps_with_avg_res.get(raw_name)):
            keypoint_types.append(0)
        else:
            keypoint_types.append(2)

    for res_name, downstream_names in DISTAL_KEYPOINTS.items():
        if not is_valid_point(frame_kps_with_avg_res.get(res_name)):
            continue

        for kp_name in downstream_names:
            if kp_name not in name_to_idx:
                continue

            idx = name_to_idx[kp_name]
            if keypoint_types[idx] != 2:
                keypoint_types[idx] = 1

    return keypoint_types


# ============================================================
# 3. 2D and 3D conversion utilities
# ============================================================

def convert_frame_keypoints_2d(
    frame_kps: Dict[str, List[float]],
    keypoint_specs: List[Tuple[str, str]],
) -> Tuple[List[float], List[float], int, List[Tuple[float, float]]]:
    """
    Return:
        keypoints: COCO style [x, y, v, ...]
        depths: [depth1, depth2, ...]
        num_visible
        visible_xy
    """
    keypoints = []
    depths = []
    visible_xy = []
    num_visible = 0

    for raw_name, _out_name in keypoint_specs:
        value = frame_kps.get(raw_name)

        if is_valid_point(value):
            x = float(value[0])
            y = float(value[1])
            d = get_third(value)

            keypoints.extend([x, y, 2])
            depths.append(d)
            visible_xy.append((x, y))
            num_visible += 1
        else:
            keypoints.extend([0.0, 0.0, 0])
            depths.append(0.0)

    return keypoints, depths, num_visible, visible_xy


def convert_frame_keypoints_3d(
    frame_kps: Dict[str, List[float]],
    keypoint_specs: List[Tuple[str, str]],
) -> Tuple[List[float], int]:
    """
    Return:
        keypoints_3d: [x, y, z, v, ...]
        num_valid
    """
    keypoints_3d = []
    num_valid = 0

    for raw_name, _out_name in keypoint_specs:
        value = frame_kps.get(raw_name)

        if is_valid_3d_point(value):
            x = float(value[0])
            y = float(value[1])
            z = float(value[2])

            keypoints_3d.extend([x, y, z, 1])
            num_valid += 1
        else:
            keypoints_3d.extend([0.0, 0.0, 0.0, 0])

    return keypoints_3d, num_valid


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


# ============================================================
# 4. File discovery utilities
# ============================================================

def make_safe_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def frame_id_from_image_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("frame_"):
        return stem.replace("frame_", "")
    return stem


def collect_image_paths(frames_dir: Path) -> Dict[str, Path]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]
    image_paths = []

    for ext in exts:
        image_paths.extend(frames_dir.glob(ext))

    image_paths = sorted(image_paths)
    return {frame_id_from_image_path(p): p for p in image_paths}


def find_camera_2d_json(camera_dir: Path) -> Path:
    expected = camera_dir / f"{camera_dir.name}_2d_keypoints.json"
    if expected.exists():
        return expected

    candidates = sorted(camera_dir.glob("*_2d_keypoints.json"))
    if candidates:
        return candidates[0]

    candidates = sorted(camera_dir.glob("*.json"))
    if candidates:
        # fallback, but prefer *_2d_keypoints.json above
        return candidates[0]

    raise FileNotFoundError(f"No 2D keypoint json found in {camera_dir}")


def find_global_3d_json(demo_dir: Path) -> Optional[Path]:
    expected = demo_dir / "global_3d_keypoints.json"
    if expected.exists():
        return expected

    candidates = sorted(demo_dir.glob("*global*3d*keypoints*.json"))
    if candidates:
        return candidates[0]

    candidates = sorted(demo_dir.glob("*3d*keypoints*.json"))
    if candidates:
        return candidates[0]

    return None


def get_demo_dirs(root_dir: Path) -> List[Path]:
    """
    Supports:
    1. root_dir is a single demo directory:
       root_dir/global_3d_keypoints.json
       root_dir/Camera_View_00/...
    2. root_dir contains multiple demo directories:
       root_dir/demo16/global_3d_keypoints.json
       root_dir/demo16/Camera_View_00/...
       root_dir/demo17/...
    """
    if list(root_dir.glob("Camera_View_*")):
        return [root_dir]

    demo_dirs = []
    for p in sorted(root_dir.iterdir()):
        if not p.is_dir():
            continue

        has_camera = bool(list(p.glob("Camera_View_*")))
        has_3d = find_global_3d_json(p) is not None

        if has_camera or has_3d:
            demo_dirs.append(p)

    return demo_dirs


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 5. Main converter
# ============================================================

def build_empty_2d_coco(output_keypoint_names, skeleton):
    return {
        "info": {
            "description": "Converted multi-demo synthetic residual/prosthetic 2D pose dataset",
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
            "absent": 2,
        },
    }


def build_empty_3d_json(output_keypoint_names, skeleton):
    return {
        "info": {
            "description": "Converted multi-demo synthetic residual/prosthetic 3D keypoint dataset",
            "version": "1.0",
            "keypoints_3d_format": "[x, y, z, v] repeated for each keypoint",
        },
        "demos": [],
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
            "absent": 2,
        },
    }


def add_3d_annotation_if_needed(
    *,
    gt3d_by_demo_frame: Dict[Tuple[str, str], int],
    ann3d: Dict,
    next_3d_ann_id: int,
    demo_id: int,
    demo_name: str,
    frame_id: str,
    global_3d_data: Optional[Dict],
    keypoint_specs: List[Tuple[str, str]],
    source_file: Optional[str],
) -> Tuple[Optional[int], int]:
    """
    Create one 3D annotation per (demo_name, frame_id), shared by all camera views.
    Return:
        gt3d_id, next_3d_ann_id
    """
    key = (demo_name, frame_id)

    if key in gt3d_by_demo_frame:
        return gt3d_by_demo_frame[key], next_3d_ann_id

    if global_3d_data is None:
        return None, next_3d_ann_id

    raw_3d_frame = global_3d_data.get(frame_id)
    if raw_3d_frame is None:
        return None, next_3d_ann_id

    frame_3d = add_averaged_residual_points(raw_3d_frame)

    keypoints_3d, num_valid_3d = convert_frame_keypoints_3d(
        frame_kps=frame_3d,
        keypoint_specs=keypoint_specs,
    )

    keypoint_types_3d = infer_keypoint_types(
        frame_kps_with_avg_res=frame_3d,
        keypoint_specs=keypoint_specs,
    )

    gt3d_id = next_3d_ann_id
    next_3d_ann_id += 1

    ann3d["annotations"].append(
        {
            "id": gt3d_id,
            "demo_id": demo_id,
            "demo_name": demo_name,
            "frame_id": frame_id,
            "category_id": 1,
            "num_keypoints": num_valid_3d,
            "keypoints_3d": keypoints_3d,
            "keypoint_types": keypoint_types_3d,
            "source_file": source_file,
        }
    )

    gt3d_by_demo_frame[key] = gt3d_id
    return gt3d_id, next_3d_ann_id


def convert_dataset(
    root_dir: Path,
    output_dir: Path,
):
    keypoint_specs = build_keypoint_specs()
    output_keypoint_names = [out_name for _, out_name in keypoint_specs]
    skeleton = build_skeleton(output_keypoint_names)

    assert len(output_keypoint_names) == 31, (
        f"Expected 31 keypoints, got {len(output_keypoint_names)}"
    )

    output_image_dir = output_dir / "images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    ann2d = build_empty_2d_coco(output_keypoint_names, skeleton)
    ann3d = build_empty_3d_json(output_keypoint_names, skeleton)

    demo_dirs = get_demo_dirs(root_dir)
    if not demo_dirs:
        print(f"❌ No demo folders found under: {root_dir}")
        return

    print(f"Found {len(demo_dirs)} demo folder(s):")
    for d in demo_dirs:
        print(f"  - {d}")

    image_id = 1
    ann_id = 1
    demo_id = 1
    next_3d_ann_id = 1

    gt3d_by_demo_frame: Dict[Tuple[str, str], int] = {}

    for demo_dir in demo_dirs:
        demo_name = demo_dir.name
        safe_demo_name = make_safe_name(demo_name)

        global_3d_json_path = find_global_3d_json(demo_dir)
        global_3d_data = None

        if global_3d_json_path is not None:
            global_3d_data = load_json(global_3d_json_path)
            global_3d_source_rel = str(global_3d_json_path.relative_to(root_dir))
        else:
            global_3d_source_rel = None
            print(f"⚠️ {demo_name}: no global 3D keypoint file found")

        ann3d["demos"].append(
            {
                "id": demo_id,
                "name": demo_name,
                "source_dir": str(demo_dir.relative_to(root_dir)),
                "global_3d_keypoints_file": global_3d_source_rel,
            }
        )

        camera_dirs = sorted([p for p in demo_dir.glob("Camera_View_*") if p.is_dir()])

        if not camera_dirs:
            print(f"⚠️ {demo_name}: no Camera_View_* folders found")
            demo_id += 1
            continue

        print(f"\nDemo {demo_name}: found {len(camera_dirs)} camera view folder(s)")

        for camera_dir in camera_dirs:
            camera_name = camera_dir.name
            safe_camera_name = make_safe_name(camera_name)
            frames_dir = camera_dir / "frames"

            if not frames_dir.exists():
                print(f"  ⚠️ Skip {demo_name}/{camera_name}: no frames directory")
                continue

            try:
                json_path_2d = find_camera_2d_json(camera_dir)
            except FileNotFoundError as e:
                print(f"  ⚠️ {e}")
                continue

            keypoint_data_2d = load_json(json_path_2d)
            image_frame_ids = collect_image_paths(frames_dir)

            valid_frame_ids = [
                fid for fid in keypoint_data_2d.keys()
                if fid in image_frame_ids
            ]

            if not valid_frame_ids:
                print(f"  ⚠️ {demo_name}/{camera_name}: no matched frames")
                print(f"     json keys example: {list(keypoint_data_2d.keys())[:5]}")
                print(f"     image ids example: {list(image_frame_ids.keys())[:5]}")
                continue

            selected_ids = sorted(valid_frame_ids)
            print(f"  {camera_name}: use all {len(selected_ids)} matched frame(s)")

            for fid in selected_ids:
                src_img_path = image_frame_ids[fid]

                out_file_name = (
                    f"{safe_demo_name}__{safe_camera_name}__frame_{fid}"
                    f"{src_img_path.suffix.lower()}"
                )
                dst_img_path = output_image_dir / out_file_name
                shutil.copy2(src_img_path, dst_img_path)

                with Image.open(src_img_path) as img:
                    width, height = img.size

                raw_frame_kps_2d = keypoint_data_2d[fid]

                frame_kps_2d = add_averaged_residual_points(raw_frame_kps_2d)

                keypoints_2d, depths, num_visible, visible_xy = convert_frame_keypoints_2d(
                    frame_kps=frame_kps_2d,
                    keypoint_specs=keypoint_specs,
                )

                keypoint_types_2d = infer_keypoint_types(
                    frame_kps_with_avg_res=frame_kps_2d,
                    keypoint_specs=keypoint_specs,
                )

                bbox, area = bbox_from_points(visible_xy, width, height)

                gt3d_id, next_3d_ann_id = add_3d_annotation_if_needed(
                    gt3d_by_demo_frame=gt3d_by_demo_frame,
                    ann3d=ann3d,
                    next_3d_ann_id=next_3d_ann_id,
                    demo_id=demo_id,
                    demo_name=demo_name,
                    frame_id=fid,
                    global_3d_data=global_3d_data,
                    keypoint_specs=keypoint_specs,
                    source_file=global_3d_source_rel,
                )

                ann2d["images"].append(
                    {
                        "id": image_id,
                        "file_name": out_file_name,
                        "width": width,
                        "height": height,
                        "demo_id": demo_id,
                        "demo_name": demo_name,
                        "camera_view": camera_name,
                        "frame_id": fid,
                        "gt3d_id": gt3d_id,
                        "source_file": str(src_img_path.relative_to(root_dir)),
                        "source_2d_keypoints_file": str(json_path_2d.relative_to(root_dir)),
                    }
                )

                ann2d["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "iscrowd": 0,
                        "bbox": bbox,
                        "area": area,
                        "num_keypoints": num_visible,
                        "keypoints": keypoints_2d,
                        "keypoint_types": keypoint_types_2d,
                        "keypoint_depths": depths,
                        "gt3d_id": gt3d_id,
                        "demo_id": demo_id,
                        "demo_name": demo_name,
                        "camera_view": camera_name,
                        "frame_id": fid,
                        "segmentation": [],
                    }
                )

                image_id += 1
                ann_id += 1

        demo_id += 1

    output_ann_2d_path = output_dir / "annotations_2d_propose_coco.json"
    output_ann_3d_path = output_dir / "annotations_3d_propose.json"

    with open(output_ann_2d_path, "w", encoding="utf-8") as f:
        json.dump(ann2d, f, indent=2)

    with open(output_ann_3d_path, "w", encoding="utf-8") as f:
        json.dump(ann3d, f, indent=2)

    print("\n✅ Done.")
    print(f"Output images: {output_image_dir}")
    print(f"Output 2D annotation: {output_ann_2d_path}")
    print(f"Output 3D annotation: {output_ann_3d_path}")
    print(f"num images = {len(ann2d['images'])}")
    print(f"num 2D annotations = {len(ann2d['annotations'])}")
    print(f"num 3D annotations = {len(ann3d['annotations'])}")
    print(f"num demos = {len(ann3d['demos'])}")
    print(f"num keypoints = {len(output_keypoint_names)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="./demo",
        help=(
            "Root directory containing multiple demo folders, "
            "or one single demo folder with Camera_View_xx folders."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./3D_data_converted",
        help="Output directory",
    )

    args = parser.parse_args()

    convert_dataset(
        root_dir=Path(args.root),
        output_dir=Path(args.out),
    )