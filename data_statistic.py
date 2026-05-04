import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path


# residual endpoint index mapping
# side 是人体解剖学左右，不是图片左右
RESIDUAL_KPT_MAPPING = {
    23: {
        "limb": "arm",
        "side": "left",
        "level": "upper",
        "segment": "upper_arm",
        "name": "left_upper_arm",
        "downstream_terminal": 17,  # left hand-related endpoint in your 31-kpt schema
    },
    24: {
        "limb": "arm",
        "side": "right",
        "level": "upper",
        "segment": "upper_arm",
        "name": "right_upper_arm",
        "downstream_terminal": 18,
    },
    25: {
        "limb": "arm",
        "side": "left",
        "level": "lower",
        "segment": "forearm",
        "name": "left_forearm",
        "downstream_terminal": 17,
    },
    26: {
        "limb": "arm",
        "side": "right",
        "level": "lower",
        "segment": "forearm",
        "name": "right_forearm",
        "downstream_terminal": 18,
    },
    27: {
        "limb": "leg",
        "side": "left",
        "level": "upper",
        "segment": "thigh",
        "name": "left_thigh",
        "downstream_terminal": 21,  # left foot-related endpoint
    },
    28: {
        "limb": "leg",
        "side": "right",
        "level": "upper",
        "segment": "thigh",
        "name": "right_thigh",
        "downstream_terminal": 22,
    },
    29: {
        "limb": "leg",
        "side": "left",
        "level": "lower",
        "segment": "lower_leg",
        "name": "left_lower_leg",
        "downstream_terminal": 21,
    },
    30: {
        "limb": "leg",
        "side": "right",
        "level": "lower",
        "segment": "lower_leg",
        "name": "right_lower_leg",
        "downstream_terminal": 22,
    },
}


def load_coco_annotation(annotation_path: str):
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "annotations" not in data:
        raise ValueError("This does not look like a COCO-style file: missing `annotations`.")

    image_id_to_name = {}
    for img in data.get("images", []):
        image_id_to_name[img.get("id")] = Path(img.get("file_name", "")).name

    return data, image_id_to_name


def get_existing_residuals(keypoint_types):
    """
    Return a list of residual endpoint records whose type == 0.
    """
    residuals = []

    for res_idx, meta in RESIDUAL_KPT_MAPPING.items():
        if res_idx >= len(keypoint_types):
            continue

        # Your definition:
        # 0 = residual endpoint exists
        # 2 = absent
        if keypoint_types[res_idx] == 0:
            residuals.append({
                "res_idx": res_idx,
                **meta
            })

    return residuals


def is_prosthesis_related(keypoint_types, residual_record):
    """
    Optional statistic:
    If downstream terminal point type == 1, we treat this residual target as prosthesis-related.
    This follows your current prompt-generation logic.
    """
    downstream_idx = residual_record["downstream_terminal"]

    if downstream_idx >= len(keypoint_types):
        return False

    return keypoint_types[downstream_idx] == 1


def summarize_annotation(annotation_path: str):
    data, image_id_to_name = load_coco_annotation(annotation_path)

    stats = {
        "num_images_in_file": len(data.get("images", [])),
        "num_annotations": len(data.get("annotations", [])),
        "num_images_with_residual": 0,
        "num_residual_endpoints": 0,
    }

    limb_counter = Counter()
    level_counter = Counter()
    side_counter = Counter()
    segment_counter = Counter()
    res_idx_counter = Counter()
    prosthesis_counter = Counter()
    per_image_residual_count = Counter()

    image_level_records = []

    for ann in data["annotations"]:
        image_id = ann.get("image_id")
        file_name = image_id_to_name.get(image_id, str(image_id))

        keypoint_types = ann.get("keypoint_types", None)
        if keypoint_types is None:
            raise ValueError(f"Annotation for image {file_name} does not contain `keypoint_types`.")

        residuals = get_existing_residuals(keypoint_types)
        n_res = len(residuals)

        per_image_residual_count[n_res] += 1

        if n_res > 0:
            stats["num_images_with_residual"] += 1
            stats["num_residual_endpoints"] += n_res

        image_record = {
            "image_id": image_id,
            "file_name": file_name,
            "num_residuals": n_res,
            "residuals": [],
        }

        for r in residuals:
            limb_counter[r["limb"]] += 1
            level_counter[r["level"]] += 1
            side_counter[r["side"]] += 1
            segment_counter[r["segment"]] += 1
            res_idx_counter[r["res_idx"]] += 1

            prosthesis_related = is_prosthesis_related(keypoint_types, r)
            prosthesis_counter["prosthesis_related" if prosthesis_related else "non_prosthesis_or_unknown"] += 1

            image_record["residuals"].append({
                "res_idx": r["res_idx"],
                "name": r["name"],
                "limb": r["limb"],
                "side": r["side"],
                "level": r["level"],
                "segment": r["segment"],
                "prosthesis_related": prosthesis_related,
            })

        image_level_records.append(image_record)

    return {
        "stats": stats,
        "limb_counter": limb_counter,
        "level_counter": level_counter,
        "side_counter": side_counter,
        "segment_counter": segment_counter,
        "res_idx_counter": res_idx_counter,
        "prosthesis_counter": prosthesis_counter,
        "per_image_residual_count": per_image_residual_count,
        "image_level_records": image_level_records,
    }


def print_summary(result):
    stats = result["stats"]

    print("\n================ Dataset Residual-Limb Statistics ================")
    print(f"Images in file:              {stats['num_images_in_file']}")
    print(f"Annotations:                 {stats['num_annotations']}")
    print(f"Images with residual limbs:  {stats['num_images_with_residual']}")
    print(f"Residual endpoints total:    {stats['num_residual_endpoints']}")

    print("\n---------------- Arm vs Leg ----------------")
    for k in ["arm", "leg"]:
        print(f"{k:>8}: {result['limb_counter'][k]}")

    print("\n---------------- Upper vs Lower Level ----------------")
    print("upper = upper arm / thigh residual")
    print("lower = forearm / lower-leg residual")
    for k in ["upper", "lower"]:
        print(f"{k:>8}: {result['level_counter'][k]}")

    print("\n---------------- Left vs Right ----------------")
    for k in ["left", "right"]:
        print(f"{k:>8}: {result['side_counter'][k]}")

    print("\n---------------- Segment Breakdown ----------------")
    for k in ["upper_arm", "forearm", "thigh", "lower_leg"]:
        print(f"{k:>12}: {result['segment_counter'][k]}")

    print("\n---------------- Residual Keypoint Index Breakdown ----------------")
    for idx in sorted(RESIDUAL_KPT_MAPPING.keys()):
        meta = RESIDUAL_KPT_MAPPING[idx]
        print(f"kpt {idx:02d} ({meta['name']:>16}): {result['res_idx_counter'][idx]}")

    print("\n---------------- Per-Image Residual Count ----------------")
    for n_res, count in sorted(result["per_image_residual_count"].items()):
        print(f"{n_res} residual endpoint(s): {count} image(s)")

    print("\n---------------- Prosthesis-Related Optional Count ----------------")
    for k, v in result["prosthesis_counter"].items():
        print(f"{k:>28}: {v}")


def save_image_level_csv(result, output_csv):
    import csv

    rows = []

    for record in result["image_level_records"]:
        if record["num_residuals"] == 0:
            rows.append({
                "image_id": record["image_id"],
                "file_name": record["file_name"],
                "num_residuals": 0,
                "res_idx": "",
                "name": "",
                "limb": "",
                "side": "",
                "level": "",
                "segment": "",
                "prosthesis_related": "",
            })
        else:
            for r in record["residuals"]:
                rows.append({
                    "image_id": record["image_id"],
                    "file_name": record["file_name"],
                    "num_residuals": record["num_residuals"],
                    "res_idx": r["res_idx"],
                    "name": r["name"],
                    "limb": r["limb"],
                    "side": r["side"],
                    "level": r["level"],
                    "segment": r["segment"],
                    "prosthesis_related": r["prosthesis_related"],
                })

    fieldnames = [
        "image_id",
        "file_name",
        "num_residuals",
        "res_idx",
        "name",
        "limb",
        "side",
        "level",
        "segment",
        "prosthesis_related",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved image-level CSV to: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann",
        type=str,
        default='./data/filtered_annotations_padded_png.json',
        help="Path to COCO-style annotation JSON."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to save image-level residual statistics CSV."
    )
    args = parser.parse_args()

    result = summarize_annotation(args.ann)
    print_summary(result)

    if args.csv is not None:
        save_image_level_csv(result, args.csv)


if __name__ == "__main__":
    main()