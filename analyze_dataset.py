#!/usr/bin/env python3

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = Path(
    "/home/vulpeslari/omr/dataset/aria/ds2_complete"
)

OUTPUT_DIR = Path(
    "./ds2_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# HELPERS
# =========================================================

def category_name(cat):
    for key in (
        "name",
        "label",
        "classname",
    ):
        if key in cat:
            return cat[key]

    return None


def normalize_bbox_xyxy(
    bbox,
    width,
    height
):
    if not bbox:
        return None

    if len(bbox) != 4:
        return None

    x1, y1, x2, y2 = map(float, bbox)

    if x2 <= x1:
        return None

    if y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    area = bw * bh

    norm_area = (
        area / (width * height)
    )

    aspect_ratio = bw / bh

    return {
        "w": bw,
        "h": bh,
        "area": area,
        "norm_area": norm_area,
        "aspect_ratio": aspect_ratio,
    }

# =========================================================
# ANALYSIS
# =========================================================

annotation_counter = Counter()
image_counter = Counter()

bbox_area_sum = defaultdict(float)
bbox_norm_area_sum = defaultdict(float)

bbox_area_min = {}
bbox_area_max = {}

aspect_ratio_sum = defaultdict(float)

annotations_per_image = []

image_widths = []
image_heights = []

total_images = 0
total_annotations = 0
valid_annotations = 0

declared_classes_saved = False

# =========================================================
# SCAN SHARDS
# =========================================================

json_paths = sorted(
    DATASET_ROOT.glob("*.json")
)

print(f"\n[INFO] found {len(json_paths)} json shards\n")

for shard_idx, json_path in enumerate(json_paths):

    print(
        f"[INFO] "
        f"({shard_idx+1}/{len(json_paths)}) "
        f"scanning {json_path.name}",
        flush=True
    )

    with open(json_path, "r") as f:
        data = json.load(f)

    categories = data["categories"]
    annotations = data["annotations"]
    images = data["images"]

    # =====================================================
    # CATEGORY MAP
    # =====================================================

    cat_id_to_name = {}

    for cid, cat in categories.items():
        cls_name = category_name(cat)

        if cls_name:
            cat_id_to_name[str(cid)] = cls_name

    # =====================================================
    # SAVE DECLARED CLASSES
    # =====================================================

    if not declared_classes_saved:

        all_dataset_classes = []

        for cid, cls_name in cat_id_to_name.items():
            all_dataset_classes.append({
                "cat_id": cid,
                "class": cls_name,
            })

        declared_df = pd.DataFrame(
            all_dataset_classes
        )

        declared_df = declared_df.sort_values(
            "class"
        )

        declared_csv = (
            OUTPUT_DIR /
            "all_declared_classes.csv"
        )

        declared_df.to_csv(
            declared_csv,
            index=False
        )

        print(
            f"[OK] saved declared classes:\n"
            f"{declared_csv}\n"
        )

        declared_classes_saved = True

    # =====================================================
    # IMAGES
    # =====================================================

    for img in images:

        total_images += 1

        width = int(img["width"])
        height = int(img["height"])

        image_widths.append(width)
        image_heights.append(height)

        local_classes = set()

        ann_count_this_image = 0

        for ann_id in img.get("ann_ids", []):

            ann = annotations.get(str(ann_id))

            if ann is None:
                continue

            total_annotations += 1

            cat_ids = ann.get("cat_id", [])

            if not cat_ids:
                continue

            bbox_stats = normalize_bbox_xyxy(
                ann.get("a_bbox"),
                width,
                height
            )

            if bbox_stats is None:
                continue

            valid_annotations += 1

            area = bbox_stats["area"]
            norm_area = bbox_stats["norm_area"]
            aspect_ratio = bbox_stats["aspect_ratio"]

            ann_count_this_image += 1

            # =================================================
            # COUNT ALL CATEGORY IDS
            # =================================================

            for cid in cat_ids:

                cid = str(cid)

                cls_name = cat_id_to_name.get(cid)

                if cls_name is None:
                    continue

                annotation_counter[cls_name] += 1

                local_classes.add(cls_name)

                # ---------------------------------------------
                # bbox stats
                # ---------------------------------------------

                bbox_area_sum[cls_name] += area

                bbox_norm_area_sum[cls_name] += norm_area

                aspect_ratio_sum[cls_name] += aspect_ratio

                if (
                    cls_name not in bbox_area_min
                    or area < bbox_area_min[cls_name]
                ):
                    bbox_area_min[cls_name] = area

                if (
                    cls_name not in bbox_area_max
                    or area > bbox_area_max[cls_name]
                ):
                    bbox_area_max[cls_name] = area

        annotations_per_image.append(
            ann_count_this_image
        )

        for cls_name in local_classes:
            image_counter[cls_name] += 1

# =========================================================
# BUILD CLASS TABLE
# =========================================================

rows = []

all_classes = sorted(
    annotation_counter.keys()
)

for cls_name in all_classes:

    ann_count = annotation_counter[cls_name]

    rows.append({
        "class": cls_name,

        "annotations":
            ann_count,

        "images":
            image_counter[cls_name],

        "avg_bbox_area_px":
            bbox_area_sum[cls_name]
            / ann_count,

        "avg_bbox_area_norm":
            bbox_norm_area_sum[cls_name]
            / ann_count,

        "min_bbox_area_px":
            bbox_area_min[cls_name],

        "max_bbox_area_px":
            bbox_area_max[cls_name],

        "avg_aspect_ratio":
            aspect_ratio_sum[cls_name]
            / ann_count,
    })

df = pd.DataFrame(rows)

df = df.sort_values(
    "annotations",
    ascending=False
)

# =========================================================
# SAVE CLASS DISTRIBUTION
# =========================================================

csv_path = (
    OUTPUT_DIR /
    "class_distribution.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print(
    f"\n[OK] saved:\n{csv_path}\n"
)

# =========================================================
# GLOBAL STATS
# =========================================================

global_stats = pd.DataFrame([{

    "total_images":
        total_images,

    "total_annotations":
        total_annotations,

    "valid_annotations":
        valid_annotations,

    "num_classes":
        len(all_classes),

    "avg_annotations_per_image":
        sum(annotations_per_image)
        / len(annotations_per_image),

    "median_annotations_per_image":
        pd.Series(
            annotations_per_image
        ).median(),

    "p95_annotations_per_image":
        pd.Series(
            annotations_per_image
        ).quantile(0.95),

    "avg_image_width":
        sum(image_widths)
        / len(image_widths),

    "avg_image_height":
        sum(image_heights)
        / len(image_heights),
}])

global_csv = (
    OUTPUT_DIR /
    "global_stats.csv"
)

global_stats.to_csv(
    global_csv,
    index=False
)

print(
    f"[OK] saved:\n{global_csv}\n"
)

# =========================================================
# RARE CLASSES
# =========================================================

rare_df = df[
    df["annotations"] < 1000
]

rare_csv = (
    OUTPUT_DIR /
    "rare_classes.csv"
)

rare_df.to_csv(
    rare_csv,
    index=False
)

print(
    f"[OK] saved:\n{rare_csv}\n"
)

# =========================================================
# MISSING DECLARED CLASSES
# =========================================================

declared_classes = set(
    declared_df["class"].tolist()
)

observed_classes = set(
    df["class"].tolist()
)

missing_classes = sorted(
    declared_classes - observed_classes
)

missing_df = pd.DataFrame({
    "missing_class": missing_classes
})

missing_csv = (
    OUTPUT_DIR /
    "missing_declared_classes.csv"
)

missing_df.to_csv(
    missing_csv,
    index=False
)

print(
    f"[OK] saved:\n{missing_csv}\n"
)

# =========================================================
# SUMMARY
# =========================================================

print("\n========== SUMMARY ==========\n")

print(f"TOTAL IMAGES       : {total_images}")
print(f"TOTAL ANNOTATIONS  : {total_annotations}")
print(f"VALID ANNOTATIONS  : {valid_annotations}")
print(f"TOTAL CLASSES      : {len(all_classes)}")

print("\nTOP 20 CLASSES:\n")

print(
    df.head(20).to_string(index=False)
)

print("\n=============================\n")