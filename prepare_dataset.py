#!/usr/bin/env python3

import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = Path("/mnt/dataset/ds2_complete")

OUTPUT_ROOT = Path(
    "/home/ubuntu/yolo_omr2/exports/ds2_compact_curated_v2"
)

SEED = 42

TARGET_IMAGES = 150000
TRAIN_RATIO = 0.90

USE_SYMLINKS = True

# excluir SOMENTE stem
EXCLUDED_CLASSES = {
    "stem"
}

# mínimo de ANOTAÇÕES por classe
MIN_ANNOTATIONS_PER_CLASS = 300

# =========================================================
# HELPERS
# =========================================================

def ensure_dirs():

    for p in [
        OUTPUT_ROOT / "images" / "train",
        OUTPUT_ROOT / "images" / "val",
        OUTPUT_ROOT / "labels" / "train",
        OUTPUT_ROOT / "labels" / "val",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def category_name(cat):

    for key in (
        "name",
        "label",
        "classname",
    ):
        if key in cat:
            return cat[key]

    return None


def find_taxonomy_field(cat):

    for key in (
        "annotation_set",
        "annotation_set_name",
        "dataset",
        "source",
        "taxonomy",
        "set",
    ):
        if key in cat:
            return cat[key]

    return None


def pick_deepscores_category(cat_ids, categories):

    if isinstance(cat_ids, str):
        cat_ids = [cat_ids]

    resolved = []

    for cid in cat_ids:

        cat = categories.get(str(cid))

        if not cat:
            continue

        name = category_name(cat)
        tax = find_taxonomy_field(cat)

        resolved.append(
            (str(cid), name, tax, cat)
        )

    if not resolved:
        return None

    for cid, name, tax, cat in resolved:

        if isinstance(tax, str):

            if "deepscore" in tax.lower():
                return cid, name, cat

    cid, name, tax, cat = resolved[0]

    return cid, name, cat


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

    x1 = max(0.0, min(x1, width))
    x2 = max(0.0, min(x2, width))

    y1 = max(0.0, min(y1, height))
    y2 = max(0.0, min(y2, height))

    if x2 <= x1:
        return None

    if y2 <= y1:
        return None

    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height

    w = (x2 - x1) / width
    h = (y2 - y1) / height

    if w <= 0 or h <= 0:
        return None

    return xc, yc, w, h


def link_or_copy(src, dst):

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if USE_SYMLINKS:
        dst.symlink_to(src)

    else:
        shutil.copy2(src, dst)

# =========================================================
# INDEX IMAGES
# =========================================================

def build_image_index():

    print("[INFO] indexing images...")

    index = {}

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):

        for p in DATASET_ROOT.rglob(ext):

            if p.name not in index:
                index[p.name] = p

    print(f"[OK] indexed {len(index)} images")

    return index

# =========================================================
# LOAD RECORDS
# =========================================================

def load_records():

    shard_paths = sorted(
        DATASET_ROOT.rglob("*.json")
    )

    image_index = build_image_index()

    records = []

    annotation_freq = Counter()
    image_freq = Counter()

    class_names = set()

    for shard_path in shard_paths:

        print(f"[INFO] scanning {shard_path.name}")

        with open(shard_path, "r") as f:
            data = json.load(f)

        categories = data["categories"]
        images = data["images"]
        annotations = data["annotations"]

        for img in images:

            filename = (
                img.get("filename")
                or img.get("file_name")
            )

            if not filename:
                continue

            img_path = image_index.get(filename)

            if img_path is None:
                continue

            width = int(img["width"])
            height = int(img["height"])

            labels = []

            image_classes = set()

            for ann_id in img.get("ann_ids", []):

                ann = annotations.get(str(ann_id))

                if ann is None:
                    continue

                picked = pick_deepscores_category(
                    ann.get("cat_id", []),
                    categories
                )

                if picked is None:
                    continue

                _, cls_name, _ = picked

                if not cls_name:
                    continue

                if cls_name in EXCLUDED_CLASSES:
                    continue

                bbox = normalize_bbox_xyxy(
                    ann.get("a_bbox"),
                    width,
                    height
                )

                if bbox is None:
                    continue

                labels.append(
                    (cls_name, bbox)
                )

                image_classes.add(cls_name)

                annotation_freq[cls_name] += 1

                class_names.add(cls_name)

            if not labels:
                continue

            for cls in image_classes:
                image_freq[cls] += 1

            records.append({
                "image_path": str(img_path),
                "classes": list(image_classes),
                "labels": labels,
            })

    return (
        records,
        annotation_freq,
        image_freq,
        sorted(class_names)
    )

# =========================================================
# SCORE
# =========================================================

def compute_scores(records, annotation_freq):

    for r in records:

        score = 0.0

        class_counter = Counter()

        for cls, _ in r["labels"]:
            class_counter[cls] += 1

        for cls, count in class_counter.items():

            freq = annotation_freq[cls]

            # score baseado em ANOTAÇÕES
            score += (
                count / math.sqrt(freq)
            )

        r["score"] = score

# =========================================================
# CURATE
# =========================================================

def curate(records):

    records = sorted(
        records,
        key=lambda x: x["score"],
        reverse=True
    )

    selected = []

    selected_set = set()

    annotation_counter = Counter()

    # =====================================================
    # PASS 1 - garantir mínimo de ANOTAÇÕES
    # =====================================================

    for r in records:

        needed = False

        local_counter = Counter()

        for cls, _ in r["labels"]:
            local_counter[cls] += 1

        for cls, count in local_counter.items():

            if (
                annotation_counter[cls]
                < MIN_ANNOTATIONS_PER_CLASS
            ):
                needed = True
                break

        if needed:

            selected.append(r)

            selected_set.add(r["image_path"])

            for cls, _ in r["labels"]:
                annotation_counter[cls] += 1

    # =====================================================
    # PASS 2 - completar até alvo
    # =====================================================

    for r in records:

        if len(selected) >= TARGET_IMAGES:
            break

        if r["image_path"] in selected_set:
            continue

        selected.append(r)

        selected_set.add(r["image_path"])

    return selected

# =========================================================
# SPLIT
# =========================================================

def split_dataset(records):

    random.seed(SEED)

    random.shuffle(records)

    split_idx = int(
        len(records) * TRAIN_RATIO
    )

    return (
        records[:split_idx],
        records[split_idx:]
    )

# =========================================================
# PACK
# =========================================================

def pack_dataset(
    train_records,
    val_records,
    class_names
):

    class_to_id = {
        c: i
        for i, c in enumerate(class_names)
    }

    used_names = set()

    for split_name, split_records in [
        ("train", train_records),
        ("val", val_records),
    ]:

        print(f"[INFO] packing {split_name}")

        for i, rec in enumerate(split_records):

            img_src = Path(
                rec["image_path"]
            ).resolve()

            stem = img_src.stem
            suffix = img_src.suffix.lower()

            base_name = f"{stem}{suffix}"

            if base_name in used_names:
                base_name = (
                    f"{stem}_{i:07d}{suffix}"
                )

            used_names.add(base_name)

            img_dst = (
                OUTPUT_ROOT /
                "images" /
                split_name /
                base_name
            )

            lbl_dst = (
                OUTPUT_ROOT /
                "labels" /
                split_name /
                (
                    Path(base_name).stem
                    + ".txt"
                )
            )

            link_or_copy(
                img_src,
                img_dst
            )

            lines = []

            for cls_name, bbox in rec["labels"]:

                cls_id = class_to_id[cls_name]

                xc, yc, w, h = bbox

                lines.append(
                    f"{cls_id} "
                    f"{xc:.6f} "
                    f"{yc:.6f} "
                    f"{w:.6f} "
                    f"{h:.6f}"
                )

            with open(lbl_dst, "w") as f:
                f.write("\n".join(lines) + "\n")

    dataset_yaml = {
        "path": str(OUTPUT_ROOT),
        "train": "images/train",
        "val": "images/val",
        "names": {
            i: c
            for i, c in enumerate(class_names)
        }
    }

    with open(
        OUTPUT_ROOT / "dataset.yaml",
        "w"
    ) as f:

        yaml.safe_dump(
            dataset_yaml,
            f,
            sort_keys=False,
            allow_unicode=True
        )

# =========================================================
# REPORT
# =========================================================

def print_stats(records, class_names):

    ann_counter = Counter()
    img_counter = Counter()

    for r in records:

        seen = set()

        for cls, _ in r["labels"]:

            ann_counter[cls] += 1

            seen.add(cls)

        for cls in seen:
            img_counter[cls] += 1

    rows = []

    for cls in class_names:

        rows.append({
            "class": cls,
            "annotations": ann_counter[cls],
            "images": img_counter[cls],
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "annotations",
        ascending=False
    )

    print("\n========== CLASS STATS ==========\n")

    print(df.to_string(index=False))

    print("\n=================================\n")

# =========================================================
# MAIN
# =========================================================

def main():

    ensure_dirs()

    (
        records,
        annotation_freq,
        image_freq,
        class_names
    ) = load_records()

    print(f"\n[INFO] total records = {len(records)}")

    compute_scores(
        records,
        annotation_freq
    )

    curated = curate(records)

    train_records, val_records = split_dataset(
        curated
    )

    pack_dataset(
        train_records,
        val_records,
        class_names
    )

    print_stats(
        curated,
        class_names
    )

    print("\n========== DONE ==========\n")

    print(f"TOTAL IMAGES : {len(curated)}")
    print(f"TRAIN IMAGES : {len(train_records)}")
    print(f"VAL IMAGES   : {len(val_records)}")

    print("\n==========================\n")


if __name__ == "__main__":
    main()
