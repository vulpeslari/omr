#!/usr/bin/env python3

import json
import random
import shutil
import subprocess
from collections import Counter
from collections import defaultdict
from pathlib import Path
import pandas as pd
import yaml

# =========================================================
# AWS CONFIG
# =========================================================

DATASET_ROOT = Path("/mnt/dataset/ds2_complete")
IMAGES_DIR = DATASET_ROOT / "images"
WORK_ROOT = Path("/home/ubuntu/yolo_omr2")

EXPORT_NAME = "ds2_yolo_static_all_compact"
EXPORT_ROOT = WORK_ROOT / "exports" / EXPORT_NAME
PACKED_ROOT = WORK_ROOT / "exports" / f"{EXPORT_NAME}_packed"

# =========================================================
# TRAIN CONFIG
# =========================================================

SEED = 42
TRAIN_RATIO = 0.80
MODEL = "yolov8n.pt"
IMGSZ = 640
BATCH = 2
EPOCHS = 12
DEVICE = "0"
WORKERS = 2
CACHE = False
RUNS_DIR = WORK_ROOT / "runs"
RUN_NAME = "ds2_static_all_compact_n640_b2_e12"
USE_SYMLINKS = True

# =========================================================
# ONLY EXCLUDED CLASS
# =========================================================

EXCLUDED_CLASSES = {
    "stem"
}

# =========================================================
# HELPERS
# =========================================================

def ensure_dirs():
    for p in [
        WORK_ROOT,
        WORK_ROOT / "exports",

        EXPORT_ROOT / "meta",

        PACKED_ROOT / "images" / "train",
        PACKED_ROOT / "images" / "val",

        PACKED_ROOT / "labels" / "train",
        PACKED_ROOT / "labels" / "val",

        RUNS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def build_image_index():
    print("[INFO] indexing image files...")

    index = {}

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for p in DATASET_ROOT.rglob(ext):
            if p.name not in index:
                index[p.name] = p

    print(f"[OK] indexed {len(index)} images")

    return index


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
        "annotationSet",
        "annotation_set_id",
    ):
        if key in cat:
            return cat[key]

    for key in ("annotation_set", "meta", "info"):
        if key in cat and isinstance(cat[key], dict):
            for subkey in (
                "name",
                "dataset",
                "source",
                "taxonomy",
                "id",
            ):
                if subkey in cat[key]:
                    return cat[key][subkey]

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
            if tax.lower() == "deepscores":
                return cid, name, cat

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

    if w <= 0:
        return None

    if h <= 0:
        return None

    return xc, yc, w, h


# =========================================================
# CLASS LIST
# =========================================================

def build_class_list(categories):
    names = []

    for _, cat in categories.items():
        name = category_name(cat)

        if not name:
            continue

        if name in EXCLUDED_CLASSES:
            continue

        tax = find_taxonomy_field(cat)

        if isinstance(tax, str):
            if "deepscore" not in tax.lower():
                continue

        if name not in names:
            names.append(name)

    return names


# =========================================================
# EXPORT
# =========================================================

def export_all_records():
    shard_paths = sorted(
        DATASET_ROOT.rglob("*.json")
    )

    if not shard_paths:
        raise RuntimeError(
            f"No JSON shards found inside {DATASET_ROOT}"
        )

    image_index = build_image_index()

    with open(shard_paths[0], "r") as f:
        sample = json.load(f)

    class_names = build_class_list(
        sample["categories"]
    )

    class_to_id = {
        c: i for i, c in enumerate(class_names)
    }

    print(f"[INFO] classes = {len(class_names)}")

    stats = {
        "per_class_boxes": Counter(),
        "images_exported": 0,
        "annotations": 0,
    }

    exported_records = []

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

            yolo_lines = []
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

                class_id = class_to_id[cls_name]

                xc, yc, w, h = bbox

                yolo_lines.append(
                    f"{class_id} "
                    f"{xc:.6f} "
                    f"{yc:.6f} "
                    f"{w:.6f} "
                    f"{h:.6f}"
                )

                image_classes.add(class_id)

                stats["per_class_boxes"][cls_name] += 1
                stats["annotations"] += 1

            if not yolo_lines:
                continue

            exported_records.append({
                "image_path": str(img_path),
                "classes": list(image_classes),
                "yolo_lines": yolo_lines,
            })

            stats["images_exported"] += 1

    # =====================================================
    # DISTRIBUTION CSV
    # =====================================================

    rows = []

    total = sum(
        stats["per_class_boxes"].values()
    )

    for cls in class_names:
        count = stats["per_class_boxes"][cls]

        pct = (
            100 * count / total
            if total > 0 else 0
        )

        rows.append({
            "class": cls,
            "count": count,
            "pct": pct,
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "count",
        ascending=False
    )

    print("\n========== CLASS DISTRIBUTION ==========\n")

    print(df.head(60))

    print("\n========================================\n")

    df.to_csv(
        EXPORT_ROOT /
        "meta" /
        "class_distribution.csv",
        index=False
    )

    with open(
        EXPORT_ROOT /
        "meta" /
        "class_names.json",
        "w"
    ) as f:
        json.dump(
            class_names,
            f,
            indent=2
        )

    print(
        f"[OK] exported records = {len(exported_records)}"
    )

    return exported_records, class_names


# =========================================================
# STRATIFIED SPLIT
# =========================================================

def split_dataset(records):
    random.seed(SEED)
    random.shuffle(records)

    split_idx = int(
        len(records) * TRAIN_RATIO
    )

    train = records[:split_idx]
    val = records[split_idx:]

    MAX_TRAIN = 60000
    MAX_VAL = 8000

    train = train[:MAX_TRAIN]
    val = val[:MAX_VAL]

    return train, val

# =========================================================
# COPY / SYMLINK
# =========================================================

def link_or_copy(src, dst):
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if USE_SYMLINKS:
        dst.symlink_to(src)

    else:
        shutil.copy2(src, dst)


# =========================================================
# PACK
# =========================================================

def pack_dataset(
    train_records,
    val_records,
    class_names
):
    used_names = set()

    for split_name, split_records in [
        ("train", train_records),
        ("val", val_records),
    ]:

        print(f"[INFO] packing {split_name}...")

        for i, rec in enumerate(split_records):
            img_src = Path(
                rec["image_path"]
            ).resolve()

            if not img_src.exists():
                continue

            yolo_lines = rec["yolo_lines"]
            stem = img_src.stem
            suffix = img_src.suffix.lower()
            base_name = f"{stem}{suffix}"

            if base_name in used_names:
                base_name = (
                    f"{stem}_{i:07d}{suffix}"
                )

            used_names.add(base_name)

            img_dst = (
                PACKED_ROOT /
                "images" /
                split_name /
                base_name
            )

            lbl_dst = (
                PACKED_ROOT /
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

            with open(lbl_dst, "w") as f:
                f.write(
                    "\n".join(yolo_lines) + "\n"
                )

    dataset_yaml = {
        "path": str(PACKED_ROOT),
        "train": "images/train",
        "val": "images/val",
        "names": {
            i: c
            for i, c in enumerate(class_names)
        }
    }

    with open(
        PACKED_ROOT / "dataset.yaml",
        "w"
    ) as f:
        yaml.safe_dump(
            dataset_yaml,
            f,
            sort_keys=False,
            allow_unicode=True
        )

    print(
        f"[OK] dataset yaml written"
    )


# =========================================================
# TRAIN
# =========================================================

def launch_training():
    cmd = [
        "yolo",
        "detect",
        "train",
        f"model={MODEL}",
        f"data={PACKED_ROOT / 'dataset.yaml'}",
        f"epochs={EPOCHS}",
        f"imgsz={IMGSZ}",
        f"batch={BATCH}",
        f"device={DEVICE}",
        f"workers={WORKERS}",
        f"cache={CACHE}",
        "optimizer=AdamW",
        "lr0=0.0005",
        "weight_decay=0.0005",
        "patience=40",
        "cos_lr=True",
        "close_mosaic=10",
        "dropout=0.05",
        "mixup=0.05",
        "copy_paste=0.0",
        "degrees=0.0",
        "translate=0.02",
        "scale=0.15",
        "fliplr=0.0",
        "hsv_h=0.0",
        "hsv_s=0.0",
        "hsv_v=0.0",
        "rect=True",
        "single_cls=False",
        "cls=3.0",
        "box=7.5",
        "dfl=1.5",
        "plots=True",
        "save=True",
        "save_json=True",
        "save_period=1",
        "exist_ok=True",
        f"project={RUNS_DIR}",
        f"name={RUN_NAME}",
    ]

    print("\n[INFO] TRAIN COMMAND\n")
    print(" ".join(map(str, cmd)))

    subprocess.run(
        cmd,
        check=True
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    best_model = (
        RUNS_DIR /
        RUN_NAME /
        "weights" /
        "best.pt"
    )

    val_cmd = [
        "yolo",
        "detect",
        "val",
        f"model={best_model}",
        f"data={PACKED_ROOT / 'dataset.yaml'}",
        "split=val",
        "plots=True",
        "save_json=True",
        "conf=0.001",
    ]

    print("\n[INFO] VALIDATION\n")

    subprocess.run(
        val_cmd,
        check=True
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n========== CONFIG ==========\n")

    print(f"DATASET_ROOT = {DATASET_ROOT}")
    print(f"WORK_ROOT    = {WORK_ROOT}")
    print(f"MODEL        = {MODEL}")
    print(f"IMGSZ        = {IMGSZ}")
    print(f"BATCH        = {BATCH}")
    print(f"EPOCHS       = {EPOCHS}")

    print("\n============================\n")

    ensure_dirs()

    records, class_names = export_all_records()

    train_records, val_records = split_dataset(
        records
    )

    print(
        f"\n[INFO] train={len(train_records)} "
        f"val={len(val_records)}"
    )

    # =====================================================
    # SPLIT STATS
    # =====================================================

    print("\n========== SPLIT STATS ==========\n")

    train_class_counter = Counter()
    val_class_counter = Counter()

    for r in train_records:
        for c in r["classes"]:
            train_class_counter[c] += 1

    for r in val_records:
        for c in r["classes"]:
            val_class_counter[c] += 1

    for cls_id, cls_name in enumerate(class_names):
        print(
            f"{cls_name:<35} "
            f"train={train_class_counter[cls_id]:>7} "
            f"val={val_class_counter[cls_id]:>7}"
        )

    print("\n=================================\n")

    pack_dataset(
        train_records,
        val_records,
        class_names
    )

    launch_training()


if __name__ == "__main__":
    main()