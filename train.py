#!/usr/bin/env python3

from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================

MODEL = "yolov8s.pt"

DATASET_YAML = (
    "/mnt/recovery/home/ubuntu/yolo_omr2/"
    "exports/ds2_compact_curated/dataset.yaml"
)

PROJECT_DIR = (
    "/mnt/recovery/home/ubuntu/yolo_omr2/runs"
)

RUN_NAME = "ds2_yolov8s_clean"

# =========================================================
# TRAIN
# =========================================================

model = YOLO(MODEL)

results = model.train(

    # dataset
    data=DATASET_YAML,

    # basic
    epochs=12,
    imgsz=640,
    batch=8,

    # hardware
    device=0,
    workers=3,
    cache=False,

    # optimizer
    optimizer="AdamW",
    lr0=0.0005,
    weight_decay=0.0005,

    # scheduler
    cos_lr=True,
    patience=20,

    # augment
    close_mosaic=10,
    dropout=0.05,
    mixup=0.05,
    copy_paste=0.0,

    degrees=0.0,
    translate=0.02,
    scale=0.15,
    fliplr=0.0,

    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,

    # layout
    rect=True,

    # logging/output
    project=PROJECT_DIR,
    name=RUN_NAME,

    # saves
    save=True,
    plots=True,
    save_json=True,

    # reproducibility
    seed=42,
)

print("\n========== TRAIN FINISHED ==========\n")

print("Best model:")
print(
    f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt"
)

print("\nLast model:")
print(
    f"{PROJECT_DIR}/{RUN_NAME}/weights/last.pt"
)

print("\n====================================\n")
