from ultralytics import YOLO
import pandas as pd

MODEL = "/home/vulpeslari/omr/runs/ds2_yolov8s_complete/weights/best.pt"
model = YOLO(MODEL)

DATASET_YAML = (
    "/home/vulpeslari/omr/exports/ds2_complete_curated/dataset.yaml"
)

PROJECT_DIR = (
    "/home/vulpeslari/omr/runs"
)

RUN_NAME = "ds2_yolov8s_complete"

metrics = model.val(
    data=DATASET_YAML,
    split="val",
    save_json=True,
    plots=True,
)

names = metrics.names

rows = []
for class_id, class_name in model.names.items():
    rows.append({
        "class_id": class_id,
        "class_name": class_name,

        "precision":
            float(metrics.box.p[class_id]),

        "recall":
            float(metrics.box.r[class_id]),

        "map50":
            float(metrics.box.ap50[class_id]),

        "map50_95":
            float(metrics.box.ap[class_id]),
    })

df = pd.DataFrame(rows)

report_path = (
    f"{PROJECT_DIR}/{RUN_NAME}/classification_report.csv"
)

df.to_csv(
    report_path,
    index=False
)

print("\n[OK] saved classification report:")
print(report_path)
