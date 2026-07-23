import torch
import tempfile
import os

path = "/home/vulpeslari/omr/runs/ds2_yolov8s_img1600_compact/weights/best.pt"

ckpt = torch.load(path, map_location="cpu", weights_only=False)

for k, v in ckpt.items():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(v, f.name)
        size = os.path.getsize(f.name) / 1024 / 1024
    os.remove(f.name)
    print(f"{k:<15} {size:8.2f} MB")