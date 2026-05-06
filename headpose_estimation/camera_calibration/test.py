import json
import cv2
import numpy as np
from pathlib import Path

camera_json = Path("/mnt/bikelab_data/IB_Lab/bike_interface_data/20260505/result/camera_filtered.json")
image_dir = Path("/mnt/bikelab_data/IB_Lab/bike_interface_data/20260505/camera_20240617_223649/frames")
out_path = Path("/mnt/bikelab_data/IB_Lab/bike_interface_data/20260505/result/calib_coverage.png")

with open(camera_json, "r") as f:
    data = json.load(f)

w = data["image_width"]
h = data["image_height"]
used_images = data["used_images"]
cols = data["board"]["inner_corners_cols"]
rows = data["board"]["inner_corners_rows"]
pattern_size = (cols, rows)

canvas = np.zeros((h, w, 3), dtype=np.uint8)

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    50,
    1e-4,
)

for name in used_images:
    img = cv2.imread(str(image_dir / name))
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if not ok:
        continue

    corners = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )

    pts = corners.reshape(-1, 2).astype(int)

    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)

# draw grid
for x in range(0, w, w // 4):
    cv2.line(canvas, (x, 0), (x, h), (80, 80, 80), 1)
for y in range(0, h, h // 3):
    cv2.line(canvas, (0, y), (w, y), (80, 80, 80), 1)

cv2.imwrite(str(out_path), canvas)
print("saved:", out_path)
