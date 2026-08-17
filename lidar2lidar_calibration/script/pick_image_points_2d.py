#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_image_points_2d.py

Click 2D image points for manual camera-LiDAR calibration.

Output CSV columns:
    point_id,u,v,image_path,note

Controls:
    left click : add point
    u          : undo last point
    s          : save CSV
    q / ESC    : quit
"""

import argparse
from pathlib import Path
import cv2
import pandas as pd


points = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--display-scale", type=float, default=0.7)
    args = ap.parse_args()

    image_path = Path(args.image)
    img0 = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img0 is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    scale = float(args.display_scale)
    img_show_base = cv2.resize(img0, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    win = "pick 2D image points"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def redraw():
        canvas = img_show_base.copy()
        for i, p in enumerate(points):
            u_show = int(round(p["u"] * scale))
            v_show = int(round(p["v"] * scale))
            cv2.circle(canvas, (u_show, v_show), 5, (0, 0, 255), -1)
            cv2.putText(canvas, str(p["point_id"]), (u_show + 7, v_show - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        help_txt = "left click add | u undo | s save | q quit"
        cv2.putText(canvas, help_txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 3)
        cv2.putText(canvas, help_txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1)
        cv2.imshow(win, canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pid = args.start_id + len(points)
            u = x / scale
            v = y / scale
            points.append({
                "point_id": pid,
                "u": u,
                "v": v,
                "image_path": str(image_path),
                "note": ""
            })
            print(f"added point_id={pid}: u={u:.2f}, v={v:.2f}")
            redraw()

    cv2.setMouseCallback(win, on_mouse)
    redraw()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    while True:
        k = cv2.waitKey(30) & 0xFF
        if k in [27, ord("q")]:
            break
        elif k == ord("u"):
            if points:
                p = points.pop()
                print(f"undo point_id={p['point_id']}")
                redraw()
        elif k == ord("s"):
            pd.DataFrame(points).to_csv(out, index=False)
            print(f"[OK] saved {len(points)} points to {out}")

    cv2.destroyAllWindows()
    if points:
        pd.DataFrame(points).to_csv(out, index=False)
        print(f"[OK] final saved {len(points)} points to {out}")


if __name__ == "__main__":
    main()
