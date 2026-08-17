#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario_headpose_and_gaze_export_middle_lidar.py

Final output: gaze_repr.csv in middle_lidar frame.

It combines:
- Tobii raw gaze data
- dynamic headpose result T_C_H
- static bike extrinsics T_B_C and T_B_M

Transform chain:
    gaze in Head frame -> middle_lidar frame
    T_M_H = inv(T_B_M) @ T_B_C @ T_C_H

Output columns:
    gaze_A_point_M_x/y/z
    gaze_B_ray_origin_M_x/y/z
    gaze_B_ray_dir_M_x/y/z
    gaze_C_ray_origin_M_x/y/z
    gaze_C_ray_dir_M_x/y/z
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-12:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return v / n


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=float)
    return (T @ ph)[:3]


def transform_dir(T: np.ndarray, d: np.ndarray) -> np.ndarray:
    return normalize(T[:3, :3] @ d)


def parse_validity(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0) > 0
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["valid", "1", "true", "yes"])


# ---------- static extrinsics ----------
def _rot_mats_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float):
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]], dtype=float)

    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0,          1, 0],
                   [-np.sin(p), 0, np.cos(p)]], dtype=float)

    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,         1]], dtype=float)
    return Rx, Ry, Rz


def rpy_deg_to_R_static(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    Static bike extrinsics follow bike_extrinsics.json:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Do NOT change this when testing the headpose Euler order.
    """
    Rx, Ry, Rz = _rot_mats_from_rpy_deg(roll_deg, pitch_deg, yaw_deg)
    return Rz @ Ry @ Rx


def rpy_deg_to_R_headpose(roll_deg: float, pitch_deg: float, yaw_deg: float, order: str = "XYZ") -> np.ndarray:
    """
    Dynamic headpose Euler reconstruction.

    For your current headpose.csv, the cam_head_* angles are expected to use XYZ.
    Therefore the default is:
        R = Rx(roll) @ Ry(pitch) @ Rz(yaw)

    If your headpose exporter later writes quaternion or a 4x4 matrix, prefer that
    and avoid Euler ambiguity.
    """
    Rx, Ry, Rz = _rot_mats_from_rpy_deg(roll_deg, pitch_deg, yaw_deg)
    order = order.upper()

    if order == "XYZ":
        return Rx @ Ry @ Rz
    if order == "ZYX":
        return Rz @ Ry @ Rx

    raise ValueError(f"Unsupported Euler order: {order}. Use XYZ or ZYX.")


def make_T_static(tx, ty, tz, roll_deg, pitch_deg, yaw_deg) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_deg_to_R_static(roll_deg, pitch_deg, yaw_deg)
    T[:3, 3] = [tx, ty, tz]
    return T


def make_T_headpose(tx, ty, tz, roll_deg, pitch_deg, yaw_deg, order: str = "XYZ") -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_deg_to_R_headpose(roll_deg, pitch_deg, yaw_deg, order=order)
    T[:3, 3] = [tx, ty, tz]
    return T


# Backward-compatible alias used only by legacy fallback code.
# It uses the headpose convention, not the static extrinsic convention.
def make_T(tx, ty, tz, roll_deg, pitch_deg, yaw_deg) -> np.ndarray:
    return make_T_headpose(tx, ty, tz, roll_deg, pitch_deg, yaw_deg, order="XYZ")


def build_transform_graph(extrinsics_json: Path) -> Dict[Tuple[str, str], np.ndarray]:
    data = read_json(extrinsics_json)
    graph = {}
    for tr in data["transforms"]:
        parent = tr["parent"]
        child = tr["child"]
        t = tr["translation"]
        r = tr["rotation_rpy_deg"]
        T_parent_child = make_T_static(float(t["x"]), float(t["y"]), float(t["z"]),
                                       float(r["roll"]), float(r["pitch"]), float(r["yaw"]))
        graph[(parent, child)] = T_parent_child
        graph[(child, parent)] = np.linalg.inv(T_parent_child)
    return graph


def find_transform(graph: Dict[Tuple[str, str], np.ndarray], parent: str, child: str) -> np.ndarray:
    """Return T_parent_child, mapping p_child -> p_parent."""
    if (parent, child) in graph:
        return graph[(parent, child)]

    frames = sorted(set([a for a, _ in graph.keys()] + [b for _, b in graph.keys()]))
    neighbors = {f: [] for f in frames}
    for (a, b), T_a_b in graph.items():
        neighbors[a].append((b, T_a_b))

    visited = {parent}
    queue = [(parent, np.eye(4))]
    while queue:
        current, T_parent_current = queue.pop(0)
        if current == child:
            return T_parent_current
        for nb, T_current_nb in neighbors.get(current, []):
            if nb in visited:
                continue
            visited.add(nb)
            queue.append((nb, T_parent_current @ T_current_nb))
    raise KeyError(f"Cannot find transform T_{parent}_{child}")


# ---------- dynamic headpose ----------
def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        return np.eye(3)
    q = q / n
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=float)


def make_T_from_quat(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = quat_to_R(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return T


def parse_headpose_row_to_T_C_H(row: pd.Series, prefer_pose: str = "cam_head", headpose_euler_order: str = "XYZ") -> np.ndarray:
    """
    Return T_C_H: maps p_head -> p_camera.

    Current headpose.csv format:
        cam_head_tx, cam_head_ty, cam_head_tz,
        cam_head_roll_deg, cam_head_pitch_deg, cam_head_yaw_deg

    cam_head_* is interpreted as T_C_H.
    Dynamic headpose Euler order is controlled by --headpose-euler-order.
    Static bike extrinsics are handled separately and always follow the JSON convention.
    """

    def vals_for(cols):
        vals = [pd.to_numeric(row.get(c, np.nan), errors="coerce") for c in cols]
        vals = np.asarray(vals, dtype=float)
        return vals if np.all(np.isfinite(vals)) else None

    cam_head_cols = [
        "cam_head_tx", "cam_head_ty", "cam_head_tz",
        "cam_head_roll_deg", "cam_head_pitch_deg", "cam_head_yaw_deg",
    ]
    back_head_cols = [
        "back_head_tx", "back_head_ty", "back_head_tz",
        "back_head_roll_deg", "back_head_pitch_deg", "back_head_yaw_deg",
    ]

    if prefer_pose == "cam_head" and all(c in row.index for c in cam_head_cols):
        vals = vals_for(cam_head_cols)
        if vals is not None:
            return make_T_headpose(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5],
                                   order=headpose_euler_order)

    if prefer_pose == "back_head" and all(c in row.index for c in back_head_cols):
        vals = vals_for(back_head_cols)
        if vals is not None:
            return make_T_headpose(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5],
                                   order=headpose_euler_order)

    # fallback: if requested source has invalid values, try cam_head once more
    if all(c in row.index for c in cam_head_cols):
        vals = vals_for(cam_head_cols)
        if vals is not None:
            return make_T_headpose(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5],
                                   order=headpose_euler_order)

    quat_sets = [
        ["tx", "ty", "tz", "qx", "qy", "qz", "qw"],
        ["t_x", "t_y", "t_z", "q_x", "q_y", "q_z", "q_w"],
        ["head_tx", "head_ty", "head_tz", "head_qx", "head_qy", "head_qz", "head_qw"],
        ["x", "y", "z", "qx", "qy", "qz", "qw"],
    ]
    for cols in quat_sets:
        if all(c in row.index for c in cols):
            vals = vals_for(cols)
            if vals is not None:
                return make_T_from_quat(*vals)

    rpy_sets = [
        ["tx", "ty", "tz", "roll", "pitch", "yaw"],
        ["t_x", "t_y", "t_z", "roll", "pitch", "yaw"],
    ]
    for cols in rpy_sets:
        if all(c in row.index for c in cols):
            vals = vals_for(cols)
            if vals is not None:
                return make_T_headpose(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5],
                                       order=headpose_euler_order)

    raise ValueError(
        "Cannot parse headpose row. Expected cam_head_* columns or other supported pose columns. "
        f"Available columns: {list(row.index)}"
    )

# ---------- Tobii / scene camera ----------
def load_tobii_scene_transforms(transforms_json: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = read_json(transforms_json)
    if "T_H_HUCS" in data:
        T_H_HUCS = np.asarray(data["T_H_HUCS"], dtype=float).reshape(4, 4)
    elif "T_H_C1" in data and "T_C1_HUCS" in data:
        T_H_HUCS = (np.asarray(data["T_H_C1"], dtype=float).reshape(4, 4)
                    @ np.asarray(data["T_C1_HUCS"], dtype=float).reshape(4, 4))
    else:
        raise KeyError("Need T_H_HUCS or both T_H_C1 and T_C1_HUCS in transforms.json")

    if "T_C1_HUCS" not in data:
        raise KeyError("Method C needs T_C1_HUCS in transforms.json")
    T_C1_HUCS = np.asarray(data["T_C1_HUCS"], dtype=float).reshape(4, 4)
    return T_H_HUCS, T_C1_HUCS


def read_camera_intrinsics(camera_json: Path):
    cam = read_json(camera_json)
    if all(k in cam for k in ["fx", "fy", "cx", "cy"]):
        return float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
    if "K" in cam:
        K = np.asarray(cam["K"], dtype=float)
        if K.shape == (3, 3):
            return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        K = K.reshape(-1)
        if K.size == 9:
            return float(K[0]), float(K[4]), float(K[2]), float(K[5])
    if "camera_matrix" in cam:
        cm = cam["camera_matrix"]
        K = np.asarray(cm["data"] if isinstance(cm, dict) and "data" in cm else cm, dtype=float)
        if K.shape == (3, 3):
            return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        K = K.reshape(-1)
        if K.size == 9:
            return float(K[0]), float(K[4]), float(K[2]), float(K[5])
    if "intrinsics" in cam:
        intr = np.asarray(cam["intrinsics"], dtype=float)
        if intr.shape == (3, 3):
            return float(intr[0, 0]), float(intr[1, 1]), float(intr[0, 2]), float(intr[1, 2])
        intr = intr.reshape(-1)
        if intr.size == 4:
            return float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])
        if intr.size == 9:
            return float(intr[0]), float(intr[4]), float(intr[2]), float(intr[5])
    raise KeyError(f"Cannot find camera intrinsics. Available top-level keys: {list(cam.keys())}")


def build_tobii_unix_ns(tobii: pd.DataFrame) -> pd.DataFrame:
    tobii = tobii.copy()
    if "t_unix_ns" in tobii.columns:
        tobii["t_unix_ns"] = pd.to_numeric(tobii["t_unix_ns"], errors="coerce")
        return tobii.dropna(subset=["t_unix_ns"]).copy()

    required = ["Recording date UTC", "Recording start time UTC", "Recording timestamp"]
    missing = [c for c in required if c not in tobii.columns]
    if missing:
        raise ValueError(f"eyetracker_raw.xlsx missing required columns: {missing}")

    base_dt = pd.to_datetime(
        tobii["Recording date UTC"].astype(str).str.strip() + " " +
        tobii["Recording start time UTC"].astype(str).str.strip(),
        errors="coerce", utc=True
    )
    rec = pd.to_numeric(tobii["Recording timestamp"], errors="coerce")
    rec_med = np.nanmedian(rec.to_numpy(dtype=float))
    if np.isfinite(rec_med):
        if rec_med > 1e6:
            rec_ns = rec * 1000.0
        elif rec_med > 1e3:
            rec_ns = rec * 1e6
        else:
            rec_ns = rec * 1e9
    else:
        rec_ns = rec * 1000.0

    base_ns = pd.Series(base_dt.astype("int64"), index=tobii.index)
    base_ns = base_ns.where(base_dt.notna(), pd.NA)
    tobii["t_unix_ns"] = pd.array(base_ns + rec_ns, dtype="Int64")
    tobii["t_unix_ns"] = pd.to_numeric(tobii["t_unix_ns"], errors="coerce")
    return tobii.dropna(subset=["t_unix_ns"]).copy()


def compute_binocular_hucs_ray(df: pd.DataFrame):
    vl = parse_validity(df["Validity left"]).to_numpy()
    vr = parse_validity(df["Validity right"]).to_numpy()

    lx = pd.to_numeric(df["Gaze direction left X"], errors="coerce").to_numpy(float)
    ly = pd.to_numeric(df["Gaze direction left Y"], errors="coerce").to_numpy(float)
    lz = pd.to_numeric(df["Gaze direction left Z"], errors="coerce").to_numpy(float)
    rx = pd.to_numeric(df["Gaze direction right X"], errors="coerce").to_numpy(float)
    ry = pd.to_numeric(df["Gaze direction right Y"], errors="coerce").to_numpy(float)
    rz = pd.to_numeric(df["Gaze direction right Z"], errors="coerce").to_numpy(float)

    pxl = pd.to_numeric(df["Pupil position left X"], errors="coerce").to_numpy(float)
    pyl = pd.to_numeric(df["Pupil position left Y"], errors="coerce").to_numpy(float)
    pzl = pd.to_numeric(df["Pupil position left Z"], errors="coerce").to_numpy(float)
    pxr = pd.to_numeric(df["Pupil position right X"], errors="coerce").to_numpy(float)
    pyr = pd.to_numeric(df["Pupil position right Y"], errors="coerce").to_numpy(float)
    pzr = pd.to_numeric(df["Pupil position right Z"], errors="coerce").to_numpy(float)

    origin = np.full((len(df), 3), np.nan)
    direc = np.full((len(df), 3), np.nan)

    both = vl & vr
    left_only = vl & (~vr)
    right_only = vr & (~vl)

    origin[both] = np.column_stack([(pxl[both] + pxr[both]) / 2.0,
                                    (pyl[both] + pyr[both]) / 2.0,
                                    (pzl[both] + pzr[both]) / 2.0])
    direc[both] = np.column_stack([(lx[both] + rx[both]) / 2.0,
                                   (ly[both] + ry[both]) / 2.0,
                                   (lz[both] + rz[both]) / 2.0])
    origin[left_only] = np.column_stack([pxl[left_only], pyl[left_only], pzl[left_only]])
    direc[left_only] = np.column_stack([lx[left_only], ly[left_only], lz[left_only]])
    origin[right_only] = np.column_stack([pxr[right_only], pyr[right_only], pzr[right_only]])
    direc[right_only] = np.column_stack([rx[right_only], ry[right_only], rz[right_only]])

    for i in range(len(df)):
        if np.all(np.isfinite(direc[i])):
            direc[i] = normalize(direc[i])
    return origin, direc


def build_method_c_rays(df: pd.DataFrame, scene_camera_json: Path, T_H_C1: np.ndarray):
    fx, fy, cx, cy = read_camera_intrinsics(scene_camera_json)
    u = pd.to_numeric(df["Gaze point X"], errors="coerce").to_numpy(float)
    v = pd.to_numeric(df["Gaze point Y"], errors="coerce").to_numpy(float)
    origin = np.full((len(df), 3), np.nan)
    direc = np.full((len(df), 3), np.nan)
    cam_origin_H = T_H_C1[:3, 3]
    for i in range(len(df)):
        if not (np.isfinite(u[i]) and np.isfinite(v[i])):
            continue
        ray_c1 = normalize(np.array([(u[i] - cx) / fx, (v[i] - cy) / fy, 1.0], dtype=float))
        origin[i] = cam_origin_H
        direc[i] = transform_dir(T_H_C1, ray_c1)
    return origin, direc


# ---------- scenario/time helpers ----------
def nearest_join(left: pd.DataFrame, right: pd.DataFrame, tolerance_ns: int) -> pd.DataFrame:
    left = left.copy()
    right = right.copy()
    left["t_unix_ns"] = pd.to_numeric(left["t_unix_ns"], errors="coerce")
    right["t_unix_ns"] = pd.to_numeric(right["t_unix_ns"], errors="coerce")
    left = left.dropna(subset=["t_unix_ns"]).copy()
    right = right.dropna(subset=["t_unix_ns"]).copy()
    left["t_unix_ns"] = left["t_unix_ns"].astype("int64")
    right["t_unix_ns"] = right["t_unix_ns"].astype("int64")
    left = left.sort_values("t_unix_ns").reset_index(drop=True)
    right = right.sort_values("t_unix_ns").reset_index(drop=True)
    return pd.merge_asof(left, right, on="t_unix_ns", direction="nearest", tolerance=int(tolerance_ns))


def load_scenarios_relative(path: Path, wanted_ids):
    df = normalize_cols(pd.read_csv(path))
    colmap = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "scenario_id":
            colmap[c] = "scenario_id"
        elif cl == "scenario_type":
            colmap[c] = "scenario_type"
        elif cl.startswith("initial time"):
            colmap[c] = "initial_time"
        elif cl == "start":
            colmap[c] = "start_rel_s"
        elif cl == "end":
            colmap[c] = "end_rel_s"
        elif cl == "note":
            colmap[c] = "note"
    df = df.rename(columns=colmap)
    required = ["scenario_id", "initial_time", "start_rel_s", "end_rel_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"critical_scenarios.csv missing required columns: {missing}")
    df["scenario_id"] = df["scenario_id"].astype(str)
    df["initial_time"] = pd.to_numeric(df["initial_time"], errors="coerce")
    df["start_rel_s"] = pd.to_numeric(df["start_rel_s"], errors="coerce")
    df["end_rel_s"] = pd.to_numeric(df["end_rel_s"], errors="coerce")
    df = df.dropna(subset=["initial_time", "start_rel_s", "end_rel_s"]).copy()
    df["start_ns"] = ((df["initial_time"] + df["start_rel_s"]) * 1e9).round().astype(np.int64)
    df["end_ns"] = ((df["initial_time"] + df["end_rel_s"]) * 1e9).round().astype(np.int64)
    if wanted_ids:
        wanted = set(str(x) for x in wanted_ids)
        df = df[df["scenario_id"].isin(wanted)].copy()
    return df


def attach_headpose_time(head: pd.DataFrame, scenario_ts: pd.DataFrame) -> pd.DataFrame:
    head = normalize_cols(head)
    scenario_ts = normalize_cols(scenario_ts)
    if "t_unix_ns" in head.columns:
        head["t_unix_ns"] = pd.to_numeric(head["t_unix_ns"], errors="coerce")
        return head.dropna(subset=["t_unix_ns"]).copy()
    if "frame_idx" in head.columns and "frame_idx" in scenario_ts.columns and "unix_ns" in scenario_ts.columns:
        tmp = head.copy()
        tmp["frame_idx"] = pd.to_numeric(tmp["frame_idx"], errors="coerce")
        ts = scenario_ts[["frame_idx", "unix_ns"]].copy()
        ts["frame_idx"] = pd.to_numeric(ts["frame_idx"], errors="coerce")
        ts["unix_ns"] = pd.to_numeric(ts["unix_ns"], errors="coerce")
        merged = tmp.merge(ts, on="frame_idx", how="left")
        merged = merged.rename(columns={"unix_ns": "t_unix_ns"})
        merged["t_unix_ns"] = pd.to_numeric(merged["t_unix_ns"], errors="coerce")
        return merged.dropna(subset=["t_unix_ns"]).copy()
    raise ValueError("headpose.csv must contain t_unix_ns, or contain frame_idx for merging with timestamps_scenario.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios-csv", required=True)
    ap.add_argument("--scenario-ids", nargs="+", default=["5", "7"])
    ap.add_argument("--timestamps-csv", required=True)
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--estimate-headpose-script", "--estimate_headpose_script", dest="estimate_headpose_script", required=True)
    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--headpose-config-json", required=True)
    ap.add_argument("--rig-calib-json", required=True)
    ap.add_argument("--neutral-frame", default=None)
    ap.add_argument("--scene-camera-json", required=True)
    ap.add_argument("--transforms-json", required=True)
    ap.add_argument("--extrinsics-json", required=True)
    ap.add_argument("--headpose-camera-frame", choices=["camera_link", "camera_optical_frame"], default="camera_link")
    ap.add_argument("--headpose-pose-source", choices=["cam_head", "back_head"], default="cam_head", help="Which headpose columns to use. Current recommended value: cam_head.")
    ap.add_argument("--headpose-euler-order", choices=["XYZ", "ZYX"], default="XYZ", help="Euler order used by dynamic cam_head_* / back_head_* angles. Static bike extrinsics always use JSON convention Rz@Ry@Rx.")
    ap.add_argument("--target-frame", default="middle_lidar", help="Final gaze output frame. Default: middle_lidar")
    ap.add_argument("--tobii-raw-xlsx", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--merge-tolerance-ms", type=float, default=20.0)
    ap.add_argument("--debug-gaze", action="store_true", help="Print validity counts and yaw statistics for A/B/C outputs.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios_relative(Path(args.scenarios_csv), args.scenario_ids)

    frame_ts = normalize_cols(pd.read_csv(args.timestamps_csv))
    if "frame_idx" not in frame_ts.columns or "unix_ns" not in frame_ts.columns:
        raise ValueError("timestamps csv must contain frame_idx and unix_ns")
    frame_ts["unix_ns"] = pd.to_numeric(frame_ts["unix_ns"], errors="coerce")
    frame_ts = frame_ts.dropna(subset=["unix_ns"]).copy()

    tobii = normalize_cols(pd.read_excel(args.tobii_raw_xlsx))
    tobii = build_tobii_unix_ns(tobii)
    if "Sensor" in tobii.columns:
        sub = tobii[tobii["Sensor"].astype(str).str.strip().isin(["Eye Tracker", ""])]
        if len(sub) > 0:
            tobii = sub

    T_H_HUCS, T_C1_HUCS = load_tobii_scene_transforms(Path(args.transforms_json))
    T_H_C1 = T_H_HUCS @ np.linalg.inv(T_C1_HUCS)

    graph = build_transform_graph(Path(args.extrinsics_json))
    T_B_C = find_transform(graph, "base_link", args.headpose_camera_frame)
    T_B_T = find_transform(graph, "base_link", args.target_frame)
    T_T_B = np.linalg.inv(T_B_T)

    frame_suffix = "M" if args.target_frame == "middle_lidar" else args.target_frame
    print(f"[INFO] Final gaze output frame: {args.target_frame} (suffix: {frame_suffix})")
    print(f"[INFO] Headpose camera frame: {args.headpose_camera_frame}")
    print(f"[INFO] Static extrinsics Euler order: ZYX from bike_extrinsics.json convention")
    print(f"[INFO] Dynamic headpose Euler order: {args.headpose_euler_order}")

    for _, srow in scenarios.iterrows():
        sid = str(srow["scenario_id"])
        start_ns, end_ns = int(srow["start_ns"]), int(srow["end_ns"])
        frame_sub = frame_ts[(frame_ts["unix_ns"] >= start_ns) & (frame_ts["unix_ns"] <= end_ns)].copy()
        if len(frame_sub) == 0:
            print(f"[WARN] scenario {sid}: no frames in range")
            continue

        scenario_dir = outdir / f"scenario_{sid}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_ts_csv = scenario_dir / "timestamps_scenario.csv"
        frame_sub.to_csv(scenario_ts_csv, index=False)
        headpose_csv = scenario_dir / "headpose.csv"

        cmd = [
            "python", args.estimate_headpose_script,
            "--camera", args.camera_json,
            "--config", args.headpose_config_json,
            "--rig-calib", args.rig_calib_json,
            "--frame-dir", args.frame_dir,
            "--timestamps-csv", str(scenario_ts_csv),
            "--output-csv", str(headpose_csv),
        ]
        if args.neutral_frame:
            cmd.extend(["--neutral-frame", args.neutral_frame])

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            if headpose_csv.exists() and headpose_csv.stat().st_size > 0:
                print(f"[WARN] headpose script returned {result.returncode}, but output exists and will be used: {headpose_csv}")
            else:
                raise RuntimeError(f"Headpose script failed with return code {result.returncode} and no usable output: {headpose_csv}")

        head_raw = normalize_cols(pd.read_csv(headpose_csv))
        head = attach_headpose_time(head_raw, frame_sub)

        tobii_seg = tobii[(tobii["t_unix_ns"] >= start_ns) & (tobii["t_unix_ns"] <= end_ns)].copy()
        if len(tobii_seg) == 0:
            print(f"[WARN] scenario {sid}: no tobii rows")
            continue

        merged = nearest_join(tobii_seg, head, tolerance_ns=int(args.merge_tolerance_ms * 1e6))

        # 1) Compute gaze in Head frame
        gaze_point_H = np.full((len(merged), 3), np.nan)
        for i, (_, r) in enumerate(merged.iterrows()):
            vals = [
                pd.to_numeric(pd.Series([r.get("Gaze point 3D X")]), errors="coerce").iloc[0],
                pd.to_numeric(pd.Series([r.get("Gaze point 3D Y")]), errors="coerce").iloc[0],
                pd.to_numeric(pd.Series([r.get("Gaze point 3D Z")]), errors="coerce").iloc[0],
            ]
            if np.all(np.isfinite(vals)):
                gaze_point_H[i] = transform_point(T_H_HUCS, np.array(vals, dtype=float) / 1000.0)

        origin_HUCS, dir_HUCS = compute_binocular_hucs_ray(merged)
        rayB_origin_H = np.full((len(merged), 3), np.nan)
        rayB_dir_H = np.full((len(merged), 3), np.nan)
        for i in range(len(merged)):
            if np.all(np.isfinite(origin_HUCS[i])):
                rayB_origin_H[i] = transform_point(T_H_HUCS, origin_HUCS[i])
            if np.all(np.isfinite(dir_HUCS[i])):
                rayB_dir_H[i] = transform_dir(T_H_HUCS, dir_HUCS[i])

        rayC_origin_H, rayC_dir_H = build_method_c_rays(merged, Path(args.scene_camera_json), T_H_C1)

        # 2) Transform gaze from Head frame to target frame, default middle_lidar
        gaze_A_point_T = np.full((len(merged), 3), np.nan)
        rayB_origin_T = np.full((len(merged), 3), np.nan)
        rayB_dir_T = np.full((len(merged), 3), np.nan)
        rayC_origin_T = np.full((len(merged), 3), np.nan)
        rayC_dir_T = np.full((len(merged), 3), np.nan)

        parse_ok = 0
        parse_fail = 0
        first_parse_error = None

        for i, (_, r) in enumerate(merged.iterrows()):
            try:
                T_C_H = parse_headpose_row_to_T_C_H(
                    r,
                    prefer_pose=args.headpose_pose_source,
                    headpose_euler_order=args.headpose_euler_order,
                )
                T_T_H = T_T_B @ T_B_C @ T_C_H
                parse_ok += 1
            except Exception as e:
                parse_fail += 1
                if first_parse_error is None:
                    first_parse_error = str(e)
                continue

            if np.all(np.isfinite(gaze_point_H[i])):
                gaze_A_point_T[i] = transform_point(T_T_H, gaze_point_H[i])
            if np.all(np.isfinite(rayB_origin_H[i])):
                rayB_origin_T[i] = transform_point(T_T_H, rayB_origin_H[i])
            if np.all(np.isfinite(rayB_dir_H[i])):
                rayB_dir_T[i] = transform_dir(T_T_H, rayB_dir_H[i])
            if np.all(np.isfinite(rayC_origin_H[i])):
                rayC_origin_T[i] = transform_point(T_T_H, rayC_origin_H[i])
            if np.all(np.isfinite(rayC_dir_H[i])):
                rayC_dir_T[i] = transform_dir(T_T_H, rayC_dir_H[i])

        if args.debug_gaze:
            def n_valid(arr):
                return int(np.isfinite(arr).all(axis=1).sum())

            def yaw_stats(arr):
                m = np.isfinite(arr).all(axis=1)
                if not np.any(m):
                    return "no valid rows"
                d = arr[m]
                yaw = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
                angle_to_plus_x = np.degrees(np.arccos(np.clip(d[:, 0] / np.linalg.norm(d, axis=1), -1.0, 1.0)))
                return {
                    "mean_dir": np.nanmean(d, axis=0).round(4).tolist(),
                    "yaw_p05_p50_p95_deg": np.percentile(yaw, [5, 50, 95]).round(2).tolist(),
                    "angle_to_plus_x_p05_p50_p95_deg": np.percentile(angle_to_plus_x, [5, 50, 95]).round(2).tolist(),
                }

            print(f"[DEBUG] scenario {sid}: merged rows = {len(merged)}")
            print(f"[DEBUG] scenario {sid}: headpose rows = {len(head)}")
            print(f"[DEBUG] scenario {sid}: parse_ok = {parse_ok}, parse_fail = {parse_fail}")
            if first_parse_error:
                print(f"[DEBUG] scenario {sid}: first parse error: {first_parse_error}")
            print(f"[DEBUG] scenario {sid}: A_H valid = {n_valid(gaze_point_H)}, A_M valid = {n_valid(gaze_A_point_T)}")
            print(f"[DEBUG] scenario {sid}: B_H origin valid = {n_valid(rayB_origin_H)}, B_H dir valid = {n_valid(rayB_dir_H)}")
            print(f"[DEBUG] scenario {sid}: B_M origin valid = {n_valid(rayB_origin_T)}, B_M dir valid = {n_valid(rayB_dir_T)}")
            print(f"[DEBUG] scenario {sid}: C_H origin valid = {n_valid(rayC_origin_H)}, C_H dir valid = {n_valid(rayC_dir_H)}")
            print(f"[DEBUG] scenario {sid}: C_M origin valid = {n_valid(rayC_origin_T)}, C_M dir valid = {n_valid(rayC_dir_T)}")
            print(f"[DEBUG] scenario {sid}: B_M yaw stats = {yaw_stats(rayB_dir_T)}")
            print(f"[DEBUG] scenario {sid}: C_M yaw stats = {yaw_stats(rayC_dir_T)}")

        out = pd.DataFrame({
            "t_unix_ns": merged["t_unix_ns"].astype(np.int64),
            "Validity left": merged.get("Validity left"),
            "Validity right": merged.get("Validity right"),
            "Eye movement type": merged.get("Eye movement type"),
            "Eye movement event duration": merged.get("Eye movement event duration"),
            "headpose_parse_ok": np.isfinite(gaze_A_point_T).any(axis=1) | np.isfinite(rayB_origin_T).any(axis=1) | np.isfinite(rayC_origin_T).any(axis=1),

            f"gaze_A_point_{frame_suffix}_x": gaze_A_point_T[:, 0],
            f"gaze_A_point_{frame_suffix}_y": gaze_A_point_T[:, 1],
            f"gaze_A_point_{frame_suffix}_z": gaze_A_point_T[:, 2],

            f"gaze_B_ray_origin_{frame_suffix}_x": rayB_origin_T[:, 0],
            f"gaze_B_ray_origin_{frame_suffix}_y": rayB_origin_T[:, 1],
            f"gaze_B_ray_origin_{frame_suffix}_z": rayB_origin_T[:, 2],
            f"gaze_B_ray_dir_{frame_suffix}_x": rayB_dir_T[:, 0],
            f"gaze_B_ray_dir_{frame_suffix}_y": rayB_dir_T[:, 1],
            f"gaze_B_ray_dir_{frame_suffix}_z": rayB_dir_T[:, 2],

            f"gaze_C_ray_origin_{frame_suffix}_x": rayC_origin_T[:, 0],
            f"gaze_C_ray_origin_{frame_suffix}_y": rayC_origin_T[:, 1],
            f"gaze_C_ray_origin_{frame_suffix}_z": rayC_origin_T[:, 2],
            f"gaze_C_ray_dir_{frame_suffix}_x": rayC_dir_T[:, 0],
            f"gaze_C_ray_dir_{frame_suffix}_y": rayC_dir_T[:, 1],
            f"gaze_C_ray_dir_{frame_suffix}_z": rayC_dir_T[:, 2],
        })

        out.to_csv(scenario_dir / "gaze_repr.csv", index=False)
        head.to_csv(scenario_dir / "headpose_with_time.csv", index=False)
        print(f"[OK] scenario {sid} -> {scenario_dir}")


if __name__ == "__main__":
    main()
