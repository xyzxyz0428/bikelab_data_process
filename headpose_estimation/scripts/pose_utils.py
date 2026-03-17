import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

try:
    from pupil_apriltags import Detector
except ImportError:
    Detector = None


def load_camera_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)
    return K, dist


def make_detector(tag_family="tagStandard41h12", quad_decimate=1.0):
    if Detector is None:
        raise ImportError("pip install pupil-apriltags")
    return Detector(
        families=tag_family,
        nthreads=4,
        quad_decimate=quad_decimate,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )


def detect_tags(detector, gray, decision_margin_min=30.0, hamming_max=0):
    dets = detector.detect(gray, estimate_tag_pose=False)
    out = []
    for d in dets:
        hamming = int(getattr(d, "hamming", 0))
        dm = float(getattr(d, "decision_margin", 999))
        if hamming <= hamming_max and dm >= decision_margin_min:
            out.append(d)
    return out


def reorder_pupil_corners(det_corners):
    """
    pupil_apriltags order: lb, rb, rt, lt
    convert to: lt, rt, rb, lb
    """
    c = np.asarray(det_corners, dtype=np.float64).reshape(4, 2)
    return np.ascontiguousarray(c[[3, 2, 1, 0], :])


def square_object_points(tag_size_m):
    s = float(tag_size_m)
    return np.array([
        [-s/2,  s/2, 0.0],   # lt
        [ s/2,  s/2, 0.0],   # rt
        [ s/2, -s/2, 0.0],   # rb
        [-s/2, -s/2, 0.0],   # lb
    ], dtype=np.float64)


def rt_to_T(Rm, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def invert_T(T):
    Rm = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = Rm.T
    Tinv[:3, 3] = -Rm.T @ t
    return Tinv


def T_to_rt(T):
    return T[:3, :3].copy(), T[:3, 3].copy()


def solve_single_tag_pose_ippe(det, tag_size_m, K, dist):
    obj = square_object_points(tag_size_m)
    img = reorder_pupil_corners(det.corners)

    ok, rvec, tvec = cv2.solvePnP(
        obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None

    Rm, _ = cv2.Rodrigues(rvec)
    T_cam_tag = rt_to_T(Rm, tvec.reshape(3))
    return T_cam_tag


def mean_transform(T_list):
    if len(T_list) == 0:
        raise ValueError("empty transform list")

    mats = np.stack([T[:3, :3] for T in T_list], axis=0)
    trans = np.stack([T[:3, 3] for T in T_list], axis=0)

    R_mean = R.from_matrix(mats).mean().as_matrix()
    t_mean = np.mean(trans, axis=0)

    return rt_to_T(R_mean, t_mean)


def rot_to_rpy_deg(Rm):
    r = R.from_matrix(Rm)
    yaw, pitch, roll = r.as_euler("zyx", degrees=True)
    return roll, pitch, yaw


def pose_rmse(obj_pts, img_pts, K, dist, T_cam_obj):
    Rm, t = T_to_rt(T_cam_obj)
    rvec, _ = cv2.Rodrigues(Rm)
    proj, _ = cv2.projectPoints(obj_pts, rvec, t.reshape(3, 1), K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))