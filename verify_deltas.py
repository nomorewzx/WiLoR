from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def quat_to_rot(q: List[float]) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rot_to_axis_angle(R: np.ndarray) -> np.ndarray:
    trace = float(np.trace(R))
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float64)
    denom = 2.0 * np.sin(theta)
    axis = np.array(
        [
            (R[2, 1] - R[1, 2]) / denom,
            (R[0, 2] - R[2, 0]) / denom,
            (R[1, 0] - R[0, 1]) / denom,
        ],
        dtype=np.float64,
    )
    return axis * theta


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def load_frames(path: Path) -> List[Dict]:
    frames = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            frames.append(json.loads(line))
    return frames


def pick_hand_pose(hands: List[Dict], is_right: float) -> Optional[Dict]:
    for h in hands:
        if float(h.get("is_right", -1.0)) == is_right:
            return h
    return None


def get_pose(frame: Dict, is_right: Optional[float]) -> Optional[np.ndarray]:
    if is_right is None:
        base_pose = frame.get("base_pose_se3")
        if base_pose:
            rot = np.array(base_pose["rotation_matrix"], dtype=np.float64)
            t = np.array(base_pose["position"], dtype=np.float64)
            return make_transform(rot, t)
        cam = frame.get("camera_pose")
        if cam:
            rot = quat_to_rot([cam["qx"], cam["qy"], cam["qz"], cam["qw"]])
            t = np.array([cam["tx"], cam["ty"], cam["tz"]], dtype=np.float64)
            return make_transform(rot, t)
        return None

    hand = pick_hand_pose(frame.get("hands", []), is_right)
    if not hand:
        return None
    pose = hand.get("end_effector_pose_world") or hand.get("end_effector_pose_cam")
    if not pose:
        return None
    rot = np.array(pose["rotation_matrix"], dtype=np.float64)
    t = np.array(pose["position"], dtype=np.float64)
    if pose is hand.get("end_effector_pose_cam"):
        cam = frame.get("camera_pose")
        if cam is None:
            return None
        cam_rot = quat_to_rot([cam["qx"], cam["qy"], cam["qz"], cam["qw"]])
        cam_t = np.array([cam["tx"], cam["ty"], cam["tz"]], dtype=np.float64)
        rot = cam_rot @ rot
        t = cam_rot @ t + cam_t
    return make_transform(rot, t)


def rotation_error(R_true: np.ndarray, R_rec: np.ndarray) -> float:
    rel = R_true.T @ R_rec
    aa = rot_to_axis_angle(rel)
    return float(np.linalg.norm(aa))


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify chunk-relative deltas from hand_poses.jsonl.")
    parser.add_argument("--input", required=True, help="Path to hand_poses.jsonl.")
    parser.add_argument("--horizon", type=int, default=10, help="Chunk horizon for round-trip checks.")
    parser.add_argument("--stride", type=int, default=1, help="Stride between chunk starts.")
    parser.add_argument("--max_chunks", type=int, default=0, help="Optional cap on chunks checked.")
    args = parser.parse_args()

    frames = load_frames(Path(args.input))
    if args.horizon <= 0 or args.stride <= 0:
        raise ValueError("horizon and stride must be positive")

    translation_errors = {"left": [], "right": [], "base": []}
    rotation_errors = {"left": [], "right": [], "base": []}
    delta_norms = {"left": [], "right": [], "base": []}
    delta_angles = {"left": [], "right": [], "base": []}

    chunk_count = 0
    for t in range(0, len(frames) - args.horizon, args.stride):
        if args.max_chunks and chunk_count >= args.max_chunks:
            break
        frame_t = frames[t]

        for name, hand_id in [("left", 0.0), ("right", 1.0), ("base", None)]:
            T_t = get_pose(frame_t, hand_id)
            if T_t is None:
                continue
            T_t_inv = invert_transform(T_t)
            for k in range(1, args.horizon + 1):
                frame_k = frames[t + k]
                T_k = get_pose(frame_k, hand_id)
                if T_k is None:
                    continue
                T_rel = T_t_inv @ T_k
                T_rec = T_t @ T_rel
                trans_err = float(np.linalg.norm(T_k[:3, 3] - T_rec[:3, 3]))
                rot_err = rotation_error(T_k[:3, :3], T_rec[:3, :3])
                translation_errors[name].append(trans_err)
                rotation_errors[name].append(rot_err)
                delta_norms[name].append(float(np.linalg.norm(T_rel[:3, 3])))
                delta_angles[name].append(float(np.linalg.norm(rot_to_axis_angle(T_rel[:3, :3]))))

        chunk_count += 1

    report = {
        "round_trip_translation": {k: summarize(v) for k, v in translation_errors.items()},
        "round_trip_rotation": {k: summarize(v) for k, v in rotation_errors.items()},
        "delta_translation_norm": {k: summarize(v) for k, v in delta_norms.items()},
        "delta_rotation_angle": {k: summarize(v) for k, v in delta_angles.items()},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
