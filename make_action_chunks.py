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
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
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


def pose_to_action(T: np.ndarray) -> List[float]:
    t = T[:3, 3]
    r = rot_to_axis_angle(T[:3, :3])
    return [float(t[0]), float(t[1]), float(t[2]), float(r[0]), float(r[1]), float(r[2])]


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


def hand_to_transform(hand: Dict) -> Optional[np.ndarray]:
    end_eff = hand.get("end_effector_pose")
    if not end_eff:
        return None
    pos = np.array(end_eff["position"], dtype=np.float64)
    rot = np.array(end_eff["rotation_matrix"], dtype=np.float64)
    return make_transform(rot, pos)


def camera_to_transform(cam: Dict) -> Optional[np.ndarray]:
    if not cam:
        return None
    t = np.array([cam["tx"], cam["ty"], cam["tz"]], dtype=np.float64)
    q = [cam["qx"], cam["qy"], cam["qz"], cam["qw"]]
    R = quat_to_rot(q)
    return make_transform(R, t)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create action chunks from hand/camera poses.")
    parser.add_argument("--input", required=True, help="Path to hand_poses.jsonl.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--horizon", type=int, required=True, help="Chunk horizon H (number of future steps).")
    parser.add_argument("--stride", type=int, default=1, help="Stride between chunk starts.")
    args = parser.parse_args()

    frames = load_frames(Path(args.input))
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if args.stride <= 0:
        raise ValueError("stride must be positive")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        for t in range(0, len(frames) - args.horizon, args.stride):
            frame_t = frames[t]
            cam_t = camera_to_transform(frame_t.get("camera_pose"))
            if cam_t is None:
                continue
            cam_t_inv = invert_transform(cam_t)

            hands_t = frame_t.get("hands", [])
            left_t = pick_hand_pose(hands_t, 0.0)
            right_t = pick_hand_pose(hands_t, 1.0)
            left_T_t = hand_to_transform(left_t) if left_t else None
            right_T_t = hand_to_transform(right_t) if right_t else None

            left_actions = []
            right_actions = []
            base_actions = []
            left_valid = []
            right_valid = []

            for k in range(1, args.horizon + 1):
                frame_k = frames[t + k]
                cam_k = camera_to_transform(frame_k.get("camera_pose"))
                if cam_k is None:
                    base_actions.append(None)
                else:
                    base_rel = cam_t_inv @ cam_k
                    base_actions.append(pose_to_action(base_rel))

                hands_k = frame_k.get("hands", [])
                left_k = pick_hand_pose(hands_k, 0.0)
                right_k = pick_hand_pose(hands_k, 1.0)
                left_T_k = hand_to_transform(left_k) if left_k else None
                right_T_k = hand_to_transform(right_k) if right_k else None

                if left_T_t is not None and left_T_k is not None and cam_k is not None:
                    world_T_hand_k = cam_k @ left_T_k
                    camt_T_hand_k = cam_t_inv @ world_T_hand_k
                    left_rel = invert_transform(left_T_t) @ camt_T_hand_k
                    left_actions.append(pose_to_action(left_rel))
                    left_valid.append(True)
                else:
                    left_actions.append(None)
                    left_valid.append(False)

                if right_T_t is not None and right_T_k is not None and cam_k is not None:
                    world_T_hand_k = cam_k @ right_T_k
                    camt_T_hand_k = cam_t_inv @ world_T_hand_k
                    right_rel = invert_transform(right_T_t) @ camt_T_hand_k
                    right_actions.append(pose_to_action(right_rel))
                    right_valid.append(True)
                else:
                    right_actions.append(None)
                    right_valid.append(False)

            record = {
                "start_frame_index": frame_t.get("frame_index"),
                "horizon": args.horizon,
                "stride": args.stride,
                "left_actions": left_actions,
                "right_actions": right_actions,
                "base_actions": base_actions,
                "left_valid": left_valid,
                "right_valid": right_valid,
            }
            out_f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
