from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import bisect

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.utils.renderer import cam_crop_to_full


def load_intrinsics(path: Optional[str], video_size: Tuple[int, int]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        intr = json.load(f)

    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    ref_w = float(intr["resolution"]["width"])
    ref_h = float(intr["resolution"]["height"])
    vid_w, vid_h = video_size

    # Scale intrinsics if video resolution differs from calibration resolution.
    if ref_w > 0 and ref_h > 0 and (ref_w != vid_w or ref_h != vid_h):
        sx = vid_w / ref_w
        sy = vid_h / ref_h
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy

    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def load_poses(path: Optional[str]) -> Optional[List[Dict[str, float]]]:
    if not path:
        return None
    poses: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append(
                {
                    "timestamp": float(row["timestamp"]),
                    "tx": float(row["tx"]),
                    "ty": float(row["ty"]),
                    "tz": float(row["tz"]),
                    "qx": float(row["qx"]),
                    "qy": float(row["qy"]),
                    "qz": float(row["qz"]),
                    "qw": float(row["qw"]),
                }
            )
    return poses


def project_points(points_cam: np.ndarray, intr: Dict[str, float]) -> np.ndarray:
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    z = np.where(z == 0, 1e-6, z)
    u = intr["fx"] * (x / z) + intr["cx"]
    v = intr["fy"] * (y / z) + intr["cy"]
    return np.stack([u, v], axis=-1)


def select_render_indices(total_frames: int, count: int) -> List[int]:
    if count <= 0 or total_frames <= 0:
        return []
    if count == 1:
        return [0]
    idx = np.linspace(0, total_frames - 1, num=count)
    idx = np.round(idx).astype(int)
    # Ensure strictly increasing, unique indices.
    out = []
    seen = set()
    for i in idx:
        if i not in seen:
            out.append(int(i))
            seen.add(int(i))
    return out


def compute_hand_axes(joints_cam: np.ndarray) -> Optional[np.ndarray]:
    end_eff = compute_end_effector_pose(joints_cam)
    if end_eff is None:
        return None
    rot = np.array(end_eff["rotation_matrix"], dtype=np.float64)
    return np.stack([rot[:, 0], rot[:, 1], rot[:, 2]], axis=0)


def compute_end_effector_pose(joints_cam: np.ndarray) -> Optional[Dict[str, List[float]]]:
    if joints_cam.shape[0] < 18:
        return None
    # Palm center from MCP joints (index, middle, ring, pinky).
    palm = (joints_cam[5] + joints_cam[9] + joints_cam[13] + joints_cam[17]) / 4.0
    middle_mcp = joints_cam[9]
    ring_mcp = joints_cam[13]

    x_axis = middle_mcp - palm
    y_axis = ring_mcp - palm
    x_norm = np.linalg.norm(x_axis)
    y_norm = np.linalg.norm(y_axis)
    if x_norm < 1e-6 or y_norm < 1e-6:
        return None
    x_axis = x_axis / x_norm
    y_axis = y_axis / y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        return None
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)

    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    return {"position": palm.tolist(), "rotation_matrix": rot.tolist()}


def draw_axes(
    img_bgr: np.ndarray,
    origin_cam: np.ndarray,
    axes: np.ndarray,
    intr: Dict[str, float],
    length: float = 0.05,
) -> None:
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    origin_2d = project_points(origin_cam[None, :], intr)[0]
    for i in range(3):
        end_cam = origin_cam + axes[i] * length
        end_2d = project_points(end_cam[None, :], intr)[0]
        p0 = (int(round(origin_2d[0])), int(round(origin_2d[1])))
        p1 = (int(round(end_2d[0])), int(round(end_2d[1])))
        cv2.line(img_bgr, p0, p1, colors[i], 2, lineType=cv2.LINE_AA)


def scale_by_bone_length(
    joints: np.ndarray,
    verts: np.ndarray,
    cam_t: np.ndarray,
    joint_a: int,
    joint_b: int,
    target_length_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if joint_a < 0 or joint_b < 0 or joint_a >= joints.shape[0] or joint_b >= joints.shape[0]:
        return joints, verts, cam_t, 1.0
    cur = np.linalg.norm(joints[joint_a] - joints[joint_b])
    if cur < 1e-6:
        return joints, verts, cam_t, 1.0
    scale = target_length_m / cur
    return joints * scale, verts * scale, cam_t * scale, scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WiLoR on a video with ARKit intrinsics/poses.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--out_folder", required=True, help="Output folder to save results.")
    parser.add_argument("--intrinsics", default=None, help="Path to intrinsics.json.")
    parser.add_argument("--poses", default=None, help="Path to poses.csv (one row per frame).")
    parser.add_argument("--save_vertices", action="store_true", help="Include vertices in output.")
    parser.add_argument("--max_frames", type=int, default=-1, help="Optional cap on frames processed.")
    parser.add_argument("--rescale_factor", type=float, default=2.0, help="Factor for padding the bbox.")
    parser.add_argument("--conf", type=float, default=0.3, help="Detector confidence threshold.")
    parser.add_argument("--render_samples", type=int, default=0, help="Render N evenly spaced frames.")
    parser.add_argument("--render_folder", default=None, help="Folder for rendered JPGs.")
    parser.add_argument(
        "--render_every",
        type=int,
        default=0,
        help="Render every Nth frame (overrides render_samples if > 0).",
    )
    parser.add_argument(
        "--scale_bone_length_cm",
        type=float,
        default=0.0,
        help="Scale hand so joint[0]-joint[9] length equals this value (cm).",
    )
    parser.add_argument(
        "--axis_length_cm",
        type=float,
        default=5.0,
        help="Axis length for rendering (cm).",
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=30,
        help="Print progress every N frames (0 disables).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    # Load model and detector
    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor_final.ckpt",
        cfg_path="./pretrained_models/model_config.yaml",
    )
    detector = YOLO("./pretrained_models/detector.pt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    detector = detector.to(device)
    model.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    intrinsics = load_intrinsics(args.intrinsics, (vid_w, vid_h))
    poses = load_poses(args.poses)
    pose_times = None
    if poses:
        t0 = poses[0]["timestamp"]
        pose_times = [p["timestamp"] - t0 for p in poses]

    out_path = Path(args.out_folder) / "hand_poses.jsonl"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.render_every > 0:
        render_indices = list(range(0, total_frames, args.render_every))
    else:
        render_indices = select_render_indices(total_frames, args.render_samples)
    render_set = set(render_indices)
    render_folder = Path(args.render_folder) if args.render_folder else Path(args.out_folder) / "renders"
    if render_indices:
        render_folder.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        while True:
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break

            detections = detector(frame_bgr, conf=args.conf, verbose=False)[0]
            bboxes = []
            is_right = []
            for det in detections:
                bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
                bboxes.append(bbox[:4].tolist())

            hands_out = []
            render_img = frame_bgr.copy() if frame_idx in render_set else None
            if len(bboxes) > 0:
                boxes = np.stack(bboxes)
                right = np.stack(is_right)
                dataset = ViTDetDataset(model_cfg, frame_bgr, boxes, right, rescale_factor=args.rescale_factor)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

                for batch in dataloader:
                    batch = recursive_to(batch, device)
                    with torch.no_grad():
                        out = model(batch)

                    multiplier = (2 * batch["right"] - 1)
                    pred_cam = out["pred_cam"]
                    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                    box_center = batch["box_center"].float()
                    box_size = batch["box_size"].float()
                    img_size = batch["img_size"].float()

                    # Keep WiLoR's default focal length unless intrinsics are provided.
                    if intrinsics:
                        scaled_focal_length = float((intrinsics["fx"] + intrinsics["fy"]) * 0.5)
                    else:
                        scaled_focal_length = (
                            model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                        )
                    pred_cam_t_full = cam_crop_to_full(
                        pred_cam, box_center, box_size, img_size, scaled_focal_length
                    ).detach().cpu().numpy()

                    batch_size = batch["img"].shape[0]
                    for n in range(batch_size):
                        joints = out["pred_keypoints_3d"][n].detach().cpu().numpy()
                        verts = out["pred_vertices"][n].detach().cpu().numpy()

                        is_right_n = float(batch["right"][n].cpu().numpy())
                        joints[:, 0] = (2 * is_right_n - 1) * joints[:, 0]
                        verts[:, 0] = (2 * is_right_n - 1) * verts[:, 0]

                        cam_t = pred_cam_t_full[n]
                        scale = 1.0
                        if args.scale_bone_length_cm > 0:
                            target_m = args.scale_bone_length_cm / 100.0
                            joints, verts, cam_t, scale = scale_by_bone_length(
                                joints, verts, cam_t, 0, 9, target_m
                            )
                        joints_cam = joints + cam_t
                        verts_cam = verts + cam_t

                        if intrinsics:
                            kpts_2d = project_points(joints_cam, intrinsics)
                        else:
                            kpts_2d = None

                        hand_entry = {
                            "is_right": is_right_n,
                            "cam_t": cam_t.tolist(),
                            "joints_3d_cam": joints_cam.tolist(),
                            "scale": scale,
                        }
                        if args.save_vertices:
                            hand_entry["verts_3d_cam"] = verts_cam.tolist()
                        if kpts_2d is not None:
                            hand_entry["keypoints_2d"] = kpts_2d.tolist()
                        end_eff = compute_end_effector_pose(joints_cam)
                        if end_eff is not None:
                            hand_entry["end_effector_pose"] = end_eff
                        hands_out.append(hand_entry)
                        if render_img is not None and intrinsics:
                            axes = compute_hand_axes(joints_cam)
                            if axes is not None:
                                end_eff = compute_end_effector_pose(joints_cam)
                                if end_eff is not None:
                                    origin = np.array(end_eff["position"], dtype=np.float64)
                                    draw_axes(
                                        render_img,
                                        origin,
                                        axes,
                                        intrinsics,
                                        length=args.axis_length_cm / 100.0,
                                    )

                if render_img is not None:
                    for bbox in bboxes:
                        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                        cv2.rectangle(render_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

            frame_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            pose_row = None
            pose_idx = None
            if poses and pose_times:
                t_video = frame_time_ms / 1000.0
                i = bisect.bisect_left(pose_times, t_video)
                if i <= 0:
                    pose_idx = 0
                elif i >= len(pose_times):
                    pose_idx = len(pose_times) - 1
                else:
                    before = pose_times[i - 1]
                    after = pose_times[i]
                    pose_idx = i - 1 if (t_video - before) <= (after - t_video) else i
                pose_row = poses[pose_idx]
            record = {
                "frame_index": frame_idx,
                "time_ms": frame_time_ms,
                "camera_pose_index": pose_idx,
                "camera_pose": pose_row,
                "hands": hands_out,
            }
            out_f.write(json.dumps(record) + "\n")

            if render_img is not None:
                out_img = render_folder / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_img), render_img)
            frame_idx += 1
            if args.progress_interval > 0 and frame_idx % args.progress_interval == 0:
                total = total_frames if total_frames > 0 else "?"
                print(f"Processed {frame_idx}/{total} frames")

    cap.release()


if __name__ == "__main__":
    main()
