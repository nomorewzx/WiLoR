from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import datasets
import shutil
import subprocess

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "pyarrow is required for conversion. Install with: pip install pyarrow"
    ) from exc


CODEBASE_VERSION = "v2.1"
DEFAULT_CHUNK_SIZE = 1000

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    outdict: dict = {}
    for key, value in d.items():
        parts = key.split(sep)
        cur = outdict
        for part in parts[:-1]:
            if part not in cur:
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = value
    return outdict


def serialize_dict(stats: dict[str, np.ndarray | dict]) -> dict:
    serialized = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, np.generic):
            serialized[key] = value.item()
        elif isinstance(value, (int, float)):
            serialized[key] = value
        else:
            raise NotImplementedError(f"Unsupported stats value type: {type(value)}")
    return unflatten_dict(serialized)


def get_hf_features_from_features(features: dict) -> datasets.Features:
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        if ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(
                length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
            )
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 3:
            hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 4:
            hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 5:
            hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
        else:
            raise ValueError(f"Invalid feature shape for {key}: {ft}")
    return datasets.Features(hf_features)


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


def compose_pose(base_rot: np.ndarray, base_t: np.ndarray, local_rot: np.ndarray, local_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    world_rot = base_rot @ local_rot
    world_t = base_rot @ local_t + base_t
    return world_rot, world_t


def load_frames(path: Path) -> List[Dict]:
    frames = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            frames.append(json.loads(line))
    return frames


def pick_hand_pose(hands: List[Dict], is_right: float) -> Optional[Dict]:
    for hand in hands:
        if float(hand.get("is_right", -1.0)) == is_right:
            return hand
    return None


def compute_world_pose_from_cam(hand_pose: Dict, cam_pose: Dict) -> Optional[Dict]:
    if not hand_pose or not cam_pose:
        return None
    local_rot = np.array(hand_pose["rotation_matrix"], dtype=np.float64)
    local_t = np.array(hand_pose["position"], dtype=np.float64)
    base_rot = quat_to_rot([cam_pose["qx"], cam_pose["qy"], cam_pose["qz"], cam_pose["qw"]])
    base_t = np.array([cam_pose["tx"], cam_pose["ty"], cam_pose["tz"]], dtype=np.float64)
    world_rot, world_t = compose_pose(base_rot, base_t, local_rot, local_t)
    return {"position": world_t.tolist(), "rotation_matrix": world_rot.tolist()}


def pose_to_vec(pose: Optional[Dict]) -> np.ndarray:
    if pose is None:
        return np.zeros(6, dtype=np.float64)
    pos = np.array(pose["position"], dtype=np.float64)
    rot = np.array(pose["rotation_matrix"], dtype=np.float64)
    aa = rot_to_axis_angle(rot)
    return np.concatenate([pos, aa], axis=0)


def frame_to_state(frame: Dict) -> Tuple[np.ndarray, bool, bool]:
    hands = frame.get("hands", [])
    left_hand = pick_hand_pose(hands, 0.0)
    right_hand = pick_hand_pose(hands, 1.0)

    left_pose = None
    right_pose = None
    if left_hand:
        left_pose = left_hand.get("end_effector_pose_world") or left_hand.get("end_effector_pose_cam")
        if left_pose is left_hand.get("end_effector_pose_cam"):
            left_pose = compute_world_pose_from_cam(left_pose, frame.get("camera_pose"))
    if right_hand:
        right_pose = right_hand.get("end_effector_pose_world") or right_hand.get("end_effector_pose_cam")
        if right_pose is right_hand.get("end_effector_pose_cam"):
            right_pose = compute_world_pose_from_cam(right_pose, frame.get("camera_pose"))

    left_valid = left_pose is not None
    right_valid = right_pose is not None

    base_pose = frame.get("base_pose_se3")
    if base_pose is None and frame.get("camera_pose"):
        cam = frame["camera_pose"]
        base_rot = quat_to_rot([cam["qx"], cam["qy"], cam["qz"], cam["qw"]])
        base_pose = {"position": [cam["tx"], cam["ty"], cam["tz"]], "rotation_matrix": base_rot.tolist()}
    if base_pose is None:
        base_pose = {"position": [0.0, 0.0, 0.0], "rotation_matrix": np.eye(3).tolist()}

    left_vec = pose_to_vec(left_pose)
    right_vec = pose_to_vec(right_pose)
    base_vec = pose_to_vec(base_pose)
    state = np.concatenate([left_vec, right_vec, base_vec]).astype(np.float32)
    return state, left_valid, right_valid


def load_segments(path: Optional[Path]) -> List[Dict]:
    if not path:
        return [{"start_time": 0.0, "end_time": None, "task": ""}]
    with path.open("r", encoding="utf-8") as f:
        segments = json.load(f)
    if not isinstance(segments, list):
        raise ValueError("segments JSON must be a list of segments")
    return segments


def build_time_index(times: List[float], target_times: List[float]) -> List[int]:
    indices: List[int] = []
    for t in target_times:
        i = np.searchsorted(times, t, side="left")
        if i <= 0:
            idx = 0
        elif i >= len(times):
            idx = len(times) - 1
        else:
            before = times[i - 1]
            after = times[i]
            idx = i - 1 if (t - before) <= (after - t) else i
        indices.append(int(idx))
    return indices


def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10000, power: float = 0.75
) -> int:
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300) -> np.ndarray:
    _, height, width = img.shape
    if max(width, height) < max_size_threshold:
        return img
    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def compute_video_stats(video_path: Path, num_frames: int) -> dict[str, np.ndarray]:
    num_samples = estimate_num_samples(num_frames)
    sample_indices = np.round(np.linspace(0, num_frames - 1, num_samples)).astype(int).tolist()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for stats: {video_path}")

    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = np.transpose(frame_rgb, (2, 0, 1)).astype(np.uint8)
        img = auto_downsample_height_width(img)
        frames.append(img)

    cap.release()
    images = np.stack(frames, axis=0)
    stats = get_feature_stats(images, axis=(0, 2, 3), keepdims=True)
    stats = {k: (v if k == "count" else np.squeeze(v / 255.0, axis=0)) for k, v in stats.items()}
    return stats


def compute_episode_stats(
    episode_data: dict[str, np.ndarray], features: dict[str, dict], video_paths: dict[str, Path]
) -> dict:
    stats: dict[str, dict] = {}
    for key, ft in features.items():
        if ft["dtype"] == "string":
            continue
        if ft["dtype"] in ["image", "video"]:
            stats[key] = compute_video_stats(video_paths[key], len(episode_data["frame_index"]))
            continue

        data = episode_data[key]
        if data.dtype == np.bool_:
            data = data.astype(np.float32)
        axes = 0
        keepdims = data.ndim == 1
        stats[key] = get_feature_stats(data, axis=axes, keepdims=keepdims)
    return stats


def ensure_empty_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def reencode_h264(src_path: Path, ffmpeg_path: str) -> None:
    tmp_path = src_path.with_suffix(".tmp.mp4")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(src_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with code {result.returncode}: {result.stderr.strip()}"
        )
    tmp_path.replace(src_path)


def build_features(height: int, width: int, camera_key: str) -> dict[str, dict]:
    features = {**DEFAULT_FEATURES}
    state_names = [
        "L_tx",
        "L_ty",
        "L_tz",
        "L_rx",
        "L_ry",
        "L_rz",
        "R_tx",
        "R_ty",
        "R_tz",
        "R_rx",
        "R_ry",
        "R_rz",
        "B_tx",
        "B_ty",
        "B_tz",
        "B_rx",
        "B_ry",
        "B_rz",
    ]
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(state_names),),
        "names": state_names,
    }
    features["observation.valid.left"] = {"dtype": "bool", "shape": (1,), "names": None}
    features["observation.valid.right"] = {"dtype": "bool", "shape": (1,), "names": None}
    features[f"observation.images.{camera_key}"] = {
        "dtype": "video",
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }
    return features


def features_to_json(features: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for key, ft in features.items():
        out[key] = {
            "dtype": ft["dtype"],
            "shape": list(ft["shape"]),
            "names": ft["names"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert hand_poses.jsonl to LeRobot v2.1 dataset.")
    parser.add_argument("--input", required=True, help="Path to hand_poses.jsonl.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--output", required=True, help="Output dataset folder.")
    parser.add_argument("--segments", default=None, help="Optional JSON file defining segments.")
    parser.add_argument("--fps", type=float, default=0.0, help="Target FPS for resampling (0 uses source).")
    parser.add_argument("--camera_key", default="ego", help="Camera key for observation.images.")
    parser.add_argument("--progress_interval", type=int, default=200, help="Print progress every N frames.")
    parser.add_argument(
        "--video_codec",
        default="h264",
        choices=["h264", "mp4v"],
        help="Video codec for output mp4 (h264 recommended for browser playback).",
    )
    parser.add_argument(
        "--ffmpeg_path",
        default=None,
        help="Optional path to ffmpeg binary (needed for h264).",
    )
    args = parser.parse_args()

    frames = load_frames(Path(args.input))
    if not frames:
        raise RuntimeError("No frames found in input JSONL.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fps = args.fps if args.fps > 0 else src_fps
    if fps <= 0:
        raise RuntimeError("Unable to infer FPS from video; specify --fps.")

    times = []
    for frame in frames:
        if frame.get("time_ms") is not None:
            times.append(float(frame["time_ms"]) / 1000.0)
        else:
            frame_index = int(frame.get("frame_index", len(times)))
            times.append(frame_index / fps)

    segments = load_segments(Path(args.segments) if args.segments else None)
    out_root = Path(args.output)
    ensure_empty_dir(out_root)

    meta_dir = out_root / "meta"
    data_root = out_root / "data"
    videos_root = out_root / "videos"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    videos_root.mkdir(parents=True, exist_ok=True)

    features = build_features(height, width, args.camera_key)
    task_to_index: dict[str, int] = {}
    tasks_entries: List[Dict[str, Any]] = []
    episodes_entries: List[Dict[str, Any]] = []
    episodes_stats_entries: List[Dict[str, Any]] = []

    global_index = 0
    total_frames = 0
    for episode_index, segment in enumerate(segments):
        start = float(segment.get("start_time", 0.0))
        end = segment.get("end_time")
        if end is None:
            end = max(times)
        end = float(end)
        if end <= start:
            continue

        task = str(segment.get("task", ""))
        if task not in task_to_index:
            task_to_index[task] = len(task_to_index)
            tasks_entries.append({"task_index": task_to_index[task], "task": task})
        task_index = task_to_index[task]

        target_times = list(np.arange(start, end, 1.0 / fps))
        if not target_times:
            continue
        src_indices = build_time_index(times, target_times)

        chunk = episode_index // DEFAULT_CHUNK_SIZE
        data_dir = data_root / f"chunk-{chunk:03d}"
        video_dir = videos_root / f"chunk-{chunk:03d}" / f"observation.images.{args.camera_key}"
        data_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        video_path = video_dir / f"episode_{episode_index:06d}.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {video_path}")

        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {args.video}")

        states: List[List[float]] = []
        left_valids: List[bool] = []
        right_valids: List[bool] = []
        timestamps: List[float] = []
        frame_indices: List[int] = []
        episode_indices: List[int] = []
        global_indices: List[int] = []
        task_indices: List[int] = []

        for frame_offset, src_idx in enumerate(src_indices):
            frame = frames[src_idx]
            src_frame_index = int(frame.get("frame_index", src_idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_index)
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {src_frame_index} from {args.video}")
            writer.write(img)

            state, left_valid, right_valid = frame_to_state(frame)
            states.append(state.tolist())
            left_valids.append(left_valid)
            right_valids.append(right_valid)

            timestamp = frame_offset / fps
            timestamps.append(float(timestamp))
            frame_indices.append(frame_offset)
            episode_indices.append(episode_index)
            global_indices.append(global_index)
            task_indices.append(task_index)
            global_index += 1
            if args.progress_interval > 0 and (frame_offset + 1) % args.progress_interval == 0:
                print(f"Episode {episode_index}: wrote {frame_offset + 1}/{len(src_indices)} frames")

        cap.release()
        writer.release()
        if args.video_codec == "h264":
            ffmpeg_path = args.ffmpeg_path or shutil.which("ffmpeg")
            if not ffmpeg_path:
                raise RuntimeError("ffmpeg not found; specify --ffmpeg_path or use --video_codec mp4v.")
            reencode_h264(video_path, ffmpeg_path)

        if not states:
            continue

        episode_data = {
            "observation.state": np.array(states, dtype=np.float32),
            "observation.valid.left": np.array(left_valids, dtype=np.bool_),
            "observation.valid.right": np.array(right_valids, dtype=np.bool_),
            "timestamp": np.array(timestamps, dtype=np.float32),
            "frame_index": np.array(frame_indices, dtype=np.int64),
            "episode_index": np.array(episode_indices, dtype=np.int64),
            "index": np.array(global_indices, dtype=np.int64),
            "task_index": np.array(task_indices, dtype=np.int64),
        }
        hf_features = get_hf_features_from_features(features)
        episode_dict = {key: episode_data[key] for key in hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=hf_features, split="train")
        parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
        ep_dataset.to_parquet(parquet_path)
        video_paths = {f"observation.images.{args.camera_key}": video_path}
        episode_stats = compute_episode_stats(episode_data, features, video_paths)
        episodes_stats_entries.append(
            {"episode_index": episode_index, "stats": serialize_dict(episode_stats)}
        )

        episodes_entries.append(
            {
                "episode_index": episode_index,
                "tasks": [task],
                "length": len(states),
            }
        )
        total_frames += len(states)
        print(f"Episode {episode_index} done: {len(states)} frames")

    total_episodes = len(episodes_entries)
    total_tasks = len(tasks_entries)
    total_chunks = (total_episodes + DEFAULT_CHUNK_SIZE - 1) // DEFAULT_CHUNK_SIZE if total_episodes else 0

    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": None,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(video_keys),
        "total_chunks": total_chunks,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features_to_json(features),
    }

    with (meta_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for entry in tasks_entries:
            f.write(json.dumps(entry) + "\n")

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for entry in episodes_entries:
            f.write(json.dumps(entry) + "\n")

    with (meta_dir / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for entry in episodes_stats_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Conversion complete: {total_episodes} episodes, {total_frames} frames")


if __name__ == "__main__":
    main()
