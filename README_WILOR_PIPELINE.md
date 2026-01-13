# WiLoR ARKit Video Pipeline

This document describes the local scripts used to process an ARKit-recorded egocentric video into per-frame hand poses and (optionally) action chunks for downstream training.

## Purpose
- Extract 3D hand poses in the iPhone camera frame from a video.
- Align each frame with ARKit camera poses by timestamp.
- Optionally rescale hand geometry to a known bone length.
- Optionally render sample frames with hand axes.
- Build action chunks (relative transforms) for hands and base motion.

## Inputs
- `video.mp4`: RGB video recorded by iPhone.
- `intrinsics.json`: Camera intrinsics (fx, fy, cx, cy) and resolution.
- `poses.csv`: ARKit camera poses per frame with timestamps.

Example input folder:
`/home/zxwang/repos/ego_centric_video_processing/resources/recording_3`

## Script 1: Per-frame hand poses
File: `run_video_arkit.py`

Key outputs:
- `hand_poses.jsonl`: one JSON record per video frame.
  - `camera_pose`: ARKit pose aligned by timestamp.
  - `base_pose_se3`: camera/base pose in world frame (position + rotation matrix).
  - `hands`: list of detected hands with:
    - `joints_3d_cam`: 21 MANO joints in camera frame.
    - `end_effector_pose_cam`: palm-based 6-DoF pose in camera frame.
    - `end_effector_pose_world`: palm-based 6-DoF pose in world frame (if poses.csv provided).
    - `end_effector_pose`: alias for `end_effector_pose_cam` (compat).
    - `keypoints_2d`: projected joints in image pixels (if intrinsics provided).
    - `scale`: applied scale factor (if enabled).
- `renders/`: optional rendered JPG frames with boxes and axes.

Run example:
```bash
python run_video_arkit.py \
  --video /home/zxwang/repos/ego_centric_video_processing/resources/recording_3/video.mp4 \
  --intrinsics /home/zxwang/repos/ego_centric_video_processing/resources/recording_3/intrinsics.json \
  --poses /home/zxwang/repos/ego_centric_video_processing/resources/recording_3/poses.csv \
  --out_folder /home/zxwang/repos/ego_centric_video_processing/resources/recording_3/wilor_out \
  --scale_bone_length_cm 10 \
  --render_every 5
```

Notes:
- Pose alignment is **timestamp-based** (nearest neighbor).
- Hand pose scale is arbitrary unless `--scale_bone_length_cm` is set.
- Axes are drawn from the end-effector pose (palm + middle/ring).

## Script 2: Action chunks
File: `make_action_chunks.py`

This script builds action chunks for:
- Left hand end-effector (relative to chunk start).
- Right hand end-effector (relative to chunk start).
- Base actions from ARKit camera pose (relative to chunk start).

Output format:
- `action_chunks.jsonl`, one record per chunk:
  - `left_actions`, `right_actions`, `base_actions`:
    - Each element is a 6D action `[tx, ty, tz, rx, ry, rz]`.
    - Rotation is axis-angle (radians).
  - `left_valid`, `right_valid`: per-step masks.

Run example:
```bash
python make_action_chunks.py \
  --input /home/zxwang/repos/ego_centric_video_processing/resources/recording_3/wilor_out/hand_poses.jsonl \
  --output /home/zxwang/repos/ego_centric_video_processing/resources/recording_3/wilor_out/action_chunks.jsonl \
  --horizon 10 \
  --stride 1
```

## Script 3: LeRobot v2.1 conversion
File: `convert_jsonl_to_lerobot_v21.py`

This converts `hand_poses.jsonl` + the source video into a LeRobot v2.1 dataset
(per-episode parquet + mp4) with absolute state vectors and validity masks.
By default, videos are re-encoded to H.264 for browser playback (requires `ffmpeg`).

Example:
```bash
python convert_jsonl_to_lerobot_v21.py \
  --input /home/zxwang/repos/ego_centric_video_processing/resources/recording_3/wilor_out/hand_poses.jsonl \
  --video /home/zxwang/repos/ego_centric_video_processing/resources/recording_3/video.mp4 \
  --output /home/zxwang/repos/ego_centric_video_processing/resources/recording_3/lerobot_out \
  --fps 30
```

Notes:
- Use `--video_codec h264` (default) for browser-compatible mp4 output.
- If `ffmpeg` is not on PATH, pass `--ffmpeg_path /path/to/ffmpeg`.

## Script 4: Delta verification
File: `verify_deltas.py`

This checks round-trip reconstruction from chunk-relative deltas and reports
translation/rotation error stats.

Example:
```bash
python verify_deltas.py \
  --input /home/zxwang/repos/ego_centric_video_processing/resources/recording_3/wilor_out/hand_poses.jsonl \
  --horizon 10 \
  --stride 1
```

Interpreting results:
- `round_trip_translation` / `round_trip_rotation` should be near zero (numerical noise). If these are large, the
  transform math or frame alignment is wrong.
- `delta_translation_norm` and `delta_rotation_angle` report per-step motion magnitudes. Large spikes usually
  indicate missing detections or resampling jumps; check frame rate and segment boundaries if you see them.

## Visualize the LeRobot dataset
LeRobot includes an HTML viewer that can serve local datasets.

Example:
```bash
/home/zxwang/miniconda3/envs/lerobot/bin/python \
  /home/zxwang/repos/lerobot/src/lerobot/scripts/visualize_dataset_html.py \
  --repo-id local/lerobot_out_v21_h264 \
  --root /home/zxwang/repos/ego_centric_video_processing/resources/recording_3/lerobot_out_v21_h264 \
  --serve 1 \
  --host 127.0.0.1 \
  --port 9090
```

Then open `http://127.0.0.1:9090` in your browser.

## Dependencies
Follow the main `README.md` for WiLoR setup, pretrained models, and MANO model placement.
