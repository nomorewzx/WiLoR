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
  - `hands`: list of detected hands with:
    - `joints_3d_cam`: 21 MANO joints in camera frame.
    - `end_effector_pose`: palm-based 6-DoF pose (position + rotation matrix).
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

## Dependencies
Follow the main `README.md` for WiLoR setup, pretrained models, and MANO model placement.

