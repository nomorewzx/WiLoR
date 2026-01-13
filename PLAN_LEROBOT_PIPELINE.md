# Plan: Build and Verify JSONL -> LeRobot v2.1 (Delta Actions)

Goal: Convert WiLoR JSONL (absolute hand/base poses) into a LeRobot v2.1 dataset with per-frame delta actions suitable for openpi training, and verify correctness before training.

## A. Define Data Contract
1) Coordinate frames:
   - Camera frame C_t (per frame): +X right, +Y down, +Z forward; right-handed.
   - Head frame e_t is defined as the camera frame C_t (paper convention).
   - World frame W: ARKit world frame as provided by `poses.csv`.
2) Pose definitions:
   - `end_effector_pose_cam`: SE(3) of hand in C_t (position + rotation matrix).
   - `camera_pose`: pose of camera in W (tx, ty, tz, qx, qy, qz, qw).
   - `end_effector_pose_world`: SE(3) of hand in W, computed as T_w_hand = T_w_cam * T_cam_hand.
   - `base_pose_se3`: SE(3) of camera/base in W (position + rotation).
3) Units and handedness:
   - Positions in meters, rotations in radians, right-handed frames.
4) Action indexing rule:
   - Store absolute poses in dataset; action chunks are built at training time.
   - Chunk actions are relative to the chunk start state `s0`: `a_i = T_{s0}^{-1} * T_{s_i}`.
   - For base actions: use full SE(3) relative pose `T_{s0}^{-1} * T_{s_i}` (not just yaw/xy).
5) Rotation delta representation:
   - Use axis-angle from relative rotation R_rel = R_{s0}^T R_{s_i}.
   - Clamp numeric trace and re-orthonormalize if needed.
6) Resampling semantics:
   - Resample absolute poses to target FPS first, then compute deltas.
7) State/action schema:
   - State dim = 18: [L_tx, L_ty, L_tz, L_rx, L_ry, L_rz, R_tx, R_ty, R_tz, R_rx, R_ry, R_rz, B_tx, B_ty, B_tz, B_rx, B_ry, B_rz] in W.
   - Action dim = 18 for chunked training: [L_tx, L_ty, L_tz, L_rx, L_ry, L_rz, R_tx, R_ty, R_tz, R_rx, R_ry, R_rz, B_tx, B_ty, B_tz, B_rx, B_ry, B_rz],
     where each action is relative to `s0` in the chunk.
   - Validity masks stored separately: `observation.valid.left`, `observation.valid.right` (bool).
   - Deltas are computed from `end_effector_pose_world` (not camera-frame poses).

## B. Build the Converter (JSONL -> LeRobot v2.1)
1) Input: `hand_poses.jsonl` with `end_effector_pose_cam` and `camera_pose` plus `base_pose_se3`.
2) Segmenting: apply user-defined start/end times and optional language prompts per segment (stored as `task`).
3) For each frame in a segment:
   - Image: extract frame from video and save into LeRobot dataset.
   - State: absolute pose vector at time t in W.
   - Actions: not stored; built at training time from chunk-relative transforms.
4) Metadata: store FPS, task/prompt, episode boundaries.
5) Output: LeRobot v2.1 format (per-episode parquet + mp4).

## C. Verification Checklist (No Training Needed)
1) Round-trip test:
   - Absolute pose at s0 + chunk-relative delta reconstructs pose at s_i.
2) Magnitude sanity:
   - Translation norms and rotation angles fall in expected ranges; no spikes at segment boundaries.
3) FPS consistency:
   - Downsampled FPS produces proportionally larger deltas.
4) Visual sanity:
   - Integrate deltas over 1-2 seconds and overlay reconstructed pose on video frames.
5) Dataset load test:
   - Load LeRobot dataset with openpi DataLoader and confirm shapes (`state`, `actions`, images).

## D. Decide on Action Representation (Delta vs Absolute)
- Default: chunk-relative actions for training, absolute poses for state.
- Keep absolute JSONL as source of truth; regenerate chunk deltas if needed.

## E. Deliverables
1) Update `run_video_arkit.py` to compute and store `end_effector_pose_world` and `base_pose_se3`.
2) `convert_jsonl_to_lerobot_v21.py` (conversion script).
3) `verify_deltas.py` (round-trip + stats checks).
4) Sample dataset in LeRobot v2.1 format and a short verification report.

## F. LeRobot v2.1 Dataset Spec (Summary)
- Dataset stored as a Hugging Face dataset (Arrow/Parquet) + mp4 videos + json/jsonl metadata.
- One episode per file (v2.1): each episode has its own Parquet row range and video file(s).
- Core HF dataset features typically include:
  - `observation.images.<cam_key>` as `VideoFrame` entries: `{path: <mp4>, timestamp: float32}`
  - `observation.state` as float32 vector
  - `observation.valid.left` / `observation.valid.right` as bool
  - `action` is not stored; chunk actions are built at training time by a custom loader.
  - `task` as string (language prompt)
  - `episode_index` (int64), `frame_index` (int64), `timestamp` (float32), `next.done` (bool), `index` (int64)
- Dataset metadata (`info`) includes:
  - `codebase_version`
  - `fps`
  - `video` (bool) and encoding options
  - `camera_keys` list for image feature keys
- `stats` include per‑feature mean/std/min/max (used for normalization).
- LeRobot supports time‑windowing with `delta_timestamps` to fetch multiple frames relative to a given index.
- LeRobot v2.1 is required by openpi; v3.0 packs multiple episodes per file and is not supported by openpi yet.
