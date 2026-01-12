  What openpi expects (from /home/zxwang/repos/openpi)

  - LeRobot dataset format: data must be converted into a LeRobot dataset with frames/episodes (see examples/libero/
    convert_libero_data_to_lerobot.py).
  - Per‑frame features: typically image(s), state, actions, and optionally task/prompt.
  - Action sequences: openpi builds action chunks using delta_timestamps in LeRobotDataset (see src/openpi/training/data_loader.py).
  - Action/state dimensions: must match model.action_dim. pi0.5 uses configurable action_dim; many configs are 14 or 8 depending on robot; you can
    set your own but must be consistent across dataset + config.
  - Normalization stats: you must compute norm_stats.json for the dataset (see docs/norm_stats.md).
  - Frame rate: dataset fps used to assemble action sequences. It must match your temporal sampling.

  What WiLoR pipeline outputs today

  - hand_poses.jsonl: per‑frame hand joints, end‑effector pose, camera pose (ARKit), optional 2D projections.
  - action_chunks.jsonl (optional): per‑chunk relative actions (axis‑angle + translation) for left/right/base.
  - No LeRobot dataset, no episodes, no images stored as dataset features.
  - No explicit state or actions fields in a training‑ready format.
  - No prompts/tasks.

  Critical gaps

  1. Dataset format
      - openpi requires LeRobot format.
      - You currently have JSONL outputs only.
  2. Observation images
      - openpi expects images stored as dataset features.
      - You currently reference a video path; not stored as image frames in dataset.
  3. State/action definition
      - openpi expects state and actions arrays with consistent dimension.
      - You currently have end‑effector poses and camera poses, but no canonical state/action vector definition.
  4. Action timing
      - openpi builds action chunks from per‑frame actions using dataset fps.
      - Your action_chunks.jsonl is already chunked, which doesn’t directly map to LeRobot’s per‑frame actions unless you write a custom loader.
  5. Prompt/task (optional)
      - openpi can use task or prompt. You currently ignore text annotations.

  ———

  ## Proposed solution (high‑level)

  A) Decide the action/state representation

  - Use the paper’s human action definition:
      - Left hand 6‑DoF
      - Right hand 6‑DoF
      - Base 6‑DoF
        → total 18 dims
  - Use per‑frame delta actions (relative transforms from t to t+1).
    This matches how pi0 models are trained (delta actions), so no extra transform needed.

  B) Define the per‑frame state

  - Option 1: same 18‑dim pose (absolute pose at time t).
  - Option 2: include more (e.g., both absolute hand pose + base pose).
  - For pi0.5, state and action must align with action_dim or get padded by transforms.

  C) Convert to LeRobot dataset

  - Extract frames from video.
  - Create LeRobot dataset with features:
      - image (RGB frame)
      - state (float32, shape = action_dim)
      - actions (float32, shape = action_dim)
      - task or prompt (optional)
  - Set fps to your actual frame rate (or downsample beforehand).

  D) Training config

  - Create a custom DataConfig (similar to LeRobotLiberoDataConfig) to map dataset keys to model inputs.
  - Set model.action_dim = 18 (or whatever you decide).
  - Compute norm_stats.json on your dataset.

  ———

  ## Key open decisions you should confirm

  1. Action definition: Is the 18‑dim vector [left6, right6, base6] what you want?
  2. Delta vs absolute: Should actions be deltas? (Recommended: deltas.)
  3. State definition: Do you want state == pose at t, or something else?
  4. Frame rate: Keep original FPS or downsample (e.g., 15 Hz)?
  5. Text prompts: Ignore for now or include later?

  If you confirm these, I can propose the exact LeRobot conversion script and the matching openpi DataConfig without touching your current
  pipeline.
