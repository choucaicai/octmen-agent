# Overview

OctMem-Agent is an advanced, memory-augmented Aerial VLN framework designed for autonomous Unmanned Aerial Vehicles (UAVs) performing object-goal navigation.

# OCT Agent Data Collection

This section describes how to collect AirSim trajectory data.

## Prepare Data

Set the environment config:

```text
tools/configs/<ENV_NAME>.yaml
```

Required inputs:

```text
tools/scene_data/pcd_map/<ENV_NAME>.ply
tools/data/dataset/<ENV_NAME>_train.json
```

`<ENV_NAME>_train.json` should contain `episode_id`, `start_pose`, and target `pose` list. Each target pose generates one trajectory.

If the map is `.pcd`, convert it first:

```bash
cd tools
python pcd_gen/convert.py \
  --input scene_data/pcd_map/BrushifyUrban.pcd \
  --output scene_data/pcd_map/BrushifyUrban.ply
```

## Plan Trajectories

```bash
cd tools
python traj_gen/collect_traj.py --env BrushifyUrban
```

Output:

```text
tools/output-v2/<ENV_NAME>/<episode_id>/<pose_idx>.json
tools/output-v2/<ENV_NAME>_result.json
```

The trajectory JSON contains `record_list`, `action_list`, `start_pos`, and `goal_pos`.

## Record Images

```bash
cd tools
python traj_gen/record_traj.py \
  --env BrushifyUrban \
  --base_folder output-v2 \
  --output_folder record_output \
  --capture_depth False
```

For RGB + depth:

```bash
cd tools
python traj_gen/record_traj.py \
  --env BrushifyUrban \
  --base_folder output-v2 \
  --output_folder record_output_with_depth \
  --capture_depth True
```

Output:

```text
tools/record_output/
├── images/<ENV_NAME>/<episode_id>/<pose_idx>/uav_on_0/00000.png
└── json/<ENV_NAME>/<episode_id>/<pose_idx>.json
```

Depth output:

```text
tools/record_output_with_depth/images/<ENV_NAME>/<episode_id>/<pose_idx>/uav_on_0_depth/d_00000.npy
```

## Optional: Generate Point Cloud

```bash
python tools/pcd_gen/airsim_pointcloud.py --env BrushifyUrban
```

Output:

```text
tools/pcd_gen_data/<ENV_NAME>_point_map_finegrain_v2/depth_final.ply
```

Copy or link it to:

```text
tools/scene_data/pcd_map/<ENV_NAME>.ply
```

## Notes

- `thread_params` in `tools/configs/<ENV_NAME>.yaml` controls AirSim ports and parallel workers.
- `record_traj.py` skips already collected JSON files and supports resume.
- Update local AirSim executable paths in `tools/pcd_gen/run_airsim.py` before running.
