# Overview

OctMem-Agent is an advanced, memory-augmented Aerial VLN framework designed for autonomous Unmanned Aerial Vehicles (UAVs) performing object-goal navigation.

# OCT Agent Data Collection

This section describes how to collect AirSim point-cloud and trajectory data. For dataset setup, refer to [iLearn-Lab/ACMMM25-UAV_ON](https://github.com/iLearn-Lab/ACMMM25-UAV_ON).

## Collect Point Cloud

First generate the scene point cloud from AirSim depth observations:

```bash
python tools/pcd_gen/airsim_pointcloud.py --env BrushifyUrban
```

Output:

```text
tools/pcd_gen_data/<ENV_NAME>_point_map_finegrain_v2/depth_final.ply
```

Then copy or link it to the path used by trajectory planning:

```text
tools/scene_data/pcd_map/<ENV_NAME>.ply
```

If you already have a `.pcd` map, convert it to `.ply`:

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

Record RGB images from the planned trajectories:

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
