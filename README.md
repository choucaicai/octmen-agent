# OCT Agent Data Collection

这个仓库主要用于基于 AirSim 场景收集 UAV 轨迹数据。核心流程是：

1. 准备场景配置和点云地图
2. 根据 start / goal 数据规划轨迹
3. 启动 AirSim 环境并沿轨迹采集 RGB / depth
4. 输出 JSON 轨迹标注和图像数据

## Project Structure

```text
.
├── dlimp/                    # tf.data / trajectory dataset utils
├── tools/
│   ├── configs/              # environment configs and AirSim settings
│   ├── pcd_gen/              # point-cloud generation / conversion tools
│   ├── traj_gen/             # path planning and trajectory recording
│   └── run_traj.sh           # example collection entry
└── README.md
```

## Requirements

需要提前准备：

- Python environment
- AirSim Python package
- AirSim scene executable
- `numpy`, `opencv-python`, `open3d`, `Pillow`, `PyYAML`, `scipy`, `tqdm`, `torch`
- GPU / display environment that can run the AirSim scene

当前代码里部分路径仍是本机绝对路径，例如：

```text
/home/zzz/code/UAV_ON/ENVS/TRAIN_ENVS
/home/zzz/code/UAV_ON/tools/configs
```

使用前需要根据你的机器修改：

- `tools/pcd_gen/run_airsim.py` 里的 `ENV_DICT`
- `tools/traj_gen/record_traj.py` 里的 `change_and_save_settings()`
- 如果使用点云采集脚本，也检查 `tools/pcd_gen/airsim_pointcloud.py` 里的 `change_and_save_settings()`

## Environment Config

每个场景对应一个 YAML：

```text
tools/configs/<ENV_NAME>.yaml
```

例如：

```text
tools/configs/BrushifyUrban.yaml
tools/configs/DownTown.yaml
tools/configs/UrbanJapan.yaml
```

关键字段：

- `datagen.env`: 场景名，需要和文件名保持一致
- `traj_map.MapBound`: 点云和轨迹规划范围
- `traj_map.MapDelta`: 点云采样步长
- `traj_map.VoxelWidth`: voxel map 分辨率
- `traj_map.DilateRadius`: 障碍膨胀半径
- `thread_params`: 多 AirSim 实例端口配置，`aim_port` 会写入 AirSim settings

## Data Inputs

轨迹规划需要两个输入。

### 1. Point Cloud Map

路径规划会读取：

```text
tools/scene_data/pcd_map/<ENV_NAME>.ply
```

如果只有 `.pcd`，可以转成 `.ply`：

```bash
cd tools
python pcd_gen/convert.py \
  --input scene_data/pcd_map/BrushifyUrban.pcd \
  --output scene_data/pcd_map/BrushifyUrban.ply
```

### 2. Start / Goal JSON

轨迹规划会读取：

```text
tools/data/dataset/<ENV_NAME>_train.json
```

每条数据需要包含类似字段：

```json
{
  "episode_id": "0",
  "start_pose": {
    "start_position": [0, 0, -10],
    "start_quaternionr": [0, 0, 0, 1]
  },
  "pose": [
    [100, 20, -10],
    [120, 50, -10]
  ]
}
```

其中 `pose` 是目标点列表，每个目标点会生成一条轨迹。

## Step 1: Plan Trajectories

从 `tools/` 目录运行：

```bash
cd tools
python traj_gen/collect_traj.py --env BrushifyUrban
```

输出默认写到：

```text
tools/output-v2/<ENV_NAME>/<episode_id>/<pose_idx>.json
tools/output-v2/<ENV_NAME>_result.json
```

每个轨迹 JSON 里包含：

- `record_list`: AirSim 采图时要走的 `[x, y, z, yaw]` 序列
- `action_list`: 轨迹动作
- `start_pos`
- `goal_pos`

## Step 2: Record Images

`record_traj.py` 会读取上一步生成的轨迹，启动多个 AirSim 实例，并沿轨迹采图。

```bash
cd tools
python traj_gen/record_traj.py \
  --env BrushifyUrban \
  --base_folder output-v2 \
  --output_folder record_output \
  --capture_depth False
```

如果需要 depth：

```bash
cd tools
python traj_gen/record_traj.py \
  --env BrushifyUrban \
  --base_folder output-v2 \
  --output_folder record_output_with_depth \
  --capture_depth True
```

输出结构：

```text
tools/record_output/
├── images/<ENV_NAME>/<episode_id>/<pose_idx>/uav_on_0/00000.png
└── json/<ENV_NAME>/<episode_id>/<pose_idx>.json
```

开启 depth 后会额外保存：

```text
tools/record_output_with_depth/images/<ENV_NAME>/<episode_id>/<pose_idx>/uav_on_0_depth/d_00000.npy
```

最终 JSON 会在原始轨迹信息基础上追加 `image_dict`，里面记录 RGB / depth 相对路径和 camera metainfo。

## Quick Run

可以直接改 `tools/run_traj.sh`：

```bash
cd tools
bash run_traj.sh
```

当前示例命令：

```bash
python traj_gen/record_traj.py --env ENV_NAME --base_folder output --output_folder output2
```

使用时把 `ENV_NAME` 改成真实场景名，例如 `BrushifyUrban`。

## Optional: Generate Point Cloud

如果需要从 AirSim 深度图生成点云，可使用：

```bash
python tools/pcd_gen/airsim_pointcloud.py --env BrushifyUrban
```

默认输出：

```text
tools/pcd_gen_data/<ENV_NAME>_point_map_finegrain_v2/depth_final.ply
```

生成后建议复制或链接到路径规划读取的位置：

```text
tools/scene_data/pcd_map/<ENV_NAME>.ply
```

## Notes

- `record_traj.py` 会根据 `thread_params` 启动多个 AirSim 实例，每个实例使用一个 `aim_port`。
- `AirSimRunner` 默认使用 4 张 GPU，并按 `p_i % total_gpus` 分配。
- 如果只想单进程调试，可以在 YAML 里只保留一个 `thread_params`。
- 已经采集过的 JSON 会被跳过，脚本支持断点继续。
- 运行失败时优先检查：AirSim executable 路径、`ApiServerPort`、`.ply` 是否存在、`<ENV_NAME>_train.json` 是否存在。
