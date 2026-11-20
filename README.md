# Overview

OctMem-Agent is an advanced, memory-augmented Aerial VLN framework designed for autonomous Unmanned Aerial Vehicles (UAVs) performing object-goal navigation.

# Data Collection

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

## Build the Training JSON

Training reads a single `train.json`: a JSON array with one object per recorded
trajectory. Build it from the RGB + depth recording (training uses depth) by
converting every file under `tools/record_output_with_depth/json/<ENV_NAME>/` and save `train.json` next to `images/`:

```text
tools/record_output_with_depth/
├── images/<ENV_NAME>/...
├── json/<ENV_NAME>/...
└── train.json
```

Example `train.json` with one trajectory, shortened to three steps:

```json
[
  {
    "gpt_instruction": "A-frame metal structure with two hanging swings, black and red color scheme, intended for playground use.",
    "action": [2, 2, 1],
    "pos": [
      [1265.679, -997.808, -6.1],
      [1265.679, -997.808, -6.1],
      [1265.679, -997.808, -6.1]
    ],
    "yaw": [-1.254, -0.731, -0.207],
    "image_dict": {
      "uav_on_0": [
        {
          "rgb": "BrushifyUrban/1137/0/uav_on_0/00000.png",
          "depth": "BrushifyUrban/1137/0/uav_on_0_depth/d_00000.npy",
          "metainfo": {"fov": 89.904, "width": 512, "height": 512,
                       "quat_wb": [0.0, 0.0, -0.587, 0.81],
                       "camera_pos": [1265.99, -998.759, -6.094]}
        },
        {
          "rgb": "BrushifyUrban/1137/0/uav_on_0/00001.png",
          "depth": "BrushifyUrban/1137/0/uav_on_0_depth/d_00001.npy",
          "metainfo": {"fov": 89.904, "width": 512, "height": 512,
                       "quat_wb": [0.0, 0.0, -0.357, 0.934],
                       "camera_pos": [1266.423, -998.476, -6.093]}
        },
        {
          "rgb": "BrushifyUrban/1137/0/uav_on_0/00002.png",
          "depth": "BrushifyUrban/1137/0/uav_on_0_depth/d_00002.npy",
          "metainfo": {"fov": 89.904, "width": 512, "height": 512,
                       "quat_wb": [0.0, 0.0, -0.103, 0.995],
                       "camera_pos": [1266.657, -998.014, -6.093]}
        }
      ]
    }
  }
]
```

For training, set `train.raw_train_json` to this `train.json` and
`train.raw_media_root` to its `images/` directory; 

# Training and Evaluation

## 1. Environment

```bash
conda create -n octmem python=3.10 -y
conda activate octmem

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.48.1 tokenizers==0.21.4 peft==0.17.1 accelerate==0.34.2 \
  deepspeed==0.15.4 timm==0.9.10 draccus==0.11.5 numpy==1.26.4 \
  safetensors einops scipy pillow opencv-python orjson pyyaml tqdm
pip install flash-attn==2.7.4.post1 --no-build-isolation

# AirSim client (evaluation only)
pip install msgpack-rpc-python
pip install airsim --no-build-isolation
```

Also prepare:

- A local OpenVLA base checkpoint, e.g. `openvla/openvla-7b`.
- The UAV-ON training data (`train.json` and its `images/` directory), the
  evaluation episodes, and the AirSim scenes (`TRAIN_ENVS/`, `TEST_ENVS/`).
  See [iLearn-Lab/ACMMM25-UAV_ON](https://github.com/iLearn-Lab/ACMMM25-UAV_ON).

## 2. Configuration

```bash
cp -n configs/config.yaml.example configs/config.yaml
```

`configs/config.yaml` this to add your config; missing keys fall back to
`configs/config.yaml.example`. Relative paths are resolved from the repository
root. Set:

| Key | Value |
| --- | --- |
| `runtime.env_dir` | Conda environment directory; `null` uses the active environment |
| `runtime.cuda_visible_devices` | GPUs to use, default `"0,1,2,3"` |
| `runtime.num_processes` | Number of GPUs/processes, default `4` |
| `train.vla_path` | OpenVLA base checkpoint directory |
| `train.raw_train_json` | Training manifest `train.json` |
| `train.raw_media_root` | Image directory referenced by `train.json` |
| `eval.eval_dataset` | Evaluation episode JSON |
| `eval.airsim_env_root` | Directory containing `TRAIN_ENVS/` and `TEST_ENVS/` |

Training hyperparameters (`max_steps`, `epoches`, `save_steps`,
`total_batch_size`, `learning_rate`, `lora_rank`, ...) are in the `train`
section of `configs/config.yaml.example`.

## 3. Training

```bash
RUN_ID=octmem-new bash train_octmem_agent.sh
```

## 4. Evaluation


Merge a LoRA checkpoint into a full model, then evaluate it in AirSim:

```bash
RUN_DIR=runs/octmem-new
CKPT=checkpoints-7500

# Merge: writes $RUN_DIR/full-$CKPT
python -m utils.merge_lora --run_dir "$RUN_DIR" --merge_lora_path "$CKPT"

# Evaluate
bash eval.sh --model_path "$RUN_DIR/full-$CKPT" --output_folder "results/$CKPT"
```

Evaluation runs one worker per GPU, and each worker starts its own headless
AirSim instance. Metrics are written to `results/$CKPT/results.json`.

### Pretrained model

Our trained and merged OctMem-Agent model is available at [Octmem-agent](https://huggingface.co/kasoushu/Octmem-agent).
Download it and evaluate directly, without training or merging:

