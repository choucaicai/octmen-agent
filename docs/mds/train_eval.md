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

