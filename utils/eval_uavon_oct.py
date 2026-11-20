"""UAV-ON evaluation for merged OctMem-Agent models, with one memory per episode.

torchrun supplies rank metadata only: workers own disjoint episodes and independent
models/simulators. There are no DDP collectives or DeepSpeed training wrappers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.parse import quote

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer

from data.prompt_llama2 import build_action_prompt
from data.raw_json_index import ACTION_MAP
from .action_statistics import resolve_action_stats_key
from .eval_utils import (
    ENV_DICT, AirSimObservations, calculate_distance_uavon,
    convert_to_action_id, getPoseAfterMakeAction, to_eularian_angles,
)
from .model_registry import checkpoint_model, resolve_model
from .project_config import config_section, resolve_path


EVAL = config_section('eval')
AIRSIM_SETTINGS = resolve_path(EVAL['airsim_settings'])


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def episode_key(row):
    return str(row['map_name']), str(row['episode_id'])


def result_path(folder, row):
    name = '--'.join(quote(part, safe='') for part in episode_key(row))
    return Path(folder) / 'temp' / f'{name}.json'


def select_episodes(cfg):
    rows = json.loads(Path(cfg.eval_dataset).read_text())
    keys = [episode_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Dataset contains duplicate (map_name, episode_id) keys')
    grouped = defaultdict(list)
    for row in rows:
        if cfg.env_names and row['map_name'] not in cfg.env_names:
            continue
        grouped[row['map_name']].append(row)
    if cfg.env_names and set(cfg.env_names) - set(grouped):
        raise ValueError(f'Unknown requested maps: {set(cfg.env_names) - set(grouped)}')
    selected = []
    for group in grouped.values():
        selected.extend(group[:cfg.eval_samples_per_env] if cfg.eval_samples_per_env else group)
    if cfg.max_episodes is not None:
        selected = selected[:cfg.max_episodes]
    if not selected:
        raise ValueError('No selected evaluation episodes')
    return selected


def partition_episodes(rows, rank, world_size):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['map_name']].append(row)
    return {name: group[rank::world_size] for name, group in grouped.items() if group[rank::world_size]}


def memory_parameters(cfg, saved_config=None):
    if saved_config is None:
        saved_config = json.loads((Path(cfg.model_path) / 'config.json').read_text())
    voxel_size = float(saved_config['memory_voxel_size'])
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('memory_voxel_size must be finite and positive')
    return {'memory_voxel_size': voxel_size}


def evaluation_settings(cfg):
    # Worker count/ports may change when resuming; evaluation semantics may not.
    values = {key: getattr(cfg, key) for key in (
        'eval_max_steps', 'success_distance', 'unnorm_key', 'attn_implementation',
        'camera_name', 'seed', 'settle_seconds', 'eval_samples_per_env', 'max_episodes', 'env_names',
    )}
    values.update(
        model_path=str(Path(cfg.model_path).resolve()),
        eval_dataset=str(Path(cfg.eval_dataset).resolve()),
        dataset_sha256=hashlib.sha256(Path(cfg.eval_dataset).read_bytes()).hexdigest(),
        model_manifest_sha256=hashlib.sha256((Path(cfg.model_path) / 'weights_manifest.json').read_bytes()).hexdigest(),
        memory_parameters=memory_parameters(cfg),
    )
    return values


def ensure_eval_config(cfg):
    folder = Path(cfg.output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    settings = evaluation_settings(cfg)
    with (folder / '.eval.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = folder / 'eval_config.json'
        if path.exists():
            previous = json.loads(path.read_text())
            for key in ('evaluator_sha256', 'model_source_sha256'):
                previous.pop(key, None)
            if 'memory_parameters' not in previous:
                previous['memory_parameters'] = settings['memory_parameters']
            if previous != settings:
                raise ValueError(f'Output folder belongs to a different evaluation: {folder}')
        else:
            atomic_json(path, settings)


def load_policy(cfg, device):
    path = Path(cfg.model_path).resolve()
    saved_config = json.loads((path / 'config.json').read_text())
    effective_memory = memory_parameters(cfg, saved_config)
    selected = resolve_model(*checkpoint_model(path))
    policy, info = selected.model.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation=cfg.attn_implementation,
        low_cpu_mem_usage=True, local_files_only=True, output_loading_info=True,
    )
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Merged model is incomplete: {info}')
    policy = policy.eval().to(device)
    policy.requires_grad_(False)
    policy.config.use_cache = True
    policy.config.text_config.use_cache = True
    policy.generation_config.use_cache = True
    processor = selected.processor(
        selected.image_processor.from_pretrained(path, local_files_only=True),
        AutoTokenizer.from_pretrained(path, local_files_only=True),
    )
    key = resolve_action_stats_key(policy.norm_stats, cfg.unnorm_key)
    if policy.get_action_dim(key) != 8:
        raise ValueError(f'UAV-ON requires eight action components, got {policy.get_action_dim(key)}')
    print(f'MODEL_READY rank={os.environ.get("RANK", "0")} checkpoint={path.name} '
          f'unnorm_key={key} memory={effective_memory["memory_voxel_size"]}', flush=True)
    return policy, processor, key


@torch.inference_mode()
def get_action(policy, processor, observation, instruction, episode_id, step_id, device, unnorm_key):
    image = Image.fromarray(observation['rgb']).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
    depth = np.asarray(observation['depth'], dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_NEAREST)
    meta = observation['metainfo']
    params = np.asarray(list(meta['quat_wb']) + [float(meta['fov'])], dtype=np.float32)
    pose = np.asarray(meta['camera_pos'], dtype=np.float32)
    if params.shape != (5,) or pose.shape != (3,) or not np.isfinite(params).all() or not np.isfinite(pose).all():
        raise ValueError(f'Invalid camera geometry: params={params}, pose={pose}')
    inputs = processor(build_action_prompt(instruction.strip()), image).to(device, dtype=torch.bfloat16)
    # predict_action appends this token itself but does not extend attention_mask.
    # Append both here so generation has a correctly aligned prompt and mask.
    if inputs['input_ids'][0, -1].item() != 29871:
        inputs['input_ids'] = torch.cat((inputs['input_ids'], inputs['input_ids'].new_tensor([[29871]])), dim=1)
        inputs['attention_mask'] = torch.cat((inputs['attention_mask'], inputs['attention_mask'].new_ones((1, 1))), dim=1)
    inputs.update(
        camera_depths=torch.from_numpy(depth).to(device)[None, None],
        camera_params=torch.from_numpy(params).to(device)[None, None],
        camera_poses=torch.from_numpy(pose).to(device)[None, None],
        episode_ids=[episode_id], step_ids=[step_id],
    )
    raw = policy.predict_action(
        **inputs, unnorm_key=unnorm_key, do_sample=False, use_cache=True,
        min_new_tokens=policy.get_action_dim(unnorm_key),
    )
    if np.asarray(raw).shape != (8,) or not np.isfinite(raw).all():
        raise ValueError(f'Invalid action output: {raw}')
    rounded = np.rint(raw).astype(int)
    action_id = int(convert_to_action_id(rounded))
    valid_action = bool(np.array_equal(rounded, ACTION_MAP[action_id]))
    memory = policy._memory_states[episode_id]
    if len(policy._memory_states) != 1 or memory.step_id != step_id:
        raise ValueError('Episode memory leaked or did not advance exactly once')
    coarse_cells = int(memory.keys[:, 3].eq(0).sum().item())
    fine_cells = len(memory.keys) - coarse_cells
    return action_id, {
        'raw': raw.tolist(), 'rounded': rounded.tolist(), 'valid_action': valid_action,
        'fine_cells': fine_cells, 'coarse_cells': coarse_cells,
    }


class ManagedSimulator(AirSimObservations):
    """Reuse established capture/pose code; own and bound this process's lifecycle."""
    def __init__(self, env_name, port, gpu_id, folder, startup_timeout):
        import airsim

        self.process = None
        self.log_file = None
        self._client = None
        with socket.socket() as probe:
            # A just-closed AirSim server may leave TIME_WAIT connections. These
            # are not live listeners; permit reuse while still rejecting a bound listener.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', port))
        directory = Path(folder) / 'simulator'
        directory.mkdir(parents=True, exist_ok=True)
        settings = json.loads(AIRSIM_SETTINGS.read_text())
        settings['ApiServerPort'] = port
        settings_path = directory / f'settings-{port}.json'
        atomic_json(settings_path, settings)
        log_path = directory / f'{env_name}-{port}.log'
        self.log_file = log_path.open('a')
        command = ['bash', ENV_DICT[env_name], '-RenderOffscreen', '-NoSound', '-NoVSync',
                   f'-GraphicsAdapter={gpu_id}', f'--settings={settings_path.resolve()}']
        print(f'SIM_START map={env_name} port={port} gpu={gpu_id} log={log_path}', flush=True)
        try:
            self.process = subprocess.Popen(command, stdout=self.log_file, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
            deadline = time.monotonic() + startup_timeout
            last_error = None
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'Simulator exited with {self.process.returncode}; see {log_path}')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1):
                        pass
                    self._client = airsim.MultirotorClient(port=port, timeout_value=10)
                    if not self._client.ping():
                        raise RuntimeError('AirSim ping failed')
                    self._client.enableApiControl(True)
                    self._client.armDisarm(True)
                    print(f'SIM_READY map={env_name} port={port}', flush=True)
                    return
                except Exception as exc:
                    last_error = exc
                    if self._client is not None:
                        self._client.client.close()
                        self._client = None
                    time.sleep(1)
            raise TimeoutError(f'Simulator not ready within {startup_timeout}s: {last_error}; see {log_path}')
        except BaseException:
            self.cleanup()
            raise

    def cleanup(self):
        if self._client is not None:
            try:
                self._client.client.close()
            except Exception:
                pass
            self._client = None
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
            self.process = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def __del__(self):
        self.cleanup()


def capture_observation(simulator, camera_name, episode_id, step_id, rank):
    # AirSim occasionally returns an empty/truncated compressed frame. Retry
    # capture only: do not advance the action step or episode memory on failure.
    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            return simulator.capture_rgbd(camera_name)
        except OSError as exc:  # Includes PIL.UnidentifiedImageError and truncated-image errors.
            if attempt == attempts:
                raise
            print(f'CAPTURE_RETRY rank={rank} episode={episode_id} step={step_id} '
                  f'attempt={attempt}/{attempts} error={type(exc).__name__}: {exc}', flush=True)
            time.sleep(0.5)


def evaluate_episode(cfg, row, simulator, policy, processor, key, device, rank):
    start = row['start_pose']
    _, _, yaw = to_eularian_angles(*start['start_quaternionr'])
    pose = [*start['start_position'], yaw]
    trajectory = [list(pose)]
    actions, details = [], []
    memory_id = f'{row["map_name"]}:{row["episode_id"]}'
    policy.reset_memory()
    oracle_success = int(calculate_distance_uavon(pose[:3], row['pose']) <= cfg.success_distance)
    started = time.monotonic()
    simulator._set_camera_pose(*pose, 0, 0)
    try:
        for step in range(cfg.eval_max_steps):
            time.sleep(cfg.settle_seconds)
            observation = capture_observation(simulator, cfg.camera_name, memory_id, step, rank)
            action, detail = get_action(policy, processor, observation, row['description'], memory_id,
                                        step, device, key)
            actions.append(action)
            details.append(detail)
            pose = getPoseAfterMakeAction(pose, action)
            simulator._set_camera_pose(*pose, 0, 0)
            trajectory.append(list(pose))
            distance = calculate_distance_uavon(pose[:3], row['pose'])
            oracle_success = max(oracle_success, int(distance <= cfg.success_distance))
            if step % cfg.log_step_interval == 0 or action == 0 or step + 1 == cfg.eval_max_steps:
                print(f'STEP rank={rank} episode={memory_id} step={step} action={action} '
                      f'valid={detail["valid_action"]} distance={distance:.3f} '
                      f'cells={detail["fine_cells"]}/{detail["coarse_cells"]}', flush=True)
            if action == 0:
                break
        distance = calculate_distance_uavon(pose[:3], row['pose'])
        result = dict(row)
        result.update(
            model_path=str(Path(cfg.model_path).resolve()),
            acc=int(distance <= cfg.success_distance), osr=oracle_success, ne=distance,
            gen_pose_list=trajectory, predicted_actions=actions, action_details=details,
            steps=len(actions), stopped=bool(actions[-1] == 0),
            invalid_action_count=sum(not item['valid_action'] for item in details),
            elapsed_seconds=time.monotonic() - started, rank=rank,
        )
        atomic_json(result_path(cfg.output_folder, row), result)
        print(f'EPISODE_COMPLETE rank={rank} episode={memory_id} steps={len(actions)} '
              f'SR={result["acc"]} OSR={result["osr"]} NE={distance:.3f}', flush=True)
        return result
    finally:
        policy.reset_memory()


def preflight(cfg):
    path = Path(cfg.model_path)
    config = json.loads((path / 'config.json').read_text())
    checkpoint_model(path)
    for name in ('model.safetensors.index.json', 'weights_manifest.json', 'config.json'):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    memory_parameters(cfg, config)
    resolve_action_stats_key(config['norm_stats'], cfg.unnorm_key)
    if cfg.eval_max_steps < 1 or cfg.success_distance <= 0 or cfg.log_step_interval < 1 or cfg.settle_seconds < 0:
        raise ValueError('Invalid evaluation limits')
    for value in (cfg.eval_samples_per_env, cfg.max_episodes):
        if value is not None and value < 1:
            raise ValueError('Episode limits must be positive')


def preflight_simulator(rows):
    if AIRSIM_SETTINGS is None or not AIRSIM_SETTINGS.is_file():
        raise FileNotFoundError(f'Set eval.airsim_settings to an AirSim settings file: {AIRSIM_SETTINGS}')
    for name in {row['map_name'] for row in rows}:
        if not ENV_DICT.get(name) or not ENV_DICT[name].is_file():
            raise FileNotFoundError(f'No simulator executable configured for {name}')


def summarize(cfg, rows):
    from .get_results import process_results, print_results
    ensure_eval_config(cfg)
    groups = defaultdict(list)
    for row in rows:
        path = result_path(cfg.output_folder, row)
        if not path.is_file():
            raise FileNotFoundError(f'Missing episode result: {path}')
        groups[row['map_name']].append(json.loads(path.read_text()))
    summary = process_results(groups)
    summary.update(status='complete', expected_episodes=len(rows),
                   model_path=str(Path(cfg.model_path).resolve()), memory_parameters=memory_parameters(cfg))
    summary['env_results'] = {
        name: {'env': name, 'result': process_results({name: group})['total_results']}
        for name, group in groups.items()
    }
    summary['invalid_action_count'] = sum(r['invalid_action_count'] for group in groups.values() for r in group)
    atomic_json(Path(cfg.output_folder) / 'results.json', summary)
    print_results(summary)
    print(f'EVAL_COMPLETE checkpoint={Path(cfg.model_path).name} episodes={len(rows)} '
          f'results={Path(cfg.output_folder) / "results.json"}', flush=True)


def run(cfg, rows):
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.set_num_threads(cfg.num_threads)
    torch.cuda.set_device(local_rank)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device('cuda', local_rank)
    ensure_eval_config(cfg)
    groups = partition_episodes(rows, rank, world)
    groups = {name: [row for row in group if not result_path(cfg.output_folder, row).is_file()]
              for name, group in groups.items()}
    groups = {name: group for name, group in groups.items() if group}
    print(f'WORKER_START rank={rank}/{world} device={device} episodes={sum(map(len, groups.values()))}', flush=True)
    if not groups:
        return
    policy, processor, key = load_policy(cfg, device)
    gpu_ids = cfg.simulator_gpu_ids or os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(torch.cuda.device_count()))))
    gpu_id = int(gpu_ids.split(',')[local_rank])
    completed = 0
    for scene_index, (name, group) in enumerate(groups.items()):
        # AirSim can leave a listening/TIME_WAIT socket briefly after its wrapper
        # exits. Never reuse a worker's preceding scene port during this run.
        scene_port = cfg.airsim_default_port + scene_index * world + rank
        if scene_port > 65535:
            raise ValueError(f'Simulator port is out of range: {scene_port}')
        simulator = ManagedSimulator(name, scene_port, gpu_id,
                                     cfg.output_folder, cfg.simulator_startup_timeout)
        try:
            for row in group:
                try:
                    result = evaluate_episode(cfg, row, simulator, policy, processor, key, device, rank)
                except Exception:
                    atomic_json(Path(cfg.output_folder) / 'errors' / f'rank-{rank}.json',
                                {'episode': episode_key(row), 'traceback': traceback.format_exc()})
                    raise
                completed += 1
                atomic_json(Path(cfg.output_folder) / f'progress-rank-{rank}.json',
                            {'rank': rank, 'completed_this_run': completed, 'last_episode': episode_key(row),
                             'last_steps': result['steps'], 'updated_at': time.time()})
        finally:
            simulator.cleanup()
    print(f'WORKER_COMPLETE rank={rank} episodes={completed}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model_path = resolve_path(EVAL['model_path'])
    output_folder = resolve_path(EVAL['output_folder'])
    parser.add_argument('--model_path', default=model_path, required=model_path is None)
    parser.add_argument('--output_folder', default=output_folder, required=output_folder is None)
    parser.add_argument('--eval_dataset', default=resolve_path(os.environ.get('EVAL_DATASET') or EVAL['eval_dataset']))
    parser.add_argument('--eval_max_steps', type=int, default=EVAL['eval_max_steps'])
    parser.add_argument('--success_distance', type=float, default=EVAL['success_distance'])
    parser.add_argument('--eval_samples_per_env', type=int)
    parser.add_argument('--max_episodes', type=int)
    parser.add_argument('--env_names', nargs='+')
    parser.add_argument('--unnorm_key', default=EVAL['unnorm_key'])
    parser.add_argument('--attn_implementation', choices=['eager', 'sdpa', 'flash_attention_2'], default=EVAL['attn_implementation'])
    parser.add_argument('--airsim_default_port', type=int, default=EVAL['airsim_default_port'])
    parser.add_argument('--simulator_gpu_ids', default=EVAL['simulator_gpu_ids'])
    parser.add_argument('--simulator_startup_timeout', type=float, default=EVAL['simulator_startup_timeout'])
    parser.add_argument('--camera_name', default=EVAL['camera_name'])
    parser.add_argument('--settle_seconds', type=float, default=EVAL['settle_seconds'])
    parser.add_argument('--log_step_interval', type=int, default=EVAL['log_step_interval'])
    parser.add_argument('--seed', type=int, default=EVAL['seed'])
    parser.add_argument('--num_threads', type=int, default=EVAL['num_threads'])
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--summarize_only', action='store_true')
    return parser.parse_args()


def main():
    cfg = parse_args()
    rows = select_episodes(cfg)
    preflight(cfg)
    if cfg.dry_run or not cfg.summarize_only:
        preflight_simulator(rows)
    if cfg.dry_run:
        print(json.dumps({'model_path': str(cfg.model_path), 'output_folder': str(cfg.output_folder),
                          'episodes': len(rows), 'maps': len({r['map_name'] for r in rows}),
                          'max_steps': cfg.eval_max_steps, 'unnorm_key': cfg.unnorm_key,
                          'memory_parameters': memory_parameters(cfg)}, indent=2))
    elif cfg.summarize_only:
        summarize(cfg, rows)
    else:
        run(cfg, rows)


if __name__ == '__main__':
    # torchrun terminates siblings on a worker error; unwind simulator ownership too.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    main()
