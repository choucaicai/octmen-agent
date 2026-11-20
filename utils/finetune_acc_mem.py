"""Train OctMem-Agent on RGB-D episode streams with DeepSpeed ZeRO-1."""
import dataclasses
import json
import math
import os
import shutil
import time
import types
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('USE_TF', '0')

import draccus
import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data.memory_data import get_memory_dataset_and_collator
from .model_registry import resolve_model, checkpoint_model, stamp_model_config
from .project_config import config_section, resolve_path


TRAIN = config_section('train')
TRAIN_DATA_DIR = resolve_path(os.environ.get('TRAIN_DATA_DIR') or None)


@dataclass
class FinetuneMemoryConfig:
    model_module: str = 'model_octmem_agent'
    model_class: str = os.environ.get('MODEL_CLASS') or TRAIN['model_class']
    vla_path: Optional[Path] = resolve_path(os.environ.get('VLA_PATH') or TRAIN['vla_path'])
    raw_train_json: Optional[Path] = (TRAIN_DATA_DIR / 'train.json' if TRAIN_DATA_DIR else
                                     resolve_path(TRAIN['raw_train_json']))
    raw_val_json: Optional[Path] = resolve_path(TRAIN['raw_val_json'])
    raw_media_root: Optional[Path] = (TRAIN_DATA_DIR / 'images' if TRAIN_DATA_DIR else
                                     resolve_path(TRAIN['raw_media_root']))
    raw_index_path: Optional[Path] = resolve_path(TRAIN['raw_index_path'])
    raw_val_fraction: float = 0.05
    raw_camera_name: str = TRAIN['raw_camera_name']
    run_root_dir: Path = resolve_path(TRAIN['run_root_dir'])
    run_id_note: str = 'octmem-agent'
    seed: int = 11
    # Number of parallel trajectory lanes globally, before gradient accumulation.
    total_batch_size: int = TRAIN['total_batch_size']
    tbptt_steps: int = int(os.environ.get('TBPTT_STEPS') or TRAIN['tbptt_steps'])
    grad_accumulation_steps: int = int(os.environ.get('GRAD_ACCUMULATION_STEPS') or TRAIN['grad_accumulation_steps'])
    learning_rate: float = TRAIN['learning_rate']
    max_grad_norm: float = TRAIN['max_grad_norm']
    epoches: float = TRAIN['epoches']
    max_steps: Optional[int] = TRAIN['max_steps']
    save_steps: int = TRAIN['save_steps']
    eval_step: int = 1000
    log_steps: int = 10
    use_lora: bool = True
    lora_rank: int = TRAIN['lora_rank']
    lora_alpha: int = TRAIN['lora_alpha']
    lora_dropout: float = 0.0
    init_oct_model: bool = True
    gradient_checkpointing: bool = TRAIN['gradient_checkpointing']
    memory_voxel_size: float = 5.0
    # Q-Former token counts (scene and frontier queries).
    num_query_tokens: int = int(os.environ.get('NUM_QUERY_TOKENS') or TRAIN['num_query_tokens'])
    num_frontier_tokens: int = int(os.environ.get('NUM_FRONTIER_TOKENS') or TRAIN['num_frontier_tokens'])
    # Loads model weights only; optimizer/data stream resume is deliberately separate.
    warm_start_checkpoint: Optional[Path] = resolve_path(TRAIN['warm_start_checkpoint'])
    attn_implementation: str = os.environ.get('ATTN_IMPLEMENTATION') or TRAIN['attn_implementation']
    save_final: bool = True


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(v, device) for v in value)
    if is_dataclass(value):
        return replace(value, **{field.name: move_to_device(getattr(value, field.name), device)
                                 for field in fields(value)})
    return value



def tensors(value, path='state'):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            yield path, torch.as_tensor(value)
    elif dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            yield from tensors(getattr(value, f.name), f'{path}.{f.name}')
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from tensors(v, f'{path}.{k}')
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from tensors(v, f'{path}[{i}]')


def finite_report(value, path='state', limit=8):
    found = []
    for name, t in tensors(value, path):
        if not (t.is_floating_point() or t.is_complex()):
            continue
        # Chunk large flat master/Adam tensors to avoid a >GB temporary.
        for chunk in t.detach().reshape(-1).split(1 << 20):
            good = torch.isfinite(chunk)
            if not bool(good.all()):
                found.append({'path': name, 'shape': list(t.shape), 'dtype': str(t.dtype),
                              'bad_in_chunk': int((~good).sum()),
                              'first_bad_values': [str(item) for item in chunk[~good][:8].cpu().tolist()]})
                break
        if len(found) >= limit:
            break
    return found

def gather_objects(value):
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, value)
    return out


def any_rank(flag, device):
    t = torch.tensor(int(bool(flag)), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def clear_gradients(engine):
    zero = engine.optimizer
    zero.zero_grad(set_to_none=True)
    # Keep this dict object: DeepSpeed links high-precision gradient views to it.
    zero.averaged_gradients.clear()
    for master in zero.single_partition_of_fp32_groups:
        master.grad = None
    zero.micro_step_id = -1
    zero.params_already_reduced[:] = [False] * len(zero.params_already_reduced)
    engine.losses = None
    engine.set_gradient_accumulation_boundary(False)


def validate_engine(engine):
    import deepspeed
    if deepspeed.__version__ != '0.15.4':
        raise RuntimeError('ZeRO-1 gradient checks and stable norm require DeepSpeed 0.15.4')
    z = engine.optimizer
    if (engine.zero_optimization_stage() != 1 or z.cpu_offload or z.dtype != torch.bfloat16
            or z.has_moe_layers or engine.lr_scheduler is not None or engine.quantizer is not None):
        raise ValueError('Training requires BF16 ZeRO-1 without offload, MoE, a scheduler, or quantization')


def install_stable_grad_norm(engine):
    """Replace DeepSpeed 0.15.4's FP32 sum-of-squares norm.

    DeepSpeed computes each partition norm in FP64 but then does
    ``stack(...).square().sum().float()``.  Large *finite* gradients therefore
    overflow at the cast, and its subsequent ``0 * Inf`` sanitization produces
    NaN.  Keep the reduction in FP64 until the final square root.  The returned
    FP64 scalar is accepted by ZeRO's clipping code and prevents a valid
    finite gradient from being treated as a failure.
    """
    z = engine.optimizer

    def stable_scaled_global_norm(self, norm_type=2):
        if norm_type != 2:
            raise AssertionError('stable ZeRO norm supports only L2')
        group_norms = []
        for group_id, gradients in self.averaged_gradients.items():
            if gradients is None:
                continue
            total = torch.zeros((), dtype=torch.float64, device=self.device)
            for grad, param in zip(gradients, self.params_in_partition[group_id]):
                if grad is None:
                    continue
                if hasattr(param, 'ds_pipe_replicated') and param.ds_pipe_replicated:
                    continue
                total = total + torch.linalg.vector_norm(grad.detach().double()).square()
            process_groups = self.dp_process_group
            process_group = (process_groups[group_id]
                             if isinstance(process_groups, (list, tuple))
                             else process_groups)
            dist.all_reduce(total, op=dist.ReduceOp.SUM, group=process_group)
            self._model_parallel_all_reduce(total, op=dist.ReduceOp.SUM)
            group_norms.append(total.sqrt())
        if not group_norms:
            return torch.zeros((), dtype=torch.float64, device=self.device)
        return torch.linalg.vector_norm(torch.stack(group_norms), ord=2)

    z.scaled_global_norm = types.MethodType(stable_scaled_global_norm, z)
    return z
def inspect_gradients(engine, require_partition):
    z = engine.optimizer
    names = {id(p): n for n, p in engine.module.named_parameters()}
    report, checked = [], 0
    for i, gradients in z.averaged_gradients.items():
        for j, grad in enumerate(gradients or []):
            if grad is None:
                continue
            checked += 1
            params = z.params_in_partition[i]
            name = names.get(id(params[j]), f'group{i}.padding{j}') if j < len(params) else f'group{i}.padding{j}'
            report.extend(finite_report(grad, f'partition.{name}'))
    # Non-boundary ZeRO-1 gradients have not necessarily been reduced yet.
    for name, p in engine.module.named_parameters():
        for attr in ('grad', 'grad_accum'):
            g = getattr(p, attr, None)
            if g is not None:
                report.extend(finite_report(g, f'{name}.{attr}'))
    if require_partition and checked == 0:
        report.append({'error': 'No partition gradients inspected at update boundary'})
    return report[:16]


def inspect_optimizer(engine):
    z = engine.optimizer
    return (finite_report(z.single_partition_of_fp32_groups, 'master') +
            finite_report([z.optimizer.state.get(p, {}) for p in z.single_partition_of_fp32_groups], 'adam') +
            finite_report(z.bit16_groups_flat, 'bf16_parameters'))


def grouped_windows(loader, size):
    iterator = iter(loader)
    while batches := list(islice(iterator, size)):
        yield batches


def run_update(engine, core, batches, autocast):
    """Run one accumulation group and return the first failure reported by any rank."""
    metrics = torch.zeros(2, dtype=torch.float64, device=engine.device)
    def check(stage, micro, report):
        if any_rank(report, engine.device):
            return {'stage': stage, 'micro': micro, 'ranks': gather_objects(report)}
        return None
    for micro, batch in enumerate(batches):
        boundary = micro == len(batches) - 1
        engine.set_gradient_accumulation_boundary(boundary)
        report = finite_report(batch, 'input')
        if not batch['memory_window']:
            report.append({'error': 'empty memory window'})
        if event := check('input', micro, report):
            return event, metrics
        local = sum(int(s['labels'][:, 1:].ne(-100).sum()) for s in batch['memory_window'])
        total = torch.tensor(local, device=engine.device, dtype=torch.long)
        dist.all_reduce(total)
        if event := check('tokens', micro, [] if local > 0 and total.item() > 0 else [{'error': 'no action labels'}]):
            return event, metrics
        with autocast():
            output = engine(**move_to_device(batch, engine.device))
            loss = output['loss'] * (local * dist.get_world_size() / total)
        report = finite_report(output, 'output') + finite_report(core._memory_states, 'memory')
        if event := check('forward', micro, report):
            core.detach_memory()
            return event, metrics
        engine.backward(loss)
        core.detach_memory()  # exactly one TBPTT boundary per window
        if event := check('backward', micro, inspect_gradients(engine, boundary)):
            return event, metrics
        metrics[0] += output['loss'].detach().double() * local
        metrics[1] += local
        del output, loss
    norm = engine.optimizer.scaled_global_norm()
    if event := check('grad_norm', len(batches) - 1, finite_report(norm, 'scaled_global_grad_norm')):
        return event, metrics
    engine.step()
    if event := check('post_optimizer', len(batches) - 1, inspect_optimizer(engine)):
        return event, metrics
    dist.all_reduce(metrics)
    return None, metrics


def stop_on_training_failure(run_dir, accelerator, event, step, samples_seen):
    if accelerator.is_main_process:
        row = {'event': 'invalid_update', 'successful_steps': step, 'samples_seen': samples_seen, **event}
        tqdm.write(json.dumps(row))
        with (run_dir / 'anomalies.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
    raise FloatingPointError(f'Training stopped at {event["stage"]}; see {run_dir / "anomalies.jsonl"}')


def base_model(model):
    return model.get_base_model() if isinstance(model, PeftModel) else model


def load_memory_vla(cfg, device):
    selected = resolve_model(cfg.model_module, cfg.model_class)
    image_processor = selected.image_processor.from_pretrained(cfg.vla_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.vla_path)
    processor = selected.processor(image_processor, tokenizer)
    config_source = cfg.warm_start_checkpoint or cfg.vla_path
    # Pass through __init__ so model-specific configuration validation runs.
    values, _ = selected.config.get_config_dict(config_source)
    config = selected.config.from_dict(values)
    # These counts are training-time overrides rather than pretrained VLA fields.
    config.num_query_tokens = cfg.num_query_tokens
    config.num_frontier_tokens = cfg.num_frontier_tokens
    stamp_model_config(config, cfg.model_module, cfg.model_class, selected.auxiliaries)
    if cfg.warm_start_checkpoint:
        checkpoint_model(cfg.warm_start_checkpoint)
    config.memory_voxel_size = cfg.memory_voxel_size
    config.use_cache = False
    model = selected.model.from_pretrained(
        cfg.vla_path, config=config, torch_dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
    ).to(device)
    if cfg.init_oct_model and cfg.warm_start_checkpoint is None:
        model.initialize_oct_model(device=device)
        model.initialize_explore_model(device=device)
        model.to(dtype=torch.bfloat16)
    if cfg.use_lora:
        excluded = ('vision_backbone.',) + tuple(name + '.' for name in selected.auxiliaries)
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and not name.startswith(excluded)
                   and not name.endswith('lm_head')]
        if cfg.warm_start_checkpoint:
            model = PeftModel.from_pretrained(model, cfg.warm_start_checkpoint, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=targets, init_lora_weights='gaussian',
            ))
    elif cfg.warm_start_checkpoint:
        raise ValueError('warm_start_checkpoint currently expects a LoRA memory checkpoint')
    core = base_model(model)
    core.vision_backbone.requires_grad_(False)
    for name in selected.auxiliaries:
        getattr(core, name).requires_grad_(True)
    if cfg.warm_start_checkpoint:
        for name in selected.auxiliaries:
            weights = torch.load(Path(cfg.warm_start_checkpoint) / f'{name}.bin', map_location='cpu', weights_only=True)
            getattr(core, name).load_state_dict(weights, strict=True)
    if cfg.gradient_checkpointing:
        core.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        core.explore_model.qformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model, processor

def make_loaders(cfg, processor, local_batch_size):
    options = dict(train_json=cfg.raw_train_json, val_json=cfg.raw_val_json,
                   media_root=cfg.raw_media_root, index_path=cfg.raw_index_path,
                   image_transform=processor.image_processor.apply_transform,
                   tokenizer=processor.tokenizer, local_batch_size=local_batch_size,
                   tbptt_steps=cfg.tbptt_steps, seed=cfg.seed,
                   val_fraction=cfg.raw_val_fraction, camera_name=cfg.raw_camera_name)
    train_ds, _, collator = get_memory_dataset_and_collator(train=True, **options)
    val_ds, _, _ = get_memory_dataset_and_collator(train=False, **options)
    # Dataset owns both rank sharding and lane batching. Do not shard it again in Accelerate.
    loaders = [DataLoader(ds, batch_size=None, collate_fn=collator, num_workers=0)
               for ds in (train_ds, val_ds)]
    return train_ds, val_ds, *loaders


def evaluate_memory_loss(model, loader, accelerator):
    unwrapped = accelerator.unwrap_model(model)
    core = base_model(unwrapped)
    saved_states = core._memory_states
    core._memory_states = {}
    was_training = model.training
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=accelerator.device)
    try:
        # ZeRO-1 replicates weights; use the unwrapped model on unequal validation streams.
        with torch.no_grad(), accelerator.autocast():
            for batch in loader:
                output = unwrapped(**move_to_device(batch, accelerator.device))
                totals[0] += output['loss'].double() * output['token_count']
                totals[1] += output['token_count']
    finally:
        core._memory_states = saved_states
        model.train(was_training)
    totals = accelerator.reduce(totals, reduction='sum')
    return (totals[0] / totals[1]).item() if totals[1] > 0 else float('nan')


def save_memory_checkpoint(model, processor, cfg, directory, step):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    core = base_model(model)
    processor.save_pretrained(directory)
    selected = resolve_model(cfg.model_module, cfg.model_class)
    stamp_model_config(core.config, cfg.model_module, cfg.model_class, selected.auxiliaries)
    core.config.save_pretrained(directory)
    for name in selected.auxiliaries:
        torch.save({k: v.detach().cpu() for k, v in getattr(core, name).state_dict().items()}, directory / f'{name}.bin')
    (directory / 'training_config.json').write_text(json.dumps(asdict(cfg), default=str, indent=2))
    (directory / 'training_progress.json').write_text(json.dumps({'optimizer_step': step, 'weights_only': True}))
    shutil.copytree(_REPO_ROOT / cfg.model_module, directory / cfg.model_module,
                    ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)


@draccus.wrap()
def finetune(cfg: FinetuneMemoryConfig):
    for name, kind in (('vla_path', 'is_dir'), ('raw_train_json', 'is_file'), ('raw_media_root', 'is_dir')):
        value = getattr(cfg, name)
        if value is None or not getattr(Path(value), kind)():
            raise ValueError(f'Set train.{name} in configs/config.yaml or pass --{name}: {value}')
    if cfg.tbptt_steps not in (1, 2):
        raise ValueError('tbptt_steps must be 1 or 2')
    if cfg.grad_accumulation_steps < 1 or cfg.log_steps < 1:
        raise ValueError('Accumulation and logging intervals must be positive')
    if cfg.max_steps is not None and cfg.max_steps < 1:
        raise ValueError('max_steps must be positive')
    if cfg.max_steps is None and cfg.epoches <= 0:
        raise ValueError('epoches must be positive')
    if cfg.num_query_tokens < 1 or cfg.num_frontier_tokens < 1:
        raise ValueError('Scene and frontier query counts must be positive')
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accumulation_steps, mixed_precision='bf16',
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    if cfg.total_batch_size < accelerator.num_processes or cfg.total_batch_size % accelerator.num_processes:
        raise ValueError('total_batch_size must be a positive multiple of world size')
    local_batch = cfg.total_batch_size // accelerator.num_processes
    plugin = accelerator.state.deepspeed_plugin
    if plugin is not None:
        if plugin.zero_stage > 2:
            raise ValueError('This persistent-memory training/validation path supports ZeRO stages 0/1/2')
        plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = local_batch
        # Keep gradient communication in FP32; model compute remains BF16.
        plugin.deepspeed_config['communication_data_type'] = 'fp32'
        if cfg.max_grad_norm > 0:
            plugin.deepspeed_config['gradient_clipping'] = cfg.max_grad_norm
    if plugin is None or plugin.zero_stage != 1:
        raise ValueError('Memory training requires DeepSpeed ZeRO-1')
    set_seed(cfg.seed)
    run_dir = cfg.run_root_dir / cfg.run_id_note
    exists = accelerator.reduce(torch.tensor(int((run_dir / 'training_config.json').exists()),
                                            device=accelerator.device), reduction='sum')
    if exists.item():
        raise FileExistsError(f'Choose a new run_id_note; run already exists: {run_dir}')
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'training_config.json').write_text(json.dumps(asdict(cfg), default=str, indent=2))
    accelerator.wait_for_everyone()
    model, processor = load_memory_vla(cfg, accelerator.device)
    # Serialize index creation, which publishes through a shared .tmp path.
    with accelerator.main_process_first():
        train_ds, val_ds, train_loader, val_loader = make_loaders(cfg, processor, local_batch)
    core = base_model(model)
    core.config.norm_stats.update(train_ds.dataset_statistics)
    core.norm_stats = core.config.norm_stats
    if accelerator.is_main_process:
        (run_dir / 'dataset_statistics.json').write_text(json.dumps(train_ds.dataset_statistics, indent=2))
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    model, optimizer = accelerator.prepare(model, optimizer)
    core = base_model(accelerator.unwrap_model(model))
    accelerator.print(f'TBPTT={cfg.tbptt_steps}; global trajectory batch={cfg.total_batch_size}; '
                      f'up to {cfg.tbptt_steps * cfg.total_batch_size * cfg.grad_accumulation_steps} steps/update')
    print(f'rank={accelerator.process_index} train={train_ds.episodes.local_partition()} '
          f'val={val_ds.episodes.local_partition()}', flush=True)
    sample_budget = math.ceil(cfg.epoches * len(train_ds))
    step, samples_seen = 0, 0
    engine = accelerator.deepspeed_engine_wrapped.engine
    validate_engine(engine)
    install_stable_grad_norm(engine)
    clear_gradients(engine)
    if any_rank(inspect_optimizer(engine), accelerator.device):
        raise FloatingPointError('Initial model/optimizer is non-finite; choose a finite checkpoint')
    recent_sum, recent_tokens = 0.0, 0
    model.train()
    optimizer.zero_grad()
    # Always display optimizer steps. With an epoch budget, estimate the total
    # from observed samples/update because terminal TBPTT windows can be shorter.
    samples_per_update = cfg.tbptt_steps * cfg.total_batch_size * cfg.grad_accumulation_steps
    progress_total = cfg.max_steps if cfg.max_steps is not None else math.ceil(sample_budget / samples_per_update)
    progress_desc = 'Training' if cfg.max_steps is not None else 'Training (est. total)'
    training_started = last_update_time = time.perf_counter()
    with tqdm(
        total=progress_total, desc=progress_desc, unit='step',
        dynamic_ncols=True, mininterval=1.0, disable=not accelerator.is_main_process,
    ) as progress:
        for batches in grouped_windows(train_loader, cfg.grad_accumulation_steps):
            local_samples = sum(len(sample['episode_ids']) for batch in batches for sample in batch['memory_window'])
            totals = accelerator.reduce(torch.tensor(local_samples, device=accelerator.device), reduction='sum')
            samples_seen += int(totals)
            event, metrics = run_update(engine, core, batches, accelerator.autocast)
            if event is not None:
                stop_on_training_failure(run_dir, accelerator, event, step, samples_seen)
            recent_sum += metrics[0].item()
            recent_tokens += int(metrics[1])
            step += 1
            done = step >= cfg.max_steps if cfg.max_steps is not None else samples_seen >= sample_budget
            now = time.perf_counter()
            update_seconds = now - last_update_time
            elapsed_seconds = now - training_started
            if cfg.max_steps is None:
                remaining_samples = max(0, sample_budget - samples_seen)
                progress_total = step + math.ceil(remaining_samples / (samples_seen / step))
                progress.total = progress_total
            progress.set_postfix(
                loss=f'{recent_sum / recent_tokens:.4f}',
                **{'s/step': f'{update_seconds:.2f}'}, refresh=False,
            )
            progress.update(1)
            if step % cfg.log_steps == 0 or done:
                remaining = max(0, progress_total - step)
                row = {
                    'step': step,
                    'samples_seen': samples_seen, 'loss': recent_sum / recent_tokens,
                    'elapsed_seconds': elapsed_seconds, 'update_seconds': update_seconds,
                    'samples_per_second': samples_seen / max(elapsed_seconds, 1e-9),
                    'steps_per_second': step / max(elapsed_seconds, 1e-9),
                    'estimated_total_steps': progress_total,
                    'eta_seconds': remaining * elapsed_seconds / step,
                }
                if accelerator.is_main_process:
                    tqdm.write(json.dumps(row))
                    with (run_dir / 'metrics.jsonl').open('a') as f:
                        f.write(json.dumps(row) + '\n')
                recent_sum, recent_tokens = 0.0, 0
            if cfg.eval_step > 0 and step % cfg.eval_step == 0:
                progress.set_description('Validation', refresh=True)
                val_loss = evaluate_memory_loss(model, val_loader, accelerator)
                if accelerator.is_main_process:
                    tqdm.write(f'validation step={step} loss={val_loss}')
                progress.set_description(progress_desc, refresh=True)
            save = (cfg.save_steps > 0 and step % cfg.save_steps == 0) or (done and cfg.save_final)
            if save:
                progress.set_description('Saving checkpoint', refresh=True)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_memory_checkpoint(accelerator.unwrap_model(model), processor, cfg,
                                           run_dir / f'checkpoints-{step}', step)
                accelerator.wait_for_everyone()
                progress.set_description(progress_desc, refresh=True)
            last_update_time = time.perf_counter()
            if done:
                break
    core.reset_memory()
    accelerator.end_training()


if __name__ == '__main__':
    finetune()
