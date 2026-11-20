"""Set up the configured environment for the training and evaluation wrappers."""
from datetime import datetime
import os
import subprocess
import sys

from .project_config import config_section, resolve_path


def main():
    mode, *args = sys.argv[1:]
    runtime = config_section('runtime')
    env_dir = resolve_path(os.environ.get('MEM_ENV_DIR') or runtime['env_dir'])
    if env_dir is not None:
        if not (env_dir / 'bin/python').is_file():
            raise SystemExit(f'Python not found in configured environment: {env_dir}')
        os.environ['PATH'] = f'{env_dir / "bin"}{os.pathsep}{os.environ["PATH"]}'
    python = str(env_dir / 'bin/python') if env_dir else sys.executable
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(runtime['cuda_visible_devices']))
    os.environ.setdefault('OMP_NUM_THREADS', str(runtime['omp_num_threads']))
    os.environ.update(TOKENIZERS_PARALLELISM='false', USE_TF='0', PYTHONUNBUFFERED='1')
    num_processes = os.environ.get('NUM_PROCESSES') or str(runtime['num_processes'])
    help_requested = any(arg in ('-h', '--help') for arg in args)
    if mode == 'train':
        settings = config_section('train')
        accelerator = resolve_path(os.environ.get('ACCELERATE_CONFIG') or settings['accelerate_config'])
        port = os.environ.get('MASTER_PORT') or settings['master_port']
        tbptt_steps = os.environ.get('TBPTT_STEPS') or settings['tbptt_steps']
        run_id = os.environ.get('RUN_ID') or f'model_octmem_agent-tbptt{tbptt_steps}-{datetime.now():%Y%m%d-%H%M%S}'
        train = ['utils.finetune_acc_mem', '--run_id_note', run_id, *args]
        if help_requested:
            os.execv(python, [python, '-m', *train])
        os.execv(python, [python, '-m', 'accelerate.commands.accelerate_cli', 'launch',
                          '--config_file', str(accelerator), '--num_processes', num_processes,
                          '--main_process_port', str(port), '--module', *train])
    if mode == 'eval':
        evaluator = ['utils.eval_uavon_oct', *args]
        if help_requested or any(arg in ('--dry_run', '--summarize_only') for arg in args):
            os.execv(python, [python, '-m', *evaluator])
        result = subprocess.run([python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                                 f'--nproc_per_node={num_processes}', '--max_restarts=0',
                                 '--module', *evaluator])
        if result.returncode:
            raise SystemExit(result.returncode)
        os.execv(python, [python, '-m', *evaluator, '--summarize_only'])
    raise SystemExit(f'Unknown launch mode: {mode}')


if __name__ == '__main__':
    main()
