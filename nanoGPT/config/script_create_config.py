"""
Example: python script_create_config.py 31D 
                [--placeholder 1]
                [--seed 1]
                [--wandb base]
                [--tokenizer gpt2]
                [--dataset openwebtext]
                [--out_dir_main_path /path/to/store/checkpoints]

Explanation for 31D:
    3 => model_size = 760M
    1 => dataset_size = 1B
    D => variant = VARIANT_D
"""

import argparse
import os
from pathlib import Path
from os.path import join, abspath, dirname
from typing import Dict, List, Union

ROOT_DIR = abspath(dirname(dirname(__file__)))

from helper_functions import get_model_size, get_dataset_size_and_steps, get_variant, get_global_batch_size, get_mbs_and_gas, get_log_interval, get_learning_rate, get_model_architecture, get_warmup, get_gpus, get_beta2, get_weight_decay, get_prefix
from settings import BATCH_SIZE, BLOCK_SIZE

# FONTS
import sys
from os.path import abspath
source_path = abspath('..')
if not source_path in sys.path:
    sys.path.append(source_path)
    from wandb_settings import PROJECT


# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def parse_input(args: argparse.Namespace) -> Dict[str, Union[float, str]]:
    exp = args.exp
    params = {}
    params['seed'] = args.seed

    params['tokenizer'] = args.tokenizer
    params['dataset'] = args.dataset
    params['wandb'] = args.wandb
    params['gamma'] = args.gamma
    params['gamma_str'] = f'{args.gamma:.0e}'
    params['model_size_str'] = get_model_size(exp[0], args.porian or args.wortsman)
    params['gpus'] = get_gpus(args.porian, params['model_size_str'])
    params['n_layer'], params['n_head'], params['n_embd'] = get_model_architecture(params['model_size_str'], args.porian or args.wortsman)

    params['bs'] = args.bs if args.bs > 0 else BATCH_SIZE
    params['lr'], params['lr_min'], params['lr_str'] = get_learning_rate(args.lr, params['model_size_str'], args.wortsman)

    params['warmup'] = get_warmup(args.porian, args.wortsman, params['model_size_str'], params['bs'])
    params['independent_weight_decay'], params['weight_decay'] = get_weight_decay(args.wortsman)

    params['dataset_size_str'], params['steps_str'], params['steps_iter'] = get_dataset_size_and_steps(exp[1], params['bs']*BLOCK_SIZE, args.porian, args.wortsman, args.scaling, params['model_size_str'])

    params['muloss'] = exp[2] in ['E', 'e']
    params['mucentering'] = exp[2] in ['R', 'r']
    params['z_loss'] = exp[2] in ['Z', 'z']
    params['wt'] = exp[2] in ['a', 'e', 'r', 'z']
    params['variant_str'] = get_variant(exp[2:])

    prefix = get_prefix(args.porian, args.wortsman)
    params['exp_name'] = f'{prefix}{exp}-{params["model_size_str"]}-{params["dataset_size_str"]}-{params["steps_str"]}-bs{params["bs"]}-lr{params["lr_str"]}-{params["variant_str"]}-g{params["gamma_str"]}-s{params["seed"]}'

    if args.beta2 > 0:
        params['beta2'] = args.beta2
        params['exp_name'] = params['exp_name'] + f'-beta2-{args.beta2}'.replace('.', 'p')
    else:
        params['beta2'] = get_beta2(exp[0], args.porian)

    params['config_file_path'] = join(ROOT_DIR, 'config', f'{params["exp_name"]}.py')
    if args.out_dir_main_path is None:
        cwd = Path(os.getcwd())
        params['out_dir_main_path'] = cwd.parent
    else:
        params['out_dir_main_path'] = args.out_dir_main_path
    return params

# -----------------------------------------------------------
# SECTION FUNCTIONS
# -----------------------------------------------------------
def get_section_types(params: Dict[str, Union[float, str]]) -> List[str]:
    dataset = params['dataset']
    wt = params['wt']
    muloss = params['muloss']
    mucentering = params['mucentering']
    z_loss = params['z_loss']
    gamma = params['gamma']
    _lines = [
        "# --- types ---",
        f"dataset = '{dataset}'",
        f"weight_tying = {wt}",
        f"muloss = {muloss}",
        f"mucentering = {mucentering}",
        f"z_loss = {z_loss}",
        f"gamma = {gamma}",
        ""
    ]

    return _lines

def get_section_experiment(params: Dict[str, Union[float, str]]) -> List[str]:
    compile = True
    wandb = params['wandb']
    exp_name = params['exp_name']
    out_dir_main_path = params['out_dir_main_path']

    _lines = [
        "# --- experiment ---",
        "wandb_log = False",
        f"wandb_project = '{wandb}'",
        f"wandb_run_name = '{exp_name}'",
        f"out_dir = '{out_dir_main_path}/output/{exp_name}'",
        f"compile = {compile}",
        "",
    ]
    return _lines

def get_section_batch_size(params: Dict[str, Union[float, str]]) -> List[str]:
    model_size_str = params["model_size_str"]

    gpus = params['gpus']
    block_size = BLOCK_SIZE
    batch_size = params["bs"]
    micro_batch_size, gradient_accumulation_steps = get_mbs_and_gas(batch_size, model_size_str, gpus, verbose=True)
    global_batch_size = get_global_batch_size(micro_batch_size, block_size, gradient_accumulation_steps, gpus)

    _lines = [
        "# --- batch size ---",
        f"# {micro_batch_size} micro batch size (samples) * {gradient_accumulation_steps} gradaccum * {gpus} GPUs * {block_size} block size",
        f"# = {batch_size} batch size (samples) * {block_size} block size",
        f"# = {global_batch_size} batch size (tokens)",
        f"gradient_accumulation_steps = {gradient_accumulation_steps}*{gpus}",
        f"batch_size = {micro_batch_size}",
        f"block_size = {block_size}",
        "",
    ]
    return _lines

def get_section_dataset_size(params: Dict[str, Union[float, str]]) -> List[str]:
    steps_iter = params['steps_iter']
    model_size_str = params['model_size_str']

    gpus = params['gpus']
    block_size = BLOCK_SIZE
    batch_size = params["bs"]
    micro_batch_size, gradient_accumulation_steps = get_mbs_and_gas(batch_size, model_size_str, gpus)
    global_batch_size = get_global_batch_size(micro_batch_size, block_size, gradient_accumulation_steps, gpus)

    global_tokens = global_batch_size * int(steps_iter)
    _lines = [
        "# --- dataset size ---",
        f"# tokens = {global_batch_size} * {steps_iter} ~ {global_tokens/10**9:.1f}B",
        f"max_iters = {steps_iter}",
        f"lr_decay_iters = {steps_iter}",
        "decay_lr = True",
        "",
    ]
    return _lines

def get_section_checkpointing(params: Dict[str, Union[float, str]]) -> List[str]:
    steps_iter = params['steps_iter']
    log_interval = get_log_interval(steps_iter)
    eval_interval = int(int(steps_iter)/10)
    _lines = [
        "# --- checkpointing ---",
        f"eval_interval = {eval_interval}",
        "eval_iters = 100",
        f"log_interval = {log_interval}",
        "",
    ]
    return _lines

def get_section_optimizer(params: Dict[str, Union[float, str]]) -> List[str]:
    beta2 = params['beta2']
    independent_weight_decay = params['independent_weight_decay']
    weight_decay = params['weight_decay']
    _lines = [
        "# --- optimizer ---",
        "optimizer_core = 'adamw'",
        "optimizer_embedding = 'adamw'",
        f"independent_weight_decay = {independent_weight_decay}",
        f"weight_decay = {weight_decay}  # general",
        "grad_clip = 1.0  # general; clip gradients at this value, or disable if == 0.0",
        "beta1 = 0.9  # adamw",
        f"beta2 = {beta2}  # adamw",
        "",
    ]
    return _lines

def get_section_model(params: Dict[str, Union[float, str]]) -> List[str]:
    model_size_str = params['model_size_str']
    n_layer = params['n_layer']
    n_head = params['n_head']
    n_embd = params['n_embd']

    _lines = [
        "# --- model ---",
        f"# {model_size_str}",
        f"n_layer = {n_layer}",
        f"n_head = {n_head}",
        f"n_embd = {n_embd}",
        f"learning_rate = {params['lr']}",
        f"min_lr = {params['lr_min']}",
        "",
    ]
    return _lines

def get_section_hyperparameters(params: Dict[str, Union[float, str]]) -> List[str]:
    warmup = params['warmup']
    _lines = [
        "# --- hyperparameters ---",
        f"# model_size / (batch_size * block_size) = {params['model_size_str']} / ({params['bs']} * {BLOCK_SIZE})",
        f"warmup_iters = {warmup}",
        "",
    ]
    return _lines

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main(_args):
    print("--- parse input ---")
    params = parse_input(_args)
    for k, v in params.items():
        print(f'{k}: {v}')
    print()

    print("--- create lines ---")
    lines = ["#", "#", "#", "", ""]
    lines += get_section_types(params)
    lines += get_section_experiment(params)
    lines += get_section_batch_size(params)
    lines += get_section_dataset_size(params)
    lines += get_section_checkpointing(params)
    lines += get_section_optimizer(params)
    lines += get_section_model(params)
    lines += get_section_hyperparameters(params)
    print(lines)
    print()

    print("--- write config file ---")
    with open(params['config_file_path'], 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"> wrote config file {params['config_file_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--gamma', default=1., type=float)
    parser.add_argument('--bs', default=0, type=int)
    parser.add_argument('--lr', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--porian', action=argparse.BooleanOptionalAction)
    parser.add_argument('--wortsman', action=argparse.BooleanOptionalAction)
    parser.add_argument('--scaling', action=argparse.BooleanOptionalAction)
    parser.add_argument('--beta2', default=-1., type=float)  # if -1, default value will be taken via get_beta2()
    parser.add_argument('--wandb', default='', type=str)  # will be taken from settings if not specified, see below
    parser.add_argument('--tokenizer', default='gpt2', type=str)
    parser.add_argument('--dataset', default='fineweb', type=str)
    parser.add_argument('--out_dir_main_path', default=None, type=str)

    args = parser.parse_args()

    if len(args.wandb) == 0:
        args.wandb = PROJECT  # take from settings

    assert args.porian is None or args.wortsman is None, f'ERROR! Cannot specify both --porian and --wortsman'

    main(args)
