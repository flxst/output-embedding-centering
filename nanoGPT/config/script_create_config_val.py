"""
Example: python script_create_config_val.py 31A 
                [--filter '']
                [--platform node]
                [--dataset openwebtext]

Explanation for 31A:
    3 => model_size = 760M
    1 => dataset_size = 1B
    A => variant = baseline

Explanation for dataset: 
    this is the validation dataset, which does not necessarily needs to be the same as the training dataset
"""

import argparse
import os
from os.path import join, abspath, dirname, isdir
from typing import Dict, Tuple
from helper_functions import get_prefix

ROOT_DIR = abspath(dirname(dirname(__file__)))

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def validate_input(args: argparse.Namespace) -> None:
    exp = args.exp
    assert len(exp) >= 3, f'ERROR! exp = {exp} needs to have at least length 3, e.g. 11A'

def get_config_paths(exp: str, filter: str, platform: str, porian: bool, wortsman: bool) -> Tuple[Dict[str, str], str]:

    prefix = get_prefix(porian, wortsman)

    # get config file path
    config_directory = join(ROOT_DIR, 'config')
    assert isdir(config_directory), f"ERROR! config directory {config_directory} does not exist"
    list_of_files = [elem for elem in os.listdir(config_directory) if elem.startswith(f'{prefix}{exp}')]
    if filter != '':
        list_of_files = [elem for elem in list_of_files if filter in elem]
    print(list_of_files, filter)
    assert len(list_of_files) == 1, f"ERROR! could not find a single config file {prefix}{exp}-{platform}* in {config_directory}"
    train_config_name = list_of_files[0] 
    val_config_name = list_of_files[0].replace(prefix, 'val')
    config_paths = {
        'train': join(config_directory, train_config_name),
        'val': join(config_directory, val_config_name),
    }
    train_config_name = train_config_name.split(".py")[0]
    return config_paths, train_config_name

def get_last_checkpoint(directory: str, train_config_name: str) -> str:
    checkpoint_directory = join(ROOT_DIR, directory, train_config_name)
    assert isdir(checkpoint_directory), f"ERROR! checkpoint directory {checkpoint_directory} does not exist"
    list_of_files = [elem for elem in os.listdir(checkpoint_directory) if elem.endswith('.pt')]
    ckpt_steps = [int(elem.split("ckpt_")[-1].split('.pt')[0]) for elem in list_of_files]
    largest_index = ckpt_steps.index(max(ckpt_steps))
    last_checkpoint = list_of_files[largest_index]
    return last_checkpoint

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main(_args):
    validate_input(_args)
    config_paths, train_config_name = get_config_paths(_args.exp, _args.filter, _args.platform, _args.porian, _args.wortsman)

    # read train config
    print(f"> read train config: {config_paths['train']}")
    with open(config_paths['train'], 'r') as f:
        _lines = f.readlines()

    # comment out line eval_iters
    _lines = [line if not line.startswith("eval_iters = ") else "# " + line for line in _lines]

    # turn off compile
    _lines = [line if not line.startswith("compile = ") else "compile = False\n" for line in _lines]

    # get last checkpoint from training
    last_checkpoint = get_last_checkpoint(_args.directory, train_config_name)
    print(f"> found checkpoint to evaluate: {last_checkpoint}")

    # start with new val lines at the top
    lines = [
        "# --- val ---\n",
        "init_from = 'resume'\n",
        f"init_directory = '{_args.directory}/{train_config_name}'\n",
        f"init_checkpoint = '{last_checkpoint}'\n",
        "eval_only = True\n",
        "eval_iters = 2000\n",
        f"dataset = '{args.dataset}'\n",
    ]

    # add train lines
    lines += _lines

    # write val config
    with open(config_paths['val'], 'w') as f:
        for line in lines:
            f.write(line)
    print(f"> wrote config file {config_paths['val']}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--filter', default='', type=str)
    parser.add_argument('--directory', default='output', type=str)
    parser.add_argument('--platform', default='node', type=str)
    parser.add_argument('--dataset', default='fineweb', type=str)
    parser.add_argument('--porian', action=argparse.BooleanOptionalAction)
    parser.add_argument('--wortsman', action=argparse.BooleanOptionalAction)
    parser.add_argument('--scaling', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    assert args.porian is None or args.wortsman is None, f'ERROR! Cannot specify both --porian and --wortsman'

    main(args)
