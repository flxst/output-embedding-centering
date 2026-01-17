import os
from os.path import join, isfile

def _get_checkpoint_name(checkpoint_directory: str, n: int, d: int, variant: str, lr: str, gamma: str) -> str:
    """
    Args:
        checkpoint_directory: e.g. '../../lm-eval-checkpoints'
        n: e.g. 1
        d: e.g. 5
        variant: e.g. 'E'
        gamma: e.g. '10p0'

    Returns:
        checkpoint_name: e.g. 'exp15E-125M-20B-203k-bs48-lr3-embedding-g10p0-s1'
    """
    checkpoint_names = [subdirectory for subdirectory in os.listdir(checkpoint_directory) if subdirectory.startswith('exp') or subdirectory.startswith('wor')]
    _checkpoint_name = [elem for elem in checkpoint_names if (elem.startswith(f'exp{n}{d}{variant}') or elem.startswith(f'wor{n}{d}{variant}')) and lr in elem and gamma in elem]
    assert len(_checkpoint_name) == 1, f'ERROR! len(_checkpoint_name) == {len(_checkpoint_name)}'
    _checkpoint_name = _checkpoint_name[0]
    return _checkpoint_name

def _get_checkpoint_path(_checkpoint_name: str, _checkpoint_directory: str, _checkpoint: str) -> str:
    """
    Args:
        checkpoint_name: e.g. 'exp15E-125M-20B-203k-bs48-lr3-embedding-g10p0-s1'
        checkpoint_directory: e.g. '../../lm-eval-checkpoints'
    
    Returns:
        checkpoint_path: e.g. '../../lm-eval-checkpoints/exp15E-125M-20B-203k-bs48-lr3-embedding-g10p0-s1/ckpt_203450.hf/pytorch_model.bin'
    """
    checkpoint_directory = join(_checkpoint_directory, _checkpoint_name)
    checkpoint_pt = [elem for elem in os.listdir(checkpoint_directory) if elem.endswith(f'{_checkpoint}.pt')]
    assert len(checkpoint_pt) == 1, f'ERROR! len(checkpoints_pt) == {len(checkpoint_pt)}'
    checkpoint_pt = checkpoint_pt[0]
    checkpoint = join(checkpoint_directory, checkpoint_pt)
    assert isfile(checkpoint)
    return checkpoint

def get_checkpoint_name_and_checkpoint_path(checkpoint_directory: str, n: int, d: int, variant: str, lr: str, gamma: str, checkpoint: str) -> tuple[str, str]:
    """
    Args:
        checkpoint_directory: e.g. '../../lm-eval-checkpoints'
        n: e.g. 1
        d: e.g. 5
        variant: e.g. 'E'
        gamma: e.g. '10p0'
    
    Returns:
        checkpoint_name: e.g. 'exp15E-125M-20B-203k-bs48-lr3-embedding-g10p0-s1'
        checkpoint_path: e.g. '../../lm-eval-checkpoints/exp15E-125M-20B-203k-bs48-lr3-embedding-g10p0-s1/ckpt_203450.hf/pytorch_model.bin'
    """
    checkpoint_name = _get_checkpoint_name(checkpoint_directory, n, d, variant, lr, gamma)
    checkpoint_path = _get_checkpoint_path(checkpoint_name, checkpoint_directory, checkpoint)
    return checkpoint_name, checkpoint_path
