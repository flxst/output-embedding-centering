import numpy as np
import torch
from os.path import isfile, join

def _load_embeddings(_checkpoint_path: str, positions) -> dict[str, np.array]:
    """
    Args:
        checkpoint_path: e.g. '../../lm-eval-checkpoints/exp15E-125M-20B-203k-bs48-lr3-embedding-g10p0-s1/ckpt_203450.hf/pytorch_model.bin'
    
    Returns:
        embeddings: e.g. {
            'input':  (np.array of shape V, H),
            'output': (np.array of shape V, H),
        }
    """
    checkpoint = torch.load(_checkpoint_path, map_location='cpu')
    embeddings = {}

    if 'input' in positions:
        try:
            embeddings['input'] = checkpoint['model.transformer.wte.weight'].detach().numpy()
        except KeyError:
            embeddings['input'] = checkpoint['model']['_orig_mod.transformer.wte.weight'].detach().numpy()
    if 'output' in positions:
        try:
            embeddings['output'] = checkpoint['model.lm_head.embedding.weight'].detach().numpy()
        except KeyError:
            embeddings['output'] = checkpoint['model']['_orig_mod.lm_head.embedding.weight'].detach().numpy()

    return embeddings

def load_embeddings(_checkpoint_path: str, _checkpoint_name: str, positions: list[str], overwrite: bool = False) -> dict[str, np.array]:
    """
    Args:
        checkpoint_path: e.g. '../../lm-eval-checkpoints/exp15E-125M-20B-203k-bs48-lr3-embedding-g10p0-s1/ckpt_203450.hf/pytorch_model.bin'
        checkpoint_name: e.g. 'exp15E-125M-20B-203k-bs48-lr3-embedding-g10p0-s1'
        positions: e.g. ['input', 'output']
        overwrite: e.g. False

    Returns:
        embeddings: e.g. {
            'input':  (np.array of shape V, H),
            'output': (np.array of shape V, H),
        }
    """

    _embeddings_path_base = join('./embeddings', _checkpoint_name)
    _embeddings_path = {position: f'{_embeddings_path_base}-{position}.npy' for position in ['output']}

    if isfile(_embeddings_path['output']) and overwrite is False:
        # load embeddings
        embeddings = {position: np.load(_embeddings_path[position]) for position in positions}
        print(f'> loaded embeddings for {_checkpoint_name}')
    else:
        # extract embeddings
        embeddings = _load_embeddings(_checkpoint_path, positions)
        # save embeddings
        for position in positions:
            np.save(_embeddings_path[position], embeddings[position])
            print(f'> saved {position} embeddings for {_checkpoint_name}')

    for position in positions:
        print(f'.. position = {position.rjust(6)}, embeddings[position].shape = {embeddings[position].shape}')

    return embeddings
