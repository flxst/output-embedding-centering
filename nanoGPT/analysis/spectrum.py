
from os.path import isfile
import numpy as np
from numpy.linalg import svd
from analysis.embeddings import get_input_embeddings_torch


def compute_spectrum(path: str) -> float:
    save_flag = 0

    if isfile(path):
        spec = np.load(path)
        print(f"..loaded spectrum from file {path}")
    else:
        save_flag = 1
        embeddings_path = path.replace(".spectrum.npy", ".embeddings.npy")
        assert isfile(embeddings_path), f"could not find embeddings at {embeddings_path}"
        e = get_input_embeddings_torch(path=embeddings_path, place='output')
        _, spec, _ = svd(e.matrix, full_matrices=False, compute_uv=True)
        print(f"..loaded spectrum from embeddings {embeddings_path}")

    if save_flag:
        np.save(path, spec)
        print(f"..saved spectrum at file {path}")
        
    return spec
