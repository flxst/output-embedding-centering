
import math
import numpy as np
from os.path import isfile
from analysis.embeddings import get_input_embeddings_torch


def _isotropy(embeddings: np.ndarray) -> float:
    """
    Computes isotropy score.
    Defined in Section 5.1, equations (7) and (8) of the paper.

    Args:
        embeddings: word vectors of shape (n_words, n_dimensions)

    Returns:
        float: isotropy score
    """
    min_z = math.inf
    max_z = -math.inf

    _, eigen_vectors = np.linalg.eig(np.matmul(embeddings.T, embeddings))
    for i in range(eigen_vectors.shape[1]):
        z_c = np.matmul(embeddings, np.expand_dims(eigen_vectors[:, i], 1))
        z_c = np.exp(z_c)
        z_c = np.sum(z_c)
        min_z = min(z_c, min_z)
        max_z = max(z_c, max_z)

    return round((min_z / max_z).item(), 4)


def isotropy_run(matrix) -> float:
    matrix = matrix  # Coupled Adam: no transpose because matrix is E^T!!!
    return _isotropy(embeddings=matrix)

def compute_isotropy(path: str, place: str) -> float:
    save_flag = 0

    if isfile(path):
        iso = np.load(path)
        print(f"..loaded isotropy from file {path}")
    else:
        save_flag = 1
        embeddings_path = path.replace(".isotropy.npy", ".embeddings.npy").replace(".isotropy-input.npy", ".embeddings-input.npy")
        assert isfile(embeddings_path), f"could not find embeddings at {embeddings_path}"
        e = get_input_embeddings_torch(path=embeddings_path, place=place)
        iso = isotropy_run(e.matrix)
        print(f"..loaded isotropy from embeddings {embeddings_path}")

    if save_flag:
        np.save(path, iso)
        print(f"..saved isotropy at file {path}")
        
    return iso
