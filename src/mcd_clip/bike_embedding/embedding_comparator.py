import numpy as np
from numpy import dot
from numpy.linalg import norm


def get_cosine_similarity(matrix_2d: np.ndarray, reference_1d_array: np.ndarray):
    assert matrix_2d.ndim == 2, "matrix_2d must be a 2D matrix"
    assert reference_1d_array.ndim == 1, "reference_1d_array must be a 1D array"
    return dot(matrix_2d, reference_1d_array) / (norm(matrix_2d, axis=1) * norm(reference_1d_array))


def get_cosine_distance(matrix_2d: np.ndarray, reference_1d_array: np.ndarray):
    return 1 - get_cosine_similarity(matrix_2d, reference_1d_array)
