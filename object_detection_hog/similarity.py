import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def cosine_similarity(a, b):

    a = normalize(a)
    b = normalize(b)

    return np.dot(a, b)


def edge_weighted_score(cos_score, edge_density, alpha=0.7):

    return alpha * cos_score + (1-alpha) * edge_density
