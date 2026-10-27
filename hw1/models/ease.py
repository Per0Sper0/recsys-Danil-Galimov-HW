import numpy as np
from scipy.linalg import solve
from .base import Model, to_matrix


class EASE(Model):
    def __init__(self, penalty=100):
        self.penalty = penalty

    def fit(self, data, shape):
        self.history = to_matrix(data, shape, (data.rating >= 4).astype(float))
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::shape[1] + 1] += self.penalty
        precision = solve(gram, np.eye(shape[1]), assume_a='pos', check_finite=False)
        self.weights = -precision / np.diag(precision)
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return self.history[users] @ self.weights
