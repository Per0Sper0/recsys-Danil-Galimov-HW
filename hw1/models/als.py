import numpy as np
from scipy.linalg import solve
from .base import Model, to_matrix


class ALS(Model):
    def __init__(self, rank=24, penalty=1, confidence=15, steps=12, seed=42):
        self.rank, self.penalty = rank, penalty
        self.confidence, self.steps, self.seed = confidence, steps, seed

    def targets(self, data):
        return (data.rating >= 4).astype(float).to_numpy()

    def solve_rows(self, observed, target, factors):
        common = factors.T @ factors + self.penalty * np.eye(self.rank)
        updated = np.zeros((observed.shape[0], self.rank))
        for row in range(observed.shape[0]):
            a, b = observed.indptr[row:row + 2]
            local = factors[observed.indices[a:b]]
            system = common + self.confidence * local.T @ local
            rhs = (1 + self.confidence) * local.T @ target.data[a:b]
            updated[row] = solve(system, rhs, assume_a='pos', check_finite=False)
        return updated

    def fit(self, data, shape):
        observed = to_matrix(data, shape)
        target = to_matrix(data, shape, self.targets(data))
        rng = np.random.default_rng(self.seed)
        self.items = rng.normal(0, .02, (shape[1], self.rank))
        for _ in range(self.steps):
            self.users = self.solve_rows(observed, target, self.items)
            self.items = self.solve_rows(observed.T.tocsr(), target.T.tocsr(), self.users)
        return self

    def score(self, users):
        return self.users[users] @ self.items.T
