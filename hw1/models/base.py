import numpy as np
from scipy.sparse import csr_matrix


def to_matrix(data, shape, values=None):
    if values is None:
        values = np.ones(len(data))
    return csr_matrix((values, (data.user, data.item)), shape=shape, dtype=float)


class Model:
    def recommend(self, users, seen, k=10):
        scores = self.score(users).copy()
        for row, user in enumerate(users):
            scores[row, seen[user].indices] = -np.inf
        return np.argsort(-scores, axis=1, kind='stable')[:, :k]
