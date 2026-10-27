import numpy as np
from scipy.sparse import csr_matrix
from .base import Model, to_matrix


class ItemKNN(Model):
    def __init__(self, neighbors=80):
        self.neighbors = neighbors

    def fit(self, data, shape):
        self.history = to_matrix(data, shape, (data.rating >= 4).astype(float))
        overlap = (self.history.T @ self.history).toarray()
        counts = np.asarray(self.history.sum(axis=0)).ravel()
        similarity = overlap / (counts[:, None] + counts[None, :] - overlap + 10)
        np.fill_diagonal(similarity, 0)
        sources = np.argsort(-similarity, axis=0, kind='stable')[:self.neighbors]
        targets = np.broadcast_to(np.arange(shape[1]), sources.shape)
        self.weights = csr_matrix((similarity[sources, targets].ravel(),
                                   (sources.ravel(), targets.ravel())), shape=similarity.shape)
        return self

    def score(self, users):
        return (self.history[users] @ self.weights).toarray()
