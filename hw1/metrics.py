import numpy as np
from models.base import to_matrix


def metrics(recommendations, relevant):
    discount = 1 / np.log2(np.arange(recommendations.shape[1]) + 2)
    ndcg, recall = [], []
    for ranked, truth in zip(recommendations, relevant):
        hits = np.isin(ranked, list(truth))
        ndcg.append(float(hits @ discount / discount[:min(len(truth), len(discount))].sum()))
        recall.append(float(hits.sum() / len(truth)))
    return {'NDCG@10': float(np.mean(ndcg)), 'Recall@10': float(np.mean(recall))}


class Evaluation:
    def __init__(self, history, future, shape):
        self.seen = to_matrix(history, shape)
        relevant = future[future.rating >= 4].groupby('user').item.agg(set)
        self.users = np.array(sorted(relevant.index), dtype=int)
        self.truth = [relevant[u] - set(self.seen[u].indices) for u in self.users]
        keep = [i for i, truth in enumerate(self.truth) if truth]
        self.users = self.users[keep]
        self.truth = [self.truth[i] for i in keep]

    def evaluate(self, model):
        rows = []
        for start in range(0, len(self.users), 128):
            users = self.users[start:start+128]
            ranked = model.recommend(users, self.seen, 10)
            rows.append(ranked)
        return metrics(np.vstack(rows), self.truth)
