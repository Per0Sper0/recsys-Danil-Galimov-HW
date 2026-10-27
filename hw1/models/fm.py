import numpy as np
import torch
from .base import Model
from history_features import examples


class FactorizationMachine(torch.nn.Module):
    def __init__(self, shape, factors, metadata):
        super().__init__()
        self.user = torch.nn.Embedding(shape[0], factors)
        self.item = torch.nn.Embedding(shape[1], factors)
        self.user_bias = torch.nn.Embedding(shape[0], 1)
        self.item_bias = torch.nn.Embedding(shape[1], 1)
        self.meta = torch.nn.Parameter(torch.randn(6, factors) * .01)
        self.linear = torch.nn.Parameter(torch.zeros(6))
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.metadata = metadata
        for emb in [self.user, self.item]:
            torch.nn.init.normal_(emb.weight, std=.01)
        for emb in [self.user_bias, self.item_bias]:
            torch.nn.init.zeros_(emb.weight)

    def forward(self, users, items, values):
        if not self.metadata:
            values = torch.zeros_like(values)
        vectors = torch.cat([self.user(users)[:, None, :], self.item(items)[:, None, :],
                             values[:, :, None] * self.meta[None, :, :]], dim=1)
        pairwise = .5 * (vectors.sum(1).square() - vectors.square().sum(1)).sum(1)
        return self.bias + self.user_bias(users).squeeze(1) + self.item_bias(items).squeeze(1) + values @ self.linear + pairwise


class FM(Model):
    def __init__(self, factors=16, reg=1e-5, metadata=True, epochs=12, seed=42):
        self.factors, self.reg, self.metadata = factors, reg, metadata
        self.epochs, self.seed = epochs, seed

    def fit(self, frame, shape, validation=None, prepared=None):
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        users, items, values, labels, self.user_values, self.item_values = prepared or examples(frame, shape, self.seed)
        users, items = torch.tensor(users), torch.tensor(items)
        values, labels = torch.tensor(values), torch.tensor(labels)
        self.net = FactorizationMachine(shape, self.factors, self.metadata)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=.002, weight_decay=self.reg)
        generator = torch.Generator().manual_seed(self.seed)
        best, stale, state = -np.inf, 0, None
        self.history = []
        self.best_epoch = self.epochs
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            permutation = torch.randperm(len(users), generator=generator)
            loss_sum = 0.
            for indices in permutation.split(4096):
                optimizer.zero_grad()
                logits = self.net(users[indices], items[indices], values[indices])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[indices])
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(indices)
            self.prepare_scores()
            value = validation(self) if validation else None
            self.history.append({'epoch': epoch, 'loss': loss_sum / len(users), 'ndcg': value})
            if validation:
                if value > best + 1e-12:
                    best, stale, self.best_epoch = value, 0, epoch
                    state = {key: tensor.detach().clone() for key, tensor in self.net.state_dict().items()}
                else:
                    stale += 1
                    if stale >= 3:
                        break
        if state is not None:
            self.net.load_state_dict(state)
            self.prepare_scores()
        return self

    def prepare_scores(self):
        # Exact FM algebra: within-side terms plus cross-side dot product.
        self.net.eval()
        with torch.no_grad():
            meta = self.net.meta.numpy()
            linear = self.net.linear.numpy()
            def side(embedding, bias, values, offset):
                if not self.metadata:
                    values = np.zeros_like(values)
                weighted = values[:, :, None] * meta[None, offset:offset+3]
                vector = embedding.weight.numpy() + weighted.sum(1)
                local = .5 * (vector ** 2 - embedding.weight.numpy() ** 2 - (weighted ** 2).sum(1)).sum(1)
                local += bias.weight.numpy().ravel() + values @ linear[offset:offset+3]
                return vector.copy(), local
            self.user_vectors, self.user_local = side(self.net.user, self.net.user_bias, self.user_values, 0)
            self.item_vectors, self.item_local = side(self.net.item, self.net.item_bias, self.item_values, 3)
            self.bias = self.net.bias.item()

    def score(self, users):
        return self.user_vectors[users] @ self.item_vectors.T + self.user_local[users, None] + self.item_local[None, :] + self.bias
