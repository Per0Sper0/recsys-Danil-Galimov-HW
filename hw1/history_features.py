import numpy as np


def histories(frame, column, size):
    result = [(np.array([], dtype=np.int64), np.zeros((0, 3))) for _ in range(size)]
    daily = frame.assign(positive=frame.rating ** 2).groupby([column, 'timestamp']).agg(
        count=('rating', 'size'), total=('rating', 'sum'), positive=('positive', 'sum'))
    for key, group in daily.groupby(level=0, sort=False):
        result[key] = (group.index.get_level_values('timestamp').to_numpy(), group.to_numpy().cumsum(axis=0))
    return result


def lookup(history, ids, times):
    result = np.zeros((len(ids), 3), dtype=np.float32)
    ordering = np.argsort(ids, kind='stable')
    bounds = np.r_[0, np.flatnonzero(np.diff(ids[ordering])) + 1, len(ids)]
    for left, right in zip(bounds[:-1], bounds[1:]):
        rows = ordering[left:right]
        timestamps, cumulative = history[ids[rows[0]]]
        positions = np.searchsorted(timestamps, times[rows], side='left') - 1
        valid = positions >= 0
        raw = cumulative[positions[valid]]
        count = raw[:, 0]
        result[rows[valid], 0] = np.log1p(count)
        result[rows[valid], 1] = raw[:, 1] / count / 5
        result[rows[valid], 2] = np.sqrt(np.maximum(raw[:, 2] / count - (raw[:, 1] / count) ** 2, 0)) / 2
    return result


def examples(frame, shape, seed=42):
    rng = np.random.default_rng(seed)
    ordered = frame.sort_values(['timestamp', 'user', 'item'], kind='stable')
    negatives = []
    known = [set() for _ in range(shape[0])]
    user_ids = ordered.user.to_numpy()
    item_ids = ordered.item.to_numpy()
    timestamps = ordered.timestamp.to_numpy()
    boundaries = np.r_[0, np.flatnonzero(np.diff(timestamps)) + 1, len(ordered)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        for user, item in zip(user_ids[start:end], item_ids[start:end]):
            known[user].add(item)
        for user in user_ids[start:end]:
            item = int(rng.integers(shape[1]))
            while item in known[user]:
                item = int(rng.integers(shape[1]))
            negatives.append(item)
    users = np.tile(ordered.user.to_numpy(), 2)
    items = np.r_[ordered.item.to_numpy(), negatives]
    times = np.tile(ordered.timestamp.to_numpy(), 2)
    labels = np.r_[(ordered.rating >= 4).astype(float), np.zeros(len(ordered))].astype('float32')
    user_history = histories(frame, 'user', shape[0])
    item_history = histories(frame, 'item', shape[1])
    values = np.c_[lookup(user_history, users, times), lookup(item_history, items, times)]
    cutoff = int(frame.timestamp.max()) + 1
    user_values = lookup(user_history, np.arange(shape[0]), np.full(shape[0], cutoff))
    item_values = lookup(item_history, np.arange(shape[1]), np.full(shape[1], cutoff))
    return users, items, values, labels, user_values, item_values
