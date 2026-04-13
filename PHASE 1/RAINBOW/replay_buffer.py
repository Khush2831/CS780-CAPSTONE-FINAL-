import numpy as np


class PER:
    def __init__(self, capacity=200000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.pos = 0

    def push(self, s, a, r, ns, d):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((s, a, r, ns, d))
        else:
            self.buffer[self.pos] = (s, a, r, ns, d)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()

        idx = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in idx]

        weights = (len(self.buffer) * probs[idx]) ** (-beta)
        weights /= weights.max()

        s, a, r, ns, d = zip(*samples)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d), idx, weights

    def update(self, idx, errors):
        for i, e in zip(idx, errors):
            self.priorities[i] = abs(e) + 1e-5