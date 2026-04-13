from collections import deque


class NStep:
    def __init__(self, n=3, gamma=0.99):
        self.n = n
        self.gamma = gamma
        self.buffer = deque()

    def push(self, transition):
        self.buffer.append(transition)

        if len(self.buffer) < self.n:
            return None

        R, ns, d = 0, None, False
        for i, (_, _, r, n_s, done) in enumerate(self.buffer):
            R += (self.gamma ** i) * r
            ns = n_s
            d = done
            if done:
                break

        s, a, _, _, _ = self.buffer[0]
        return s, a, R, ns, d

    def pop(self):
        self.buffer.popleft()