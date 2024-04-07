import bz2
import os
import pickle
import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)
        # self.r = random.Random()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]

    def __setitem__(self, key, value):
        self.buffer[key] = value

    def __iter__(self):
        yield from self.buffer

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def extend(self, states, actions, rewards, next_states, dones):
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def last_n(self, n):
        return [self.buffer[-(n - i)] for i in range(n)]

    def save(self, model_dir, compression=None):
        filepath = os.path.join(model_dir, "replay_buffer.pkl")
        if compression is None:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
            return
        if compression == "bz2":
            with bz2.open(filepath + ".bz2", "wb", compresslevel=9) as f:
                data = pickle.dumps(self)
                f.write(data)
            return
        raise ValueError(f"Unsupported compression method: {compression}")

    @staticmethod
    def restore(model_dir):
        filepath = os.path.join(model_dir, "replay_buffer.pkl")
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                rb = pickle.load(f)
            return rb
        if os.path.isfile(filepath + ".bz2"):
            with bz2.open(filepath + ".bz2", "rb", compresslevel=9) as f:
                rb = pickle.load(f)
            return rb
        raise FileNotFoundError(f"File of replay buffer not found in {model_dir}")
