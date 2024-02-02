from abc import ABC, abstractmethod

import numpy as np
import torch


class Agent(ABC):
    def __init__(self, train: bool = True):
        self.train = train

        self.count = 0

    def step(self):
        self.count += 1

    @abstractmethod
    def take_action(self, state: np.ndarray, option: np.ndarray) -> int:
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        pass

    @abstractmethod
    def save(self, model_dir: str, checkpoint: int | None = None) -> str:
        pass

    @staticmethod
    @abstractmethod
    def restore(model_dir: str, device: torch.device, train: bool = False):
        pass
