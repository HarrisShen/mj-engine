import json
import os
from abc import ABC, abstractmethod

import numpy as np
import torch


class Agent(ABC):
    def __init__(self, on_policy: bool, device: str | torch.device, train: bool = True, **kwargs):
        self.on_policy = on_policy

        self.device = device
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

    @classmethod
    def restore(
            cls,
            model_path: str,
            device: str | torch.device,
            train: bool = False,
            checkpoint: int | None = None):
        if os.path.isdir(model_path):
            with open(os.path.join(model_path, "model_settings.json"), "r") as f:
                kwargs = json.load(f)
            obj = cls(device=device, train=train, **kwargs)
            state_file = "model_state.pt" if checkpoint is None else f"model_state_cp_{checkpoint}.pt"
            model_state = torch.load(os.path.join(model_path, state_file))
        else:
            with open(os.path.join(model_path, "../model_settings.json"), "r") as f:
                kwargs = json.load(f)
            obj = cls(device=device, train=train, **kwargs)
            model_state = torch.load(os.path.join(model_path))
        for k, v in model_state.items():
            if isinstance(v, dict):  # restore from state dict
                getattr(obj, k).load_state_dict(model_state[k])
            else:
                setattr(obj, k, model_state[k])
        return obj
