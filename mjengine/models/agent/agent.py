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
    def save(self, model_dir: str, checkpoint: int | None = None, best: bool = False) -> str:
        pass

    @classmethod
    def restore(
            cls,
            model_path: str,
            device: str | torch.device,
            train: bool = False,
            checkpoint: int | None = None, **kwargs):
        settings_path = "model_settings.json"
        if os.path.isfile(model_path):
            settings_path = os.path.join("..", settings_path)
        with open(os.path.join(model_path, settings_path), "r") as f:
            model_params = json.load(f)
        model_params.update(kwargs)
        obj = cls(device=device, train=train, **model_params)
        if os.path.isdir(model_path):
            state_file = "model_state.pt" if checkpoint is None else f"model_state_cp_{checkpoint}.pt"
            model_state = torch.load(os.path.join(model_path, state_file))
        else:
            model_state = torch.load(os.path.join(model_path))
        for k, v in model_state.items():
            if isinstance(v, dict):  # restore from state dict
                sd = model_state[k]
                sd_keys = list(sd.keys())
                for sdk in sd_keys:
                    new_sdk = sdk
                    new_sdk = new_sdk.replace("input", "layers.0")
                    new_sdk = new_sdk.replace("output", "layers.1")
                    if new_sdk != sdk:
                        sd[new_sdk] = sd.pop(sdk)
                getattr(obj, k).load_state_dict(sd)
                if k.endswith("optimizer"):
                    optimizer = getattr(obj, k)
                    for group in optimizer.param_groups:
                        group["lr"] = model_params[k.replace("optimizer", "lr")]
            else:
                setattr(obj, k, model_state[k])
        return obj
