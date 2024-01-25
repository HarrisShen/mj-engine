import json
import os
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mjengine.models.agents import DQN, DQN_ALGORITHMS, Deterministic, PPO
from mjengine.models.env import MahjongEnv
from mjengine.models.trainer import train_off_policy, train_on_policy
from mjengine.models.utils import ReplayBuffer


def value_input(
        description: str,
        dtype: type,
        default: int | float,
        min_val: int | float = None, 
        max_val: int | float = None,
        choice: list | dict | None = None,
        retry: int = 0) -> Any:
    """Ask user for a number as input, retry if invalid input is given"""
    if (min_val is not None and default < min_val) or (max_val is not None and default > max_val):
        raise ValueError(f"Default value {default} is not in range ({min_val}, {max_val})")
    if choice is not None:
        if default not in choice:
            raise ValueError(f"Default value {default} is not in choice {choice}")
        if min_val is not None and min_val not in choice:
            raise ValueError(f"Cannot set range ('min_val' and 'max_val') when 'choice' is given")

    if isinstance(choice, dict):
        prompt = f"{description} - "
        prompt += ", ".join([str(k) + ". " + str(v) + ("*" if default == k else "") for k, v in choice.items()])
        prompt += ": "
    else:
        prompt = f"{description} (default: {default}): "
    for _ in range(retry + 1):
        try:
            num = dtype(input(prompt))
            if min_val is not None and num < min_val:
                raise ValueError
            if max_val is not None and num > max_val:
                raise ValueError
            if choice is not None and num not in choice:
                raise ValueError
            return choice[num] if isinstance(choice, dict) else num
        except ValueError:
            prompt = f"Invalid input, please enter a number"
            if min_val is not None:
                prompt += f" >= {min_val}"
            if max_val is not None:
                prompt += f" <= {max_val}"
            if choice is not None:
                prompt += f" in {choice}"
    print(f"Using default value {default}")
    return choice[default] if isinstance(choice, dict) else default


def unit_interval_input(
        description: str,
        default: int | float,
        retry: int = 0) -> float:
    return value_input(description, float, default, min_val=0, max_val=1, retry=retry)


def setup() -> dict:
    print("[Mahjong model - training setup]")

    agent_type = input("Agent type: ")
    if agent_type not in ["PPO", "DQN", "random", "analyzer", "value", "exp0", "exp1"]:
        exit(1)
    if agent_type == "PPO":
        return setup_ppo()
    if agent_type == "DQN":
        return setup_dqn()

    num_episodes = value_input("Number of episodes", int, default=500, min_val=1)
    return {"agent": agent_type, "num_episodes": num_episodes}


def confirm_inputs(name: str, content: dict) -> dict:
    print(f"{name}: \n{json.dumps(content, indent=2)}")
    while True:
        confirm = input("Confirm? (Y - yes, n - no, e - abort & exit): ")
        if confirm in ["Y", "y", ""]:
            break
        elif confirm in ["N", "n"]:
            return setup()
        elif confirm in ["E", "e"]:
            print("Cancelled")
            exit(0)
        else:
            print("Invalid input, please enter Y, n or e")
    return content


def setup_ppo() -> dict:
    num_episodes = value_input("Number of episodes", int, default=500, min_val=1)
    hidden_dim = value_input("Hidden dimension", int, default=128, min_val=1)
    actor_lr = unit_interval_input("Initial learning rate of actor", default=1e-4)
    critic_lr = unit_interval_input("Initial learning rate of critic", default=5e-3)
    lmbda = unit_interval_input("Lambda", default=0.9)
    epochs = value_input("Number of epochs", int, default=10, min_val=1)
    eps = unit_interval_input("Epsilon for clip", default=0.2)
    gamma = unit_interval_input("Gamma", default=0.9)

    device = "cpu"
    if torch.cuda.is_available():
        device = value_input("Device", int, default=1, choice={1: "cuda", 2: "cpu"})

    return confirm_inputs("Settings", {
        "agent": "PPO",
        "num_episodes": num_episodes,
        "hidden_dim": hidden_dim,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "lmbda": lmbda,
        "epochs": epochs,
        "gamma": gamma,
        "eps": eps,
        "device": device
    })


def setup_dqn() -> dict:
    algorithm = value_input(
        description="Algorithm",
        dtype=int,
        default=1,
        choice={1: "DQN", 2: "DoubleDQN", 3: "DuelingDQN"}
    )

    num_episodes = value_input("Number of episodes", int, default=500, min_val=1)
    hidden_dim = value_input("Hidden dimension", int, default=128, min_val=1)
    lr = unit_interval_input("Learning rate", default=1e-4)
    gamma = unit_interval_input("Gamma", default=0.98)
    epsilon = unit_interval_input("Epsilon", default=0.05)
    target_update = value_input("Target update", int, default=10, min_val=1)
    buffer_size = value_input("Buffer size", int, default=100_000, min_val=1)
    minimal_size = value_input("Minimal size", int, default=500, min_val=1)
    batch_size = value_input("Batch size", int, default=64, min_val=1)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = value_input("Device", int, default=1, choice={1: "cuda", 2: "cpu"})

    return confirm_inputs("Settings", {
        "agent": "DQN",
        "algorithm": algorithm,
        "num_episodes": num_episodes,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "gamma": gamma,
        "epsilon": epsilon,
        "target_update": target_update,
        "buffer_size": buffer_size,
        "minimal_size": minimal_size,
        "batch_size": batch_size,
        "device": device
    })


def train(settings: dict) -> None:
    env = MahjongEnv()
    random.seed(0)
    np.random.seed(0)
    # env.seed(0)
    torch.manual_seed(0)

    num_episodes = settings["num_episodes"]
    state_dim = 314
    action_dim = env.action_space.shape[0]

    if settings["agent"] == "PPO":
        hidden_dim = settings["hidden_dim"]
        actor_lr = settings["actor_lr"]
        critic_lr = settings["critic_lr"]
        lmbda = settings["lmbda"]
        epochs = settings["epochs"]
        eps = settings["eps"]
        gamma = settings["gamma"]
        device = torch.device(settings["device"])

        agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

        return_list, n_action_list = train_on_policy(env, agent, num_episodes)
    elif settings["agent"] == "DQN":
        hidden_dim = settings["hidden_dim"]
        lr = settings["lr"]
        gamma = settings["gamma"]
        epsilon = settings["epsilon"]
        target_update = settings["target_update"]
        buffer_size = settings["buffer_size"]
        minimal_size = settings["minimal_size"]
        batch_size = settings["batch_size"]
        device = torch.device(settings["device"])
        algorithm = settings["algorithm"]

        replay_buffer = ReplayBuffer(buffer_size)
        agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, algorithm)

        return_list, n_action_list = train_off_policy(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)
    else:
        replay_buffer = ReplayBuffer(1000)
        minimal_size = 100
        batch_size = 1
        agent = Deterministic(env.game, settings["agent"])

        return_list, n_action_list = train_off_policy(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

    model_dir = agent.save("./trained_models/")
    df = pd.DataFrame({"episode_return": return_list, "n_action": n_action_list})
    df.to_csv(os.path.join(model_dir, "train_output.csv"))
    with open(os.path.join(model_dir, "training_settings.json"), "w") as f:
        json.dump(settings, f, indent=2)


if __name__ == "__main__":
    settings = setup()
    train(settings)
