import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from cli.inputs import numeric_input, unit_interval_input, confirm_inputs, bool_input
from mjengine.models.agent import Deterministic, DQN, PPO
from mjengine.models.env import MahjongEnv
from mjengine.models.trainer import train_off_policy, train_on_policy
from mjengine.models.utils import ReplayBuffer


def setup() -> dict:
    print("[Mahjong model - training setup]")

    agent_type = input("Agent type: ")
    if agent_type.lower() not in ["ppo", "dqn", "random", "analyzer", "value", "exp0", "exp1"]:
        exit(1)
    if agent_type.lower() == "ppo":
        return setup_ppo()
    if agent_type.lower() == "dqn":
        return setup_dqn()

    num_episodes = numeric_input("Number of episodes", int, default=500, min_val=1)
    return {"agent": agent_type, "num_episodes": num_episodes}


def setup_train() -> dict:
    n_episodes = numeric_input("Number of episodes", int, default=500, min_val=1)
    n_checkpoints = numeric_input("Number of check points", int, default=10, min_val=1)
    save_cp = bool_input("Save check points?", default=True)
    eval_params = {"evaluate": bool_input("Enable evaluation?", default=True)}
    if eval_params["evaluate"]:
        eval_params["benchmark"] = numeric_input("Benchmark strategy",
                                                 int, default=1, choice={1: "random", 2: "random1"})
        eval_params["game_limit"] = numeric_input("Number of games", int, default=128, min_val=1)
    return {
        "train_params": {
            "n_episodes": n_episodes,
            "n_checkpoints": n_checkpoints,
            "save_checkpoints": save_cp,
            **eval_params
        }
    }


def setup_ppo() -> dict:
    train_settings = setup_train()
    hidden_dim = numeric_input("Hidden dimension", int, default=128, min_val=1)
    actor_lr = unit_interval_input("Initial learning rate of actor", default=1e-4)
    critic_lr = unit_interval_input("Initial learning rate of critic", default=5e-3)
    lmbda = unit_interval_input("Lambda", default=0.9)
    gamma = unit_interval_input("Gamma", default=0.95)
    eps = unit_interval_input("Epsilon for clip", default=0.2)
    epochs = numeric_input("Number of epochs", int, default=10, min_val=1)

    device = "cpu"
    if torch.cuda.is_available():
        device = numeric_input("Device", int, default=1, choice={1: "cuda", 2: "cpu"})

    return confirm_inputs("Settings", {
        **train_settings,
        "agent": "PPO",
        "agent_params": {
            "hidden_dim": hidden_dim,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "lmbda": lmbda,
            "gamma": gamma,
            "eps": eps,
            "epochs": epochs,
            "device": device
        }
    }, setup)


def setup_dqn() -> dict:
    algorithm = numeric_input(
        description="Algorithm",
        dtype=int,
        default=1,
        choice={1: "DQN", 2: "DoubleDQN", 3: "DuelingDQN"}
    )

    num_episodes = numeric_input("Number of episodes", int, default=500, min_val=1)
    hidden_dim = numeric_input("Hidden dimension", int, default=128, min_val=1)
    lr = unit_interval_input("Learning rate", default=1e-4)
    gamma = unit_interval_input("Gamma", default=0.98)
    epsilon = unit_interval_input("Epsilon", default=0.05)
    target_update = numeric_input("Target update", int, default=10, min_val=1)
    buffer_size = numeric_input("Buffer size", int, default=100_000, min_val=1)
    minimal_size = numeric_input("Minimal size", int, default=500, min_val=1)
    batch_size = numeric_input("Batch size", int, default=64, min_val=1)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = numeric_input("Device", int, default=1, choice={1: "cuda", 2: "cpu"})

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
    }, setup)


def train(settings: dict) -> None:
    env = MahjongEnv()
    random.seed(0)
    np.random.seed(0)
    # env.seed(0)
    torch.manual_seed(0)

    # n_episodes = settings["n_episodes"]
    state_dim = 314
    action_dim = env.action_space.shape[0]

    if settings["agent"] == "PPO":
        agent = PPO(state_dim=state_dim, action_dim=action_dim, **settings["agent_params"])
        timestamp = datetime.strftime(datetime.utcnow(), "%y%m%d%H%M%S")
        model_name = f'PPO_{settings["agent_params"]["hidden_dim"]}_{timestamp}'
        return_list, n_action_list = train_on_policy(env, agent, **settings["train_params"])
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

        timestamp = datetime.strftime(datetime.utcnow(), "%y%m%d%H%M%S")
        algorithm = "default" if algorithm == "DQN" else algorithm
        model_name = f"DQN_{hidden_dim}_{algorithm}_{timestamp}"

        # return_list, n_action_list = train_off_policy(env, agent, n_episodes, replay_buffer, minimal_size, batch_size, True, benchmark="random", game_limit=128)
        return_list, n_action_list = train_off_policy(env, agent, replay_buffer, **settings["train_params"])
    else:
        agent = Deterministic(env.game, settings["agent"])

        timestamp = datetime.strftime(datetime.utcnow(), "%y%m%d%H%M%S")
        model_name = f"{agent.strategy}_{timestamp}"

        return_list, n_action_list = train_on_policy(env, agent, **settings["train_params"])

    model_dir = os.path.join("./trained_models/", model_name)
    agent.save(model_dir)
    df = pd.DataFrame({"episode_return": return_list, "n_action": n_action_list})
    df.to_csv(os.path.join(model_dir, "train_output.csv"))
    with open(os.path.join(model_dir, "training_settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    print(f'Training artifacts saved at "{os.path.abspath(model_dir)}"')


if __name__ == "__main__":
    train(setup())
