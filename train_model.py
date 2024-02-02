import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import scipy as sp
import torch

from cli.inputs import numeric_input, unit_interval_input, confirm_inputs, bool_input
from mjengine.models.agent import Deterministic, DQN, PPO
from mjengine.models.agent.sac import SAC
from mjengine.models.env import MahjongEnv
from mjengine.models.gail import GAIL
from mjengine.models.trainer import train_off_policy, train_on_policy, train_gail
from mjengine.models.utils import ReplayBuffer


def setup() -> dict:
    print("[Mahjong model - training setup]")

    agent_type = input("Agent/algorithm type: ").lower()
    if agent_type not in ["gail", "ppo", "sac", "dqn", "random", "analyzer", "value", "exp0", "exp1"]:
        exit(1)
    if agent_type not in ["gail", "ppo", "sac", "dqn"]:
        seed = numeric_input("Random seed", int, default=0)
        n_episode = numeric_input("Number of episodes", int, default=500, min_val=1)
        return {
            "agent": agent_type,
            "seed": seed,
            "train_params": {
                "n_episode": n_episode,
                "n_checkpoint": 10,
                "save_checkpoint": False
            }
        }
    train_settings = setup_train()
    if agent_type == "gail":
        agent_settings = setup_gail()
    elif agent_type == "ppo":
        agent_settings = setup_ppo()
    elif agent_type == "sac":
        agent_settings = setup_sac()
        train_settings["train_params"].update(agent_settings["train_params"])
        del agent_settings["train_params"]
    elif agent_type == "dqn":
        agent_settings = setup_dqn()
        train_settings["train_params"].update(agent_settings["train_params"])
        del agent_settings["train_params"]
    else:
        raise ValueError
    return confirm_inputs("Settings", {
        "agent": agent_type.upper(),
        **train_settings,
        **agent_settings
    }, setup)


def setup_train() -> dict:
    print("=" * 20 + " Training params " + "=" * 20)
    seed = numeric_input("Random seed", int, default=0)
    n_episode = numeric_input("Number of episodes", int, default=500, min_val=1)
    n_checkpoint = numeric_input("Number of check points", int, default=10, min_val=0)
    if n_checkpoint == 0:
        save_cp, eval_params = False, {"evaluate": False}
    else:
        save_cp = bool_input("Save check points?", default=False)
        eval_params = {"evaluate": bool_input("Enable evaluation?", default=False)}
    if eval_params["evaluate"]:
        eval_params["benchmark"] = numeric_input("Benchmark strategy",
                                                 int, default=1, choice={1: "random", 2: "random1"})
        eval_params["game_limit"] = numeric_input("Number of games", int, default=128, min_val=1)
        eval_params["seed"] = numeric_input("Random seed for game simulation: ", int, default=0)
    return {
        "seed": seed,
        "train_params": {
            "n_episode": n_episode,
            "n_checkpoint": n_checkpoint,
            "save_checkpoint": save_cp,
            **eval_params
        }
    }


def setup_gail() -> dict:
    print("=" * 20 + " GAIL params " + "=" * 20)
    gail_hidden_dim = numeric_input("Hidden dimension of discriminator", int, default=128, min_val=1)
    gail_lr = unit_interval_input("Learning rate of discriminator", default=1e-4)
    gail_params = {"hidden_dim": gail_hidden_dim, "lr": gail_lr}

    agent_settings = setup_ppo()
    gail_params["device"] = agent_settings["agent_params"]["device"]
    return {"gail_params": gail_params, **agent_settings}


def setup_ppo() -> dict:
    print("=" * 20 + " PPO params " + "=" * 20)
    hidden_dim = numeric_input("Hidden dimension", int, default=128, min_val=1)
    hidden_layer = numeric_input("Number of hidden layer", int, default=1, choice=[1, 2])
    actor_lr = unit_interval_input("Initial learning rate of actor", default=1e-4)
    critic_lr = unit_interval_input("Initial learning rate of critic", default=1e-3)
    lr_schedule = bool_input("Enable learning rate scheduling?", default=True)
    lmbda = unit_interval_input("Lambda", default=0.95)
    gamma = unit_interval_input("Gamma", default=0.98)
    eps = unit_interval_input("Epsilon for clip", default=0.2)
    epochs = numeric_input("Number of epochs", int, default=10, min_val=1)

    device = "cpu"
    if torch.cuda.is_available():
        device = numeric_input("Device", int, default=1, choice={1: "cuda", 2: "cpu"})

    return {
        "agent_params": {
            "hidden_dim": hidden_dim,
            "hidden_layer": hidden_layer,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "lr_schedule": lr_schedule,
            "lmbda": lmbda,
            "gamma": gamma,
            "eps": eps,
            "epochs": epochs,
            "device": device
        }
    }


def setup_sac() -> dict:
    print("=" * 20 + " SAC params " + "=" * 20)
    hidden_dim = numeric_input("Hidden dimension", int, default=128, min_val=1)
    actor_lr = unit_interval_input("Learning rate of actor", default=1e-3)
    critic_lr = unit_interval_input("Learning rate of critic", default=1e-2)
    alpha_lr = unit_interval_input("Learning rate of Alpha", default=1e-2)
    gamma = unit_interval_input("Gamma", default=0.98)
    tau = unit_interval_input("Tau (soft update param)", default=0.005)
    target_entropy = numeric_input("Entropy target", int, default=-1)
    buffer_setting = {"buffer_size": numeric_input("Buffer size", int, default=100_000, min_val=1)}
    minimal_size = numeric_input("Minimal size", int, default=500, min_val=1)
    batch_size = numeric_input("Batch size", int, default=64, min_val=1)

    device = "cpu"
    if torch.cuda.is_available():
        device = numeric_input("Device", int, default=1, choice={1: "cuda", 2: "cpu"})

    return {
        "train_params": {
            "minimal_size": minimal_size,
            "batch_size": batch_size
        },
        "buffer_params": buffer_setting,
        "agent_params": {
            "hidden_dim": hidden_dim,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "alpha_lr": alpha_lr,
            "gamma": gamma,
            "tau": tau,
            "target_entropy": target_entropy,
            "device": device
        }
    }


def setup_dqn() -> dict:
    print("=" * 20 + " DQN params " + "=" * 20)
    algorithm = numeric_input("Algorithm", int, default=1,
                              choice={1: "DQN", 2: "DoubleDQN", 3: "DuelingDQN"})
    hidden_dim = numeric_input("Hidden dimension", int, default=128, min_val=1)
    lr = unit_interval_input("Learning rate", default=1e-4)
    gamma = unit_interval_input("Gamma", default=0.98)
    epsilon = unit_interval_input("Epsilon", default=0.05)
    target_update = numeric_input("Target update", int, default=10, min_val=1)
    buffer_setting = {"buffer_size": numeric_input("Buffer size", int, default=100_000, min_val=1)}
    minimal_size = numeric_input("Minimal size", int, default=500, min_val=1)
    batch_size = numeric_input("Batch size", int, default=64, min_val=1)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = numeric_input("Device", int, default=1, choice={1: "cuda", 2: "cpu"})

    return {
        "train_params": {
            "minimal_size": minimal_size,
            "batch_size": batch_size
        },
        "buffer_params": buffer_setting,
        "agent_params": {
            "algorithm": algorithm,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "gamma": gamma,
            "epsilon": epsilon,
            "target_update": target_update,
            "device": device
        }
    }


def train(settings: dict) -> None:
    env = MahjongEnv()

    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)

    state_dim = len(env.game.to_numpy())
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    timestamp = datetime.strftime(datetime.utcnow(), "%y%m%d%H%M%S")
    if settings["agent"] == "GAIL":
        agent = PPO(state_dim=state_dim, action_dim=action_dim, **settings["agent_params"])
        gail = GAIL(agent, state_dim, action_dim, **settings["gail_params"])
        model_name = f'GAIL_PPO_{settings["agent_params"]["hidden_dim"]}_{timestamp}'
        model_dir = os.path.join("./trained_models/", model_name)
        expert_data = "./trained_models/exp1_240128183233/state_action_replay.npz"
        return_list, n_action_list = train_gail(env, agent, gail, expert_data, model_dir, **settings["train_params"])
    elif settings["agent"] == "PPO":
        agent = PPO(state_dim=state_dim, action_dim=action_dim, **settings["agent_params"])
        hidden_layer, hidden_dim = settings["agent_params"]["hidden_layer"], settings["agent_params"]["hidden_dim"]
        model_name = f'PPO_{hidden_layer}_{hidden_dim}_{timestamp}'
        model_dir = os.path.join("./trained_models/", model_name)
        return_list, n_action_list = train_on_policy(env, agent, model_dir, **settings["train_params"])
    elif settings["agent"] == "SAC":
        replay_buffer = ReplayBuffer(settings["buffer_params"]["buffer_size"])
        agent = SAC(state_dim=state_dim, action_dim=action_dim, **settings["agent_params"])
        model_name = f'SAC_{settings["agent_params"]["hidden_dim"]}_{timestamp}'
        model_dir = os.path.join("./trained_models/", model_name)
        return_list, n_action_list = train_off_policy(env, agent, replay_buffer, model_dir, **settings["train_params"])
    elif settings["agent"] == "DQN":
        replay_buffer = ReplayBuffer(settings["buffer_params"]["buffer_size"])
        agent = DQN(state_dim=state_dim, action_dim=action_dim, **settings["agent_params"])
        algorithm = settings["agent_params"]["algorithm"]
        algorithm = "default" if algorithm == "DQN" else algorithm
        model_name = f'DQN_{settings["agent_params"]["hidden_dim"]}_{algorithm}_{timestamp}'
        model_dir = os.path.join("./trained_models/", model_name)
        return_list, n_action_list = train_off_policy(env, agent, replay_buffer, model_dir, **settings["train_params"])
    else:
        replay_buffer = ReplayBuffer(None)
        agent = Deterministic(env.game, settings["agent"])
        model_name = f"{agent.strategy}_{timestamp}"
        model_dir = os.path.join("./trained_models/", model_name)
        return_list, n_action_list = train_off_policy(
            env, agent, replay_buffer, model_dir,
            minimal_size=int(1e9), batch_size=int(1e8), **settings["train_params"])
        replay_mtx = sp.sparse.csr_matrix(np.array([np.concatenate((s, [a])) for s, a, _, _, _ in replay_buffer],
                                                   dtype=int))
        sp.sparse.save_npz(os.path.join(model_dir, "state_action_replay.npz"), replay_mtx)

    df = pd.DataFrame({"episode_return": return_list, "n_action": n_action_list})
    df.to_csv(os.path.join(model_dir, "train_output.csv"))
    with open(os.path.join(model_dir, "training_settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    print(f'Training artifacts saved at "{os.path.abspath(model_dir)}"')


if __name__ == "__main__":
    train(setup())
