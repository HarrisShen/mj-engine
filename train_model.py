import json
import os
import random
from datetime import datetime

import numpy as np
import scipy as sp
import torch

from cli.inputs import numeric_input, unit_interval_input, confirm_inputs, bool_input
from mjengine.models.agent import Deterministic, DQN, PPO
from mjengine.models.agent.sac import SAC
from mjengine.models.env import MahjongEnv
from mjengine.models.gail import GAIL
from mjengine.models.trainer import train_off_policy, train_on_policy, train_gail, Trainer
from mjengine.models.utils import ReplayBuffer


def setup() -> dict:
    print("[Mahjong model - training setup]")

    mode = input("Training mode ([N]ew or [L]oad): ").lower()
    if mode == "l":
        return setup_load()
    if mode == "n":
        return setup_new()
    else:
        exit(1)


def setup_new():
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
    else:
        raise ValueError
    return confirm_inputs("Settings", {
        "agent": agent_type.upper(),
        **train_settings,
        **agent_settings
    }, setup)


def setup_load() -> dict:
    model_dir = input("Load from dir: ")
    if not os.path.isdir(model_dir):
        print(f'"{model_dir}" is not a valid directory')
        exit(1)
    model_name = os.path.split(model_dir)[-1]
    agent_type = model_name.split("_")[0].lower()
    if agent_type not in ["ppo", "dqn"]:
        print(f"Unsupported model type")
        exit(1)

    with open(os.path.join(model_dir, "training_settings.json"), "r") as f:
        print(f"training_settings: {f.read()}")
    train_settings = setup_train()
    if agent_type == "ppo":
        agent_settings = setup_ppo()
    else:
        agent_settings = setup_dqn()
    settings = {
        "agent": model_name.split("_")[0],
        **train_settings
    }
    for k, v in agent_settings.items():
        if k in settings:
            settings[k].update(v)
        else:
            settings[k] = v
    settings = confirm_inputs("Settings", settings, setup)
    return {**settings, "load_from": model_dir}


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
        eval_params["game_limit"] = numeric_input("Number of games", int, default=1000, min_val=1)
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
    hidden_dim = numeric_input("Hidden dimension", int, default=256, min_val=1)
    hidden_layer = numeric_input("Number of hidden layer", int, default=1, min_val=1)
    actor_lr = unit_interval_input("Initial learning rate of actor", default=1e-4)
    critic_lr = unit_interval_input("Initial learning rate of critic", default=1e-3)
    lr_schedule = bool_input("Enable learning rate scheduling?", default=False)
    clip_grad = bool_input("Enable gradient clipping?", default=False)
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
            "clip_grad": clip_grad,
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
    ###################################################################################
    if algorithm == 3:
        print("Error: DuelingDQN is temporarily disabled.")
        exit(1)
    ###################################################################################
    hidden_dim = numeric_input("Hidden dimension", int, default=256, min_val=1)
    hidden_layer = numeric_input("Number of hidden layer", int, default=1, min_val=1)
    lr = unit_interval_input("Learning rate", default=1e-4)
    gamma = unit_interval_input("Gamma", default=0.98)
    epsilon = unit_interval_input("Epsilon", default=0.05)
    target_update = numeric_input("Target update", int, default=10, min_val=1)
    buffer_cap = numeric_input("Buffer capacity", int, default=100_000, min_val=1)
    minimal_size = numeric_input("Minimal size", int, default=500, min_val=1)
    batch_size = numeric_input("Batch size", int, default=64, min_val=1)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = numeric_input("Device", int, default=1, choice={1: "cuda", 2: "cpu"})

    return {
        "off_policy_params": {
            "buffer_cap": buffer_cap,
            "minimal_size": minimal_size,
            "batch_size": batch_size
        },
        "agent_params": {
            "algorithm": algorithm,
            "hidden_dim": hidden_dim,
            "hidden_layer": hidden_layer,
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
    # if settings["agent"] == "GAIL":
    #     agent = PPO(state_dim=state_dim, action_dim=action_dim, **settings["agent_params"])
    #     gail = GAIL(agent, state_dim, action_dim, **settings["gail_params"])
    #     model_name = f'GAIL_PPO_{settings["agent_params"]["hidden_dim"]}_{timestamp}'
    #     model_dir = os.path.join("./trained_models/", model_name)
    #     expert_data = "./trained_models/exp1_240128183233/state_action_replay.npz"
    #     train_gail(env, agent, gail, expert_data, model_dir, **settings["train_params"])
    if settings["agent"] == "PPO":
        if settings.get("load_from"):
            agent = PPO.restore(settings["load_from"], settings["agent_params"]["device"], train=True)
            for k, v in settings["agent_params"].items():
                if k == "device":
                    continue
                setattr(agent, k, v)
            agent.actor_optimizer.param_groups[0]["lr"] = settings["agent_params"]["actor_lr"]
            agent.critic_optimizer.param_groups[0]["lr"] = settings["agent_params"]["critic_lr"]
        else:
            agent = PPO(state_dim=state_dim, action_dim=action_dim, **settings["agent_params"])
        hidden_layer, hidden_dim = settings["agent_params"]["hidden_layer"], settings["agent_params"]["hidden_dim"]
        model_name = f'PPO_{hidden_layer}_{hidden_dim}_{timestamp}'
    elif settings["agent"] == "SAC":
        replay_buffer = ReplayBuffer(settings["buffer_params"]["buffer_size"])
        agent = SAC(state_dim=state_dim, action_dim=action_dim, **settings["agent_params"])
        model_name = f'SAC_{settings["agent_params"]["hidden_dim"]}_{timestamp}'
        # model_dir = os.path.join("./trained_models/", model_name)
        # train_off_policy(env, agent, replay_buffer, model_dir, **settings["train_params"])
    elif settings["agent"] == "DQN":
        if settings.get("load_from"):
            # replay_buffer = ReplayBuffer.restore(settings["load_from"])
            agent = DQN.restore(settings["load_from"], settings["agent_params"]["device"], train=True)
            for k, v in settings["agent_params"].items():
                if k == "device":
                    continue
                setattr(agent, k, v)
        else:
            agent = DQN(state_dim=state_dim, action_dim=action_dim, **settings["agent_params"])
        hidden_layer, hidden_dim = settings["agent_params"]["hidden_layer"], settings["agent_params"]["hidden_dim"]
        algorithm = settings["agent_params"]["algorithm"]
        algorithm = "default" if algorithm == "DQN" else algorithm
        model_name = f'DQN_{hidden_layer}_{hidden_dim}_{algorithm}_{timestamp}'
    else:
        agent = Deterministic(env.game, settings["agent"])
        model_name = f"{agent.strategy}_{timestamp}"
        settings["off_policy_params"] = {
            "buffer_cap": None,
            "minimal_size": 1e9,
            "batch_size": 1e8
        }

    model_dir = os.path.join("./trained_models/", model_name)
    trainer = Trainer(env, agent, model_dir, **settings["train_params"])
    if agent.on_policy:
        trainer.train_on_policy()
    else:
        trainer.prepare_off_policy(**settings["off_policy_params"])
        trainer.train_off_policy()
    with open(os.path.join(model_dir, "training_settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    if isinstance(agent, Deterministic):
        replay_buffer = trainer.replay_buffer
        replay_mtx = np.array([np.concatenate((s, [a])) for s, a, _, _, _ in replay_buffer], dtype=int)
        replay_mtx = sp.sparse.csr_matrix(replay_mtx)
        sp.sparse.save_npz(os.path.join(model_dir, "state_action_replay.npz"), replay_mtx)
    print(f'Training artifacts saved at "{os.path.abspath(model_dir)}"')


def save_settings(settings, out_dir):
    with open(os.path.join(out_dir, "training_settings.json"), "w") as f:
        json.dump(settings, f, indent=2)


if __name__ == "__main__":
    training_settings = setup()
    trainer = Trainer.from_settings(training_settings, out_dir="./trained_models/")
    trainer.run()
    if isinstance(trainer.agent, Deterministic):
        replay_buffer = trainer.replay_buffer
        replay_mtx = np.array([np.concatenate((s, [a])) for s, a, _, _, _ in replay_buffer], dtype=int)
        replay_mtx = sp.sparse.csr_matrix(replay_mtx)
        sp.sparse.save_npz(os.path.join(trainer.model_dir, "state_action_replay.npz"), replay_mtx)
        print(f"Saved {replay_mtx.shape[0]} state-action pairs")
