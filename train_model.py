import json
import os

import torch

from cli.inputs import numeric_input, unit_interval_input, confirm_inputs, bool_input
from mjengine.models.trainer import Trainer
from mjengine.models.utils import LATEST_ENCODING_VERSION


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
    if agent_type not in ["gail", "ppo", "sac", "dqn", "random", "random1", "analyzer", "value", "exp0", "exp1"]:
        exit(1)

    env_settings = setup_env()

    if agent_type not in ["gail", "ppo", "sac", "dqn"]:
        seed = numeric_input("Random seed", int, default=0)
        n_episode = numeric_input("Number of episodes", int, default=500, min_val=1)
        return {
            "agent": agent_type,
            **env_settings,
            "seed": seed,
            "train_params": {
                "n_episode": n_episode,
                "n_checkpoint": 10,
                "save_checkpoint": False
            },
            "off_policy_params": {
                "buffer_cap": None,
                "minimal_size": int(1e9),
                "batch_size": int(1e8)
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
        **env_settings,
        **train_settings,
        **agent_settings
    }, setup)


def setup_load() -> dict:
    model_path = input("Load from dir or model state file: ")
    if os.path.isfile(model_path):
        model_dir = os.path.split(model_path)[0]
    elif os.path.isdir(model_path):
        model_dir = model_path
    else:
        raise FileNotFoundError(f'"{model_path}" is not a file or a directory')
    model_name = os.path.split(model_dir)[1]
    agent_type = model_name.split("_")[0].lower()
    if agent_type not in ["ppo", "dqn", "sac"]:
        print(f"Unsupported model type")
        exit(1)

    resume = False
    if os.path.isdir(os.path.join(model_dir, ".bp")):
        resume = bool_input("Break point found. Resume trainer?", default=True)

    with open(os.path.join(model_dir, "training_settings.json"), "r") as f:
        prev_settings = json.load(f)
        print(f"training_settings: {json.dumps(prev_settings, indent=2)}")
    if resume:
        return {"resume": True, "load_from": model_dir}

    env_settings = {"env_params": prev_settings["env_params"]}
    train_settings = setup_train()
    if agent_type == "ppo":
        agent_settings = setup_ppo(load_settings=prev_settings)
    else:
        agent_settings = setup_dqn(load_settings=prev_settings)
    settings = {
        "agent": model_name.split("_")[0],
        **env_settings,
        **train_settings
    }
    for k, v in agent_settings.items():
        if k in settings:
            settings[k].update(v)
        else:
            settings[k] = v
    settings = confirm_inputs("Settings", settings, setup)
    return {**settings, "load_from": model_path}


def setup_env() -> dict:
    print("=" * 20 + " Environment params " + "=" * 20)
    version = input(f"Game state encoding version (default {LATEST_ENCODING_VERSION}): ")
    if version not in ["0.1.1", "0.1.2", "0.2.0", "0.2.0a"]:
        print(f"Using default value {LATEST_ENCODING_VERSION}")
        version = LATEST_ENCODING_VERSION
    return {"env_params": {"encoding_version": version}}


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


def setup_ppo(load_settings: dict | None = None) -> dict:
    print("=" * 20 + " PPO params " + "=" * 20)
    if load_settings is None:
        hidden_dim = numeric_input("Hidden dimension", int, default=256, min_val=1)
        hidden_layer = numeric_input("Number of hidden layer", int, default=1, min_val=1)
    else:
        hidden_dim = load_settings["agent_params"]["hidden_dim"]
        hidden_layer = load_settings["agent_params"]["hidden_layer"]
        print(f"Hidden dimension: {hidden_dim}")
        print(f"Number of hidden layer: {hidden_layer}")
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


def setup_dqn(load_settings: dict | None = None) -> dict:
    print("=" * 20 + " DQN params " + "=" * 20)
    if load_settings is None:
        algorithm = numeric_input("Algorithm", int, default=1,
                                  choice={1: "DQN", 2: "DoubleDQN", 3: "DuelingDQN"})
        hidden_dim = numeric_input("Hidden dimension", int, default=256, min_val=1)
        hidden_layer = numeric_input("Number of hidden layer", int, default=1, min_val=1)
    else:
        algorithm = load_settings["agent_params"]["algorithm"]
        hidden_dim = load_settings["agent_params"]["hidden_dim"]
        hidden_layer = load_settings["agent_params"]["hidden_layer"]
        print(f"Algorithm: {algorithm}")
        print(f"Hidden dimension: {hidden_dim}")
        print(f"Number of hidden layer: {hidden_layer}")
    lr = unit_interval_input("Learning rate", default=1e-4)
    gamma = unit_interval_input("Gamma", default=0.98)
    eps_start = unit_interval_input("Epsilon at start", default=0.99)
    eps_end = unit_interval_input("Final Epsilon", default=0.05)
    eps_decay = numeric_input("Steps for Epsilon decay", int, default=1000)
    target_update = numeric_input("Target update", int, default=10_000, min_val=1)
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
            "eps_start": eps_start,
            "eps_end": eps_end,
            "eps_decay": eps_decay,
            "target_update": target_update,
            "device": device
        }
    }


if __name__ == "__main__":
    default_dir = "./trained_models/"
    out_dir = input(f"Output dir (default {os.path.abspath(default_dir)}): ")
    if not out_dir:
        out_dir = default_dir
    training_settings = setup()
    if "load_from" in training_settings:
        if training_settings.get("resume"):
            trainer = Trainer.restore(training_settings["load_from"])
        else:
            trainer = Trainer.from_settings(training_settings, out_dir=out_dir)
    else:
        trainer = Trainer.from_settings(training_settings, out_dir=out_dir)
    trainer.run()
