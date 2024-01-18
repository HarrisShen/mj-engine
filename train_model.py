import json
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mjengine.models.agents import DQN, DQN_ALGORITHMS
from mjengine.models.env import MahjongEnv
from mjengine.models.utils import ReplayBuffer


def number_input(
        prompt: str, 
        num_type: type,
        default: int | float,
        min_val: int | float = None, 
        max_val: int | float = None,
        choice: list | None = None,
        retry: int = 0) -> int:
    """Ask user for a number input, retry if invalid input is given"""
    if (min_val is not None and default < min_val) or (max_val is not None and default > max_val):
        raise ValueError(f"Default value {default} is not in range ({min_val}, {max_val})")
    if choice is not None:
        if default not in choice:
            raise ValueError(f"Default value {default} is not in choice {choice}")
        if min_val is not None and min_val not in choice:
            raise ValueError(f"Cannot set range ('min_val' and 'max_val') when 'choice' is given")
    
    for _ in range(retry + 1):
        try:
            num = num_type(input(prompt))
            if min_val is not None and num < min_val:
                raise ValueError
            if max_val is not None and num > max_val:
                raise ValueError
            if choice is not None and num not in choice:
                raise ValueError
            return num
        except ValueError:
            prompt = f"Invalid input, please enter a number"
            if min_val is not None:
                prompt += f" >= {min_val}"
            if max_val is not None:
                prompt += f" <= {max_val}"
            if choice is not None:
                prompt += f" in {choice}"
    print(f"Using default value {default}")
    return default


def setup() -> dict:
    print("[Mahjong model - training setup]")

    algorithm = number_input(
        prompt="Algorithm - 1. 'DQN' (default), 2. 'DoubleDQN': 3. 'DuelingDQN': ",
        num_type=int,
        default=1,
        choice=[1, 2, 3]
    )
    algorithm = DQN_ALGORITHMS[algorithm - 1]

    num_episodes = number_input(
        prompt="Number of episodes (default 500): ",
        num_type=int,
        default=500,
        min_val=1
    )

    hidden_dim = number_input(
        prompt="Hidden dimension (default 128): ",
        num_type=int,
        default=128,
        min_val=1
    )

    lr = number_input(
        prompt="Learning rate (default 2e-3): ",
        num_type=float,
        default=2e-3,
        min_val=0,
        max_val=1
    )

    gamma = number_input(
        prompt="Gamma (default 0.98): ",
        num_type=float,
        default=0.98,
        min_val=0,
        max_val=1
    )

    epsilon = number_input(
        prompt="Epsilon (default 0.01): ",
        num_type=float,
        default=0.01,
        min_val=0,
        max_val=1
    )

    target_update = number_input(
        prompt="Target update (default 10): ",
        num_type=int,
        default=10,
        min_val=1
    )

    buffer_size = number_input(
        prompt="Buffer size (default 100,000): ",
        num_type=int,
        default=100_000,
        min_val=1
    )

    minimal_size = number_input(
        prompt="Minimal size (default 500): ",
        num_type=int,
        default=500,
        min_val=1
    )

    batch_size = number_input(
        prompt="Batch size (default 64): ",
        num_type=int,
        default=64,
        min_val=1
    )
    
    device = 2
    if torch.cuda.is_available():
        device = number_input(
            prompt="Device - 1. 'cuda' (default), 2. 'cpu': ",
            num_type=int,
            default=1,
            choice=[1, 2]
        )
    device = "cuda" if device == 1 else "cpu"

    settings = {
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
    }

    print(f"Settings: \n{json.dumps(settings, indent=2)}")
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
    return settings


def train(settings: dict) -> None:
    num_episodes = settings["num_episodes"]
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

    env = MahjongEnv()
    random.seed(0)
    np.random.seed(0)
    # env.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = 309
    action_dim = env.action_space.shape[0]
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, algorithm)

    return_list, action_return = [], []
    best_action_return = float("-inf")
    n_division = 10
    for i in range(n_division):
        with tqdm(total=int(num_episodes / n_division), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / n_division)):
                episode_return, episode_actions = 0, 0
                state, info = env.reset()
                option = info["option"]
                done = False
                while not done:
                    action = agent.take_action(state, option)  # testing - option ignored
                    next_state, reward, done, _, info = env.step(action)
                    # print(f"acting for player {env.game.acting_player}, action {action}, reward {reward}")
                    replay_buffer.add(state, action, reward, next_state, done)
                    option = info["option"]
                    state = info["next_player_state"]
                    episode_return += reward
                    episode_actions += 1
                    best_action_return = max(best_action_return, reward)
                    if info.get("chuck_tile"):  # Revise the reward if player chucks
                        for j in range(episode_actions):
                            if replay_buffer[-(1 + j)][1] == info["chuck_tile"]:
                                c_s, c_a, c_r, c_ns, c_d = replay_buffer[-(1 + j)]
                                replay_buffer[-(1 + j)] = (c_s, c_a, -128, c_ns, True)
                                episode_return -= c_r + 128
                                break
                    # Train the Q network until buffer reached minimal size
                    if len(replay_buffer) > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            "states": b_s,
                            "actions": b_a,
                            "next_states": b_ns,
                            "rewards": b_r,
                            "dones": b_d
                        }
                        agent.update(**transition_dict)
                return_list.append(episode_return)
                action_return.append(episode_return / episode_actions)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'epi.': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'ret.': '%.3f' % np.mean(return_list[-10:]),
                        'act. ret.': '%.3f' % np.mean(action_return[-10:]),
                        'b. a.': '%d' % best_action_return
                    })
                pbar.update(1)

    model_dir = agent.save("./trained_models/")
    df = pd.DataFrame({"episode_return": return_list, "action_return": action_return})
    df.to_csv(os.path.join(model_dir, "train_output.csv"))
    with open(os.path.join(model_dir, "training_settings.json"), "w") as f:
        json.dump(settings, f, indent=2)


if __name__ == "__main__":
    settings = setup()
    train(settings)
