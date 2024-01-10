import random

import numpy as np
import torch
from tqdm import tqdm

from mjengine.models.agents import DQN
from mjengine.models.env import MahjongEnv
from mjengine.models.utils import ReplayBuffer


def main():
    num_episodes = 500
    hidden_dim = 128
    lr, gamma, epsilon = 2e-3, 0.98, 0.01
    target_update = 10
    buffer_size = 100_000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = MahjongEnv()
    random.seed(0)
    np.random.seed(0)
    # env.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = 309
    action_dim = env.action_space.shape[0]
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

    return_list = []
    n_division = 10
    for i in range(n_division):
        with tqdm(total=int(num_episodes / n_division), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / n_division)):
                episode_return = 0
                state, info = env.reset()
                option = info["option"]
                done = False
                while not done:
                    action = agent.take_action(state, option)
                    next_state, reward, done, _, info = env.step(action)
                    # print(f"acting for player {env.game.acting_player}, action {action}, reward {reward}")
                    replay_buffer.add(state, action, reward, next_state, done)
                    option = info["option"]
                    state = info["next_player_state"]
                    episode_return += reward
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
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)


if __name__ == "__main__":
    main()
