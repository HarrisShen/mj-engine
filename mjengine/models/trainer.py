import os

import numpy as np
from gymnasium import Env
from scipy.sparse import load_npz
from tqdm import tqdm

from mjengine.game import Game
from mjengine.models.agent.agent import Agent
from mjengine.models.gail import GAIL
from mjengine.models.utils import ReplayBuffer
from mjengine.player import make_player, Player
from mjengine.strategy import RLAgentStrategy


def train_gail(
        env: Env,
        agent: Agent,
        gail: GAIL,
        expert_data: str,
        model_dir: str,
        n_episode: int,
        n_checkpoint: int,
        save_checkpoint: bool,
        evaluate: bool, **kwargs) -> tuple[list, list]:
    expert_data = load_npz(expert_data).toarray().astype(np.int64)
    expert_s, expert_a = expert_data[:, :-1], expert_data[:, -1]
    print(f"Loaded {expert_data.shape[0]} state-action pairs from expert")
    return_list, n_action_list = [], []
    for i in range(n_checkpoint):
        with tqdm(total=int(n_episode / n_checkpoint), desc=f'Iter. {i}') as pbar:
            for i_episode in range(int(n_episode / n_checkpoint)):
                episode_return, episode_actions = 0, 0
                state, info = env.reset()
                option = info["option"]
                done = False
                state_list = []
                action_list = []
                next_state_list = []
                done_list = []
                while not done:
                    action = agent.take_action(state, option)
                    next_state, reward, done, _, info = env.step(action)
                    state_list.append(state)
                    action_list.append(action)
                    next_state_list.append(next_state)
                    done_list.append(done)
                    option = info["option"]
                    state = info["next_player_state"]
                    episode_return += reward
                    episode_actions += 1
                return_list.append(episode_return)
                n_action_list.append(episode_actions)
                gail.learn(expert_s, expert_a,
                           np.array(state_list, dtype=np.int64),
                           np.array(action_list, dtype=np.int64),
                           next_state_list, done_list)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'epi': '%d' % (n_episode / 10 * i + i_episode + 1),
                        'ret': '%.3f' % np.mean(return_list[-10:]),
                        'r/a': '%.3f' % (np.sum(return_list[-10:]) / np.sum(n_action_list[-10:])),
                        'a/e': '%.1f' % np.mean(n_action_list[-10:])
                    })
                pbar.update(1)
        if evaluate:
            eval_agent(i, agent, **kwargs)
        if i + 1 < n_checkpoint and save_checkpoint:
            agent.save(model_dir, i + 1)
    agent.save(model_dir)
    return return_list, n_action_list


def train_on_policy(
        env: Env,
        agent: Agent,
        model_dir: str,
        n_episode: int,
        n_checkpoint: int,
        save_checkpoint: bool,
        evaluate: bool, **kwargs) -> tuple[list, list]:
    if n_checkpoint == 0:
        n_checkpoint = 1
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    return_list, n_action_list = [], []
    for i in range(n_checkpoint):
        with tqdm(total=int(n_episode / n_checkpoint), desc=f'Iter. {i}') as pbar:
            for i_episode in range(int(n_episode / n_checkpoint)):
                episode_return, episode_actions = 0, 0
                state, info = env.reset()
                option = info["option"]
                done = False
                transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": []}
                while not done:
                    action = agent.take_action(state, option)
                    next_state, reward, done, _, info = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    option = info["option"]
                    state = info["next_player_state"]
                    episode_return += reward
                    episode_actions += 1
                return_list.append(episode_return)
                n_action_list.append(episode_actions)
                agent.update(**transition_dict)
                with open(os.path.join(model_dir, "train_output.csv"), "a") as f:
                    f.write(",".join([
                        str(n_episode / 10 * i + i_episode + 1),
                        str(episode_return), str(episode_actions)]) + "\n")
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'epi': '%d' % (n_episode / 10 * i + i_episode + 1),
                        'ret': '%.3f' % np.mean(return_list[-10:]),
                        'r/a': '%.3f' % (np.sum(return_list[-10:]) / np.sum(n_action_list[-10:])),
                        'a/e': '%.1f' % np.mean(n_action_list[-10:])
                    })
                pbar.update(1)
        if evaluate:
            eval_agent(i, agent, **kwargs)
        if i + 1 < n_checkpoint and save_checkpoint:
            agent.save(model_dir, i + 1)
    agent.save(model_dir)
    return return_list, n_action_list


def train_off_policy(
        env: Env,
        agent: Agent,
        replay_buffer: ReplayBuffer,
        model_dir: str,
        n_episode: int,
        n_checkpoint: int,
        save_checkpoint: bool,
        minimal_size: int,
        batch_size: int,
        evaluate: bool = False, **kwargs) -> tuple[list, list]:
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, "train_output.csv"), "w") as f:
        f.write(",episode_return,n_action\n")
    return_list, n_action_list = [], []
    for i in range(n_checkpoint):
        with tqdm(total=int(n_episode / n_checkpoint), desc=f'Iter. {i}') as pbar:
            for i_episode in range(int(n_episode / n_checkpoint)):
                episode_return, episode_actions = 0, 0
                state, info = env.reset()
                option = info["option"]
                done = False
                while not done:
                    try:
                        action = agent.take_action(state, option)
                    except Exception as e:
                        print(f"[Error] Occurred at action {episode_actions}, episode {i_episode}")
                        raise e
                    # acting_player = env.game.acting_player
                    next_state, reward, done, _, info = env.step(action)
                    # print(f"acting for player {acting_player}, action {action}, reward {reward}")
                    replay_buffer.add(state, action, reward, next_state, done)
                    option = info["option"]
                    state = info["next_player_state"]
                    episode_return += reward
                    episode_actions += 1
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
                n_action_list.append(episode_actions)
                with open(os.path.join(model_dir, "train_output.csv"), "a") as f:
                    f.write(",".join([
                        str(n_episode / 10 * i + i_episode + 1),
                        str(episode_return), str(episode_actions)]) + "\n")
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'epi': '%d' % (n_episode / 10 * i + i_episode + 1),
                        'ret': '%.3f' % np.mean(return_list[-10:]),
                        'r/a': '%.3f' % (np.sum(return_list[-10:]) / np.sum(n_action_list[-10:])),
                        'a/e': '%.1f' % np.mean(n_action_list[-10:])
                    })
                pbar.update(1)
        if evaluate:
            eval_agent(i, agent, **kwargs)
        if i + 1 < n_checkpoint and save_checkpoint:
            agent.save(model_dir, i + 1)
    agent.save(model_dir)
    replay_buffer.save(model_dir, compression="bz2")
    return return_list, n_action_list


def eval_agent(
        index: int,
        agent: Agent,
        benchmark: str,
        game_limit: int | None = None, **kwargs) -> None:
    agent_train_cp = agent.train
    agent.train = False
    players = [make_player(benchmark) for _ in range(3)]
    players.append(Player(RLAgentStrategy(agent)))
    game = Game(
        players=players,
        game_limit=game_limit,
        **kwargs)
    with tqdm(total=int(game_limit), desc=f'Eval. {index}') as pbar:
        for i_game in range(game_limit):
            game.play(games=1)
            if (i_game + 1) % 10 == 0:
                summary = game.players[3].summary(game.games)
                pbar.set_postfix({
                    "vs": benchmark,
                    "s/g": f"{summary['avg_score']:.3f}",
                    "win%": f"{summary['win_rate'] * 100:.1f}",
                    "s.w%": f"{summary['self_win_rate'] * 100:.1f}",
                    "c%": f"{summary['chuck_rate'] * 100:.1f}"
                })
            pbar.update(1)
    agent.train = agent_train_cp
