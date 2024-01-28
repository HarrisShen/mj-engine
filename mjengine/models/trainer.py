import numpy as np
from gymnasium import Env
from tqdm import tqdm

from mjengine.game import Game
from mjengine.models.agent.agent import Agent
from mjengine.models.utils import ReplayBuffer
from mjengine.player import make_player, Player
from mjengine.strategy import RLAgentStrategy


def train_on_policy(
        env: Env,
        agent: Agent,
        n_episodes: int,
        n_checkpoints: int,
        save_checkpoints: bool,
        evaluate: bool = False, **kwargs) -> tuple[list, list]:
    return_list, n_action_list = [], []
    n_division = n_checkpoints
    for i in range(n_division):
        with tqdm(total=int(n_episodes / n_division), desc=f'Iter. {i}') as pbar:
            for i_episode in range(int(n_episodes / n_division)):
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
                agent.update(**transition_dict)
                return_list.append(episode_return)
                n_action_list.append(episode_actions)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'epi': '%d' % (n_episodes / 10 * i + i_episode + 1),
                        'ret': '%.3f' % np.mean(return_list[-10:]),
                        'a r': '%.3f' % (np.sum(return_list[-10:]) / np.sum(n_action_list[-10:])),
                        'n a': '%.1f' % np.mean(n_action_list[-10:])
                    })
                pbar.update(1)
        if evaluate:
            eval_agent(agent, **kwargs)
    return return_list, n_action_list


def train_off_policy(
        env: Env,
        agent: Agent,
        replay_buffer: ReplayBuffer,
        n_episodes: int,
        min_size: int,
        batch_size: int,
        evaluate: bool = False, **kwargs) -> tuple[list, list]:
    return_list, n_action_list = [], []
    # best_action_return = float("-inf")
    n_division = 10
    for i in range(n_division):
        with tqdm(total=int(n_episodes / n_division), desc=f'Iter. {i}') as pbar:
            for i_episode in range(int(n_episodes / n_division)):
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
                    # best_action_return = max(best_action_return, float(reward))
                    # if info.get("chuck_tile"):  # Revise the reward if player chucks
                    #     for j in range(episode_actions):
                    #         if replay_buffer[-(1 + j)][1] == info["chuck_tile"]:
                    #             c_s, c_a, c_r, c_ns, c_d = replay_buffer[-(1 + j)]
                    #             replay_buffer[-(1 + j)] = (c_s, c_a, -128, c_ns, True)
                    #             episode_return -= c_r + 128
                    #             break
                    # Train the Q network until buffer reached minimal size
                    if len(replay_buffer) > min_size:
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
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'epi.': '%d' % (n_episodes / 10 * i + i_episode + 1),
                        'ret.': '%.3f' % np.mean(return_list[-10:]),
                        'a. r.': '%.3f' % (np.sum(return_list[-10:]) / np.sum(n_action_list[-10:])),
                        'n. a.': '%.1f' % np.mean(n_action_list[-10:])
                    })
                pbar.update(1)
        if evaluate:
            eval_agent(agent, **kwargs)
    return return_list, n_action_list


def eval_agent(
        agent: Agent,
        benchmark: str,
        round_limit: int | None = None,
        game_limit: int | None = None, **kwargs) -> None:
    players = [make_player(benchmark) for _ in range(3)]
    players.append(Player(RLAgentStrategy(agent)))
    game = Game(
        players=players,
        round_limit=round_limit,
        game_limit=game_limit,
        **kwargs)
    game.play()
    summary = game.players[3].summary(game.games)
    print(f"Eval. vs '{benchmark}': games={game_limit}, avg. score={summary['avg_score']:.4f}, "
          f"win%={summary['win_rate'] * 100:.3f}, s.w.%={summary['self_win_rate'] * 100:.3f}, "
          f"chuck%={summary['chuck_rate'] * 100:.3f}")
