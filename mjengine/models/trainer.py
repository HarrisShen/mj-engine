import json
import os
import pickle
import random
import shutil
import signal
import sys
from datetime import datetime

import numpy as np
import torch
from gymnasium import Env
from scipy.sparse import csr_matrix, load_npz, save_npz
from tqdm import tqdm

from mjengine.game import Game
from mjengine.models.agent import PPO, DQN, SAC, Deterministic
from mjengine.models.agent.agent import Agent
from mjengine.models.env import MahjongEnv
from mjengine.models.gail import GAIL
from mjengine.models.replay import ReplayBuffer
from mjengine.player import make_player, Player
from mjengine.strategy import RLAgentStrategy

TRAINABLES = ["GAIL", "PPO", "SAC", "DQN"]
AGENT_CLASS_MAP = {
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC
}


BEST_WINDOW = 100


class Trainer:
    def __init__(
            self,
            env: MahjongEnv,
            agent: Agent,
            model_dir: str,
            n_episode: int,
            n_checkpoint: int,
            save_checkpoint: bool,
            evaluate: bool = False,
            **kwargs) -> None:
        self.env = env
        self.agent = agent

        self.model_dir = model_dir

        self.n_episode = n_episode
        self.n_checkpoint = n_checkpoint
        self.save_checkpoint = save_checkpoint

        self.evaluate = evaluate
        self.eval_args = kwargs

        # reserved for off policy
        self.replay_buffer = None
        self.minimal_size = -1
        self.batch_size = -1

        self.return_list = []
        self.n_action_list = []

        self.episode_count = 0

        self._seed = None

        self.stopped = False

    def _set_stop_callback(self):
        signal.signal(signal.SIGINT, lambda sig, f: self.stop())

    def stop(self):
        self.stopped = True

    def _check_stop_flag(self):
        if self.stopped:
            print("Stop requested. Saving trainer status...")
            self._save_bp()
            sys.exit(0)

    def _train(self, step_method):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        if self.episode_count == 0:
            with open(os.path.join(self.model_dir, "train_output.csv"), "w") as f:
                f.write(",episode_return,n_action\n")
            self.return_list, self.n_action_list = [], []
        epi_per_cp = self.n_episode // self.n_checkpoint
        i0, j0 = self.episode_count // epi_per_cp, self.episode_count % epi_per_cp
        best_epi_return, epi_window = -1000, BEST_WINDOW
        for i in range(i0, self.n_checkpoint):
            if self.stopped:
                break
            with tqdm(total=epi_per_cp, desc=f'Iter. {i}') as pbar:
                if j0:
                    pbar.update(j0)
                for j in range(j0, epi_per_cp):
                    if self.stopped:
                        break

                    torch.cuda.empty_cache()

                    episode_return, episode_actions = step_method()
                    self.return_list.append(episode_return)
                    self.n_action_list.append(episode_actions)
                    with open(os.path.join(self.model_dir, "train_output.csv"), "a") as f:
                        f.write(",".join([
                            str(epi_per_cp * i + j + 1),
                            str(episode_return), str(episode_actions)]) + "\n")
                    if (j + 1) % 10 == 0:
                        pbar.set_postfix({
                            'epi': '%d' % (epi_per_cp * i + j + 1),
                            'ret': '%.3f' % np.mean(self.return_list[-10:]),
                            'a/e': '%.1f' % np.mean(self.n_action_list[-10:])
                        })
                        window_avg_ret = np.mean(self.return_list[-epi_window:])
                        if len(self.return_list) >= epi_window and window_avg_ret > best_epi_return:
                            self.agent.save(self.model_dir, best=True)
                            best_epi_return = window_avg_ret
                    pbar.update(1)
                    self.episode_count += 1
            j0 = 0

            # When stopped is set to True, evaluation will be skipped
            # while checkpoints will still be saved
            if self.evaluate and not self.stopped:
                eval_agent(i, self.agent, **self.eval_args)
            if i + 1 < self.n_checkpoint and self.save_checkpoint:
                self.agent.save(self.model_dir, i + 1)
        if self.stopped:
            print("Stop requested. Saving trainer status...")
            self._save_bp()
        else:
            self.agent.save(self.model_dir)
            if self.replay_buffer is not None:
                self.replay_buffer.save(self.model_dir, compression="bz2")
            if isinstance(self.agent, Deterministic):
                replay_buffer = self.replay_buffer
                replay_mtx = np.array([np.concatenate((s, [a])) for s, a, _, _, _ in replay_buffer], dtype=int)
                replay_mtx = csr_matrix(replay_mtx)
                save_npz(os.path.join(self.model_dir, "state_action_replay.npz"), replay_mtx)
                print(f"Saved {replay_mtx.shape[0]} state-action pairs")
        print(f'Training artifacts saved at "{os.path.abspath(self.model_dir)}"')

    def _train_off_policy_episode(self):
        episode_return, episode_actions = 0, 0
        state, info = self.env.reset()
        option = info["option"]
        done = False
        transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": []}
        acting_list = []
        while not done:
            action = self.agent.take_action(state, option)
            next_state, reward, done, _, info = self.env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            acting_list.append(info["acting_player"])
            # self.replay_buffer.add(state, action, reward, next_state, done)
            option = info["option"]
            state = info["next_player_state"]
            episode_return += reward
            if action < 75:  # count only non-pass actions
                episode_actions += 1
            if action == 74:
                winner, loser = acting_list[-1], -1
                for i in range(len(acting_list) - 1, -1, -1):
                    if transition_dict['actions'][i] < 34:
                        loser = acting_list[i]
                        break
                if loser == -1:
                    raise ValueError
                wc, lc = 0, 0
                for i in range(len(acting_list) - 1, -1, -1):
                    if acting_list[i] == winner:
                        transition_dict['rewards'][i] += 128 * (0.5 ** wc)
                        wc += 1
                    elif acting_list[i] == loser:
                        transition_dict['rewards'][i] -= 128 * (0.5 ** lc)
                        lc += 1
            if action == 68:
                winner = acting_list[-1]
                counts = [0, 0, 0, 0]
                for i in range(len(acting_list) - 1, -1, -1):
                    if acting_list[i] == winner:
                        transition_dict['rewards'][i] += 3 * 128 * (0.5 ** counts[winner])
                    else:
                        transition_dict['rewards'][i] -= 128 * (0.5 ** counts[acting_list[i]])
                    counts[acting_list[i]] += 1
            # if action == 74:  # revise the reward of chucking player
            #     for i in range(len(self.replay_buffer) - 1, -1, -1):
            #         if self.replay_buffer[i][1] < 34:
            #             s, a, r, ns, d = self.replay_buffer[i]
            #             self.replay_buffer[i] = (s, a, r - 1.0, ns, d)
            #             break
            # Train the Q network until buffer reached minimal size
            if len(self.replay_buffer) > self.minimal_size:
                b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.batch_size)
                batch_transitions = {
                    "states": b_s,
                    "actions": b_a,
                    "next_states": b_ns,
                    "rewards": b_r,
                    "dones": b_d
                }
                self.agent.update(**batch_transitions)
        self.replay_buffer.extend(**transition_dict)
        return episode_return, episode_actions

    def prepare_off_policy(self, buffer_cap, minimal_size, batch_size):
        self.replay_buffer = ReplayBuffer(capacity=buffer_cap)
        self.minimal_size = minimal_size
        self.batch_size = batch_size

    def train_off_policy(self):
        if self.replay_buffer is None or self.minimal_size < 0 or self.batch_size < 0:
            raise ValueError("Off-policy params not set yet. Use 'prepare_off_policy()' to set up")
        self._train(self._train_off_policy_episode)

    def _train_on_policy_episode(self):
        episode_return, episode_actions = 0, 0
        state, info = self.env.reset()
        option = info["option"]
        done = False
        transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": [], "options": []}
        acting_list = []
        while not done:
            action = self.agent.take_action(state, option)
            next_state, reward, done, _, info = self.env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            transition_dict['options'].append(option)
            acting_list.append(info["acting_player"])
            option = info["option"]
            state = info["next_player_state"]
            episode_return += reward
            if action < 75:  # count only non-pass actions
                episode_actions += 1
            if action == 74:
                winner, loser = acting_list[-1], -1
                for i in range(len(acting_list) - 1, -1, -1):
                    if transition_dict['actions'][i] < 34:
                        loser = acting_list[i]
                        break
                if loser == -1:
                    raise ValueError
                wc, lc = 0, 0
                for i in range(len(acting_list) - 1, -1, -1):
                    if acting_list[i] == winner:
                        transition_dict['rewards'][i] += 128 * (0.5 ** wc)
                        wc += 1
                    elif acting_list[i] == loser:
                        transition_dict['rewards'][i] -= 128 * (0.5 ** lc)
                        lc += 1
            if action == 68:
                winner = acting_list[-1]
                counts = [0, 0, 0, 0]
                for i in range(len(acting_list) - 1, -1, -1):
                    if acting_list[i] == winner:
                        transition_dict['rewards'][i] += 3 * 128 * (0.5 ** counts[winner])
                    else:
                        transition_dict['rewards'][i] -= 128 * (0.5 ** counts[acting_list[i]])
                    counts[acting_list[i]] += 1
            # if action == 74:  # revise the reward of chucking player
            #     for i in range(len(transition_dict['states']) - 1, -1, -1):
            #         if transition_dict['actions'][i] < 34:
            #             transition_dict['rewards'][i] -= -1.0
            #             break
        self.agent.update(**transition_dict)
        return episode_return, episode_actions

    def train_on_policy(self):
        self._train(self._train_on_policy_episode)

    def run(self):
        self._set_stop_callback()
        if self.agent.on_policy:
            self.train_on_policy()
        else:
            self.train_off_policy()

    def _save_bp(self):
        bp_dir = os.path.join(self.model_dir, ".bp")
        shutil.rmtree(bp_dir, ignore_errors=True)
        os.makedirs(bp_dir)
        self.env.save(bp_dir)
        self.env = None
        self.agent.save(bp_dir)
        self.agent = None
        if self.replay_buffer is not None:
            self.replay_buffer.save(bp_dir, compression="bz2")
            self.replay_buffer = None
        with open(os.path.join(bp_dir, "trainer.pkl"), "wb") as f:
            pickle.dump(self, f)
        rand_states = {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state()
        }
        with open(os.path.join(bp_dir, "rng_states.pkl"), "wb") as f:
            pickle.dump(rand_states, f)

    @staticmethod
    def from_settings(settings, out_dir=".", save_settings=True):
        env = MahjongEnv(**settings["env_params"])
        state_dim = env.encode_state().shape
        action_dim = env.action_space.shape[0]
        print(f"State dim: {state_dim}, Action dim: {action_dim}")

        if "seed" in settings:
            seed = settings["seed"]
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            env.seed(seed)

        if settings["agent"] in AGENT_CLASS_MAP:
            agt_cls = AGENT_CLASS_MAP[settings["agent"]]
            if settings.get("load_from"):
                agent = agt_cls.restore(settings["load_from"], train=True, **settings["agent_params"])
            else:
                agent = agt_cls(state_dim, action_dim=action_dim, **settings["agent_params"])
        else:
            agent = Deterministic(env.game, settings["agent"])

        timestamp = datetime.strftime(datetime.utcnow(), "%y%m%d%H%M%S")
        naming_list = []
        if settings["agent"] in TRAINABLES:
            naming_list += [
                settings["agent"],
                settings["agent_params"]["hidden_layer"],
                settings["agent_params"]["hidden_dim"]]
        else:
            naming_list.append(agent.strategy)
        if settings["agent"] == "DQN":
            algorithm = settings["agent_params"]["algorithm"]
            algorithm = "default" if algorithm == "DQN" else algorithm
            naming_list.append(algorithm)
        naming_list.append(timestamp)
        model_name = "_".join([str(n) for n in naming_list])
        model_dir = os.path.join(out_dir, model_name)
        # env.game.wall_file = os.path.join(model_dir, "game_walls.csv")
        trainer = Trainer(env, agent, model_dir, **settings["train_params"])
        if "seed" in settings:
            trainer._seed = settings["seed"]
        if not trainer.agent.on_policy:
            trainer.prepare_off_policy(**settings["off_policy_params"])
        if not os.path.isdir(trainer.model_dir):
            os.makedirs(trainer.model_dir)
        if save_settings:
            with open(os.path.join(model_dir, "training_settings.json"), "w") as f:
                json.dump(settings, f, indent=2)
        return trainer

    @staticmethod
    def restore(model_dir):
        bp_dir = os.path.join(model_dir, ".bp")
        if not os.path.isdir(bp_dir):
            raise FileNotFoundError('Break point directory ".bp" not found')

        with open(os.path.join(model_dir, "training_settings.json")) as f:
            settings = json.load(f)

        env = MahjongEnv.restore(bp_dir)
        if settings["agent"] not in AGENT_CLASS_MAP:
            raise ValueError(f'Trainer for agent type "{settings["agent"]}" cannot be restored')
        agent_cls = AGENT_CLASS_MAP[settings["agent"]]
        agent = agent_cls.restore(bp_dir, device=settings["agent_params"]["device"], train=True)
        with open(os.path.join(bp_dir, "trainer.pkl"), "rb") as f:
            trainer = pickle.load(f)
        trainer.env = env
        trainer.agent = agent
        replay_path = os.path.join(bp_dir, "replay_buffer.pkl.bz2")
        if os.path.exists(replay_path):
            replay_buffer = ReplayBuffer.restore(bp_dir)
            trainer.replay_buffer = replay_buffer
        trainer.stopped = False

        with open(os.path.join(bp_dir, "rng_states.pkl"), "rb") as f:
            rng_states = pickle.load(f)
        random.setstate(rng_states["random"])
        np.random.set_state(rng_states["numpy"])
        torch.set_rng_state(rng_states["torch"])
        torch.cuda.set_rng_state(rng_states["torch_cuda"])

        shutil.rmtree(bp_dir, ignore_errors=True)

        return trainer


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


# def train_on_policy(
#         env: Env,
#         agent: Agent,
#         model_dir: str,
#         n_episode: int,
#         n_checkpoint: int,
#         save_checkpoint: bool,
#         evaluate: bool, **kwargs) -> tuple[list, list]:
#     if n_checkpoint == 0:
#         n_checkpoint = 1
#     if not os.path.isdir(model_dir):
#         os.makedirs(model_dir)
#     return_list, n_action_list = [], []
#     for i in range(n_checkpoint):
#         with tqdm(total=int(n_episode / n_checkpoint), desc=f'Iter. {i}') as pbar:
#             for i_episode in range(int(n_episode / n_checkpoint)):
#                 episode_return, episode_actions = 0, 0
#                 state, info = env.reset()
#                 option = info["option"]
#                 done = False
#                 transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": []}
#                 while not done:
#                     action = agent.take_action(state, option)
#                     next_state, reward, done, _, info = env.step(action)
#                     transition_dict['states'].append(state)
#                     transition_dict['actions'].append(action)
#                     transition_dict['next_states'].append(next_state)
#                     transition_dict['rewards'].append(reward)
#                     transition_dict['dones'].append(done)
#                     option = info["option"]
#                     state = info["next_player_state"]
#                     episode_return += reward
#                     episode_actions += 1
#                 return_list.append(episode_return)
#                 n_action_list.append(episode_actions)
#                 agent.update(**transition_dict)
#                 with open(os.path.join(model_dir, "train_output.csv"), "a") as f:
#                     f.write(",".join([
#                         str(n_episode / 10 * i + i_episode + 1),
#                         str(episode_return), str(episode_actions)]) + "\n")
#                 if (i_episode + 1) % 10 == 0:
#                     pbar.set_postfix({
#                         'epi': '%d' % (n_episode / 10 * i + i_episode + 1),
#                         'ret': '%.3f' % np.mean(return_list[-10:]),
#                         'r/a': '%.3f' % (np.sum(return_list[-10:]) / np.sum(n_action_list[-10:])),
#                         'a/e': '%.1f' % np.mean(n_action_list[-10:])
#                     })
#                 pbar.update(1)
#         if evaluate:
#             eval_agent(i, agent, **kwargs)
#         if i + 1 < n_checkpoint and save_checkpoint:
#             agent.save(model_dir, i + 1)
#     agent.save(model_dir)
#     return return_list, n_action_list
#
#
# def train_off_policy(
#         env: Env,
#         agent: Agent,
#         replay_buffer: ReplayBuffer,
#         model_dir: str,
#         n_episode: int,
#         n_checkpoint: int,
#         save_checkpoint: bool,
#         minimal_size: int,
#         batch_size: int,
#         evaluate: bool = False, **kwargs) -> tuple[list, list]:
#     if not os.path.isdir(model_dir):
#         os.makedirs(model_dir)
#     with open(os.path.join(model_dir, "train_output.csv"), "w") as f:
#         f.write(",episode_return,n_action\n")
#     return_list, n_action_list = [], []
#     for i in range(n_checkpoint):
#         with tqdm(total=int(n_episode / n_checkpoint), desc=f'Iter. {i}') as pbar:
#             for i_episode in range(int(n_episode / n_checkpoint)):
#                 episode_return, episode_actions = 0, 0
#                 state, info = env.reset()
#                 option = info["option"]
#                 done = False
#                 while not done:
#                     try:
#                         action = agent.take_action(state, option)
#                     except Exception as e:
#                         print(f"[Error] Occurred at action {episode_actions}, episode {i_episode}")
#                         raise e
#                     # acting_player = env.game.acting_player
#                     next_state, reward, done, _, info = env.step(action)
#                     # print(f"acting for player {acting_player}, action {action}, reward {reward}")
#                     replay_buffer.add(state, action, reward, next_state, done)
#                     option = info["option"]
#                     state = info["next_player_state"]
#                     episode_return += reward
#                     episode_actions += 1
#                     # Train the Q network until buffer reached minimal size
#                     if len(replay_buffer) > minimal_size:
#                         b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
#                         transition_dict = {
#                             "states": b_s,
#                             "actions": b_a,
#                             "next_states": b_ns,
#                             "rewards": b_r,
#                             "dones": b_d
#                         }
#                         agent.update(**transition_dict)
#                 return_list.append(episode_return)
#                 n_action_list.append(episode_actions)
#                 with open(os.path.join(model_dir, "train_output.csv"), "a") as f:
#                     f.write(",".join([
#                         str(n_episode / 10 * i + i_episode + 1),
#                         str(episode_return), str(episode_actions)]) + "\n")
#                 if (i_episode + 1) % 10 == 0:
#                     pbar.set_postfix({
#                         'epi': '%d' % (n_episode / 10 * i + i_episode + 1),
#                         'ret': '%.3f' % np.mean(return_list[-10:]),
#                         'r/a': '%.3f' % (np.sum(return_list[-10:]) / np.sum(n_action_list[-10:])),
#                         'a/e': '%.1f' % np.mean(n_action_list[-10:])
#                     })
#                 pbar.update(1)
#         if evaluate:
#             eval_agent(i, agent, **kwargs)
#         if i + 1 < n_checkpoint and save_checkpoint:
#             agent.save(model_dir, i + 1)
#     agent.save(model_dir)
#     replay_buffer.save(model_dir, compression="bz2")
#     return return_list, n_action_list


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
