import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from gymnasium import Env
from scipy.sparse import load_npz
from tqdm import tqdm

from mjengine.game import Game
from mjengine.models.agent import PPO, DQN, Deterministic
from mjengine.models.agent.agent import Agent
from mjengine.models.agent.sac import SAC
from mjengine.models.env import MahjongEnv
from mjengine.models.gail import GAIL
from mjengine.models.utils import ReplayBuffer
from mjengine.player import make_player, Player
from mjengine.strategy import RLAgentStrategy

TRAINABLES = ["GAIL", "PPO", "SAC", "DQN"]


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

    # def seed(self, sd):
    #     self._seed = sd
    #     self.env.seed(self._seed)
    #     if self.replay_buffer is not None:
    #         self.replay_buffer.seed(self._seed)

    def _train(self, step_method):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        with open(os.path.join(self.model_dir, "train_output.csv"), "w") as f:
            f.write(",episode_return,n_action\n")
        self.return_list, self.n_action_list = [], []
        for i in range(self.n_checkpoint):
            with tqdm(total=int(self.n_episode / self.n_checkpoint), desc=f'Iter. {i}') as pbar:
                for i_episode in range(int(self.n_episode / self.n_checkpoint)):
                    episode_return, episode_actions = step_method()
                    self.return_list.append(episode_return)
                    self.n_action_list.append(episode_actions)
                    with open(os.path.join(self.model_dir, "train_output.csv"), "a") as f:
                        f.write(",".join([
                            str(self.n_episode / 10 * i + i_episode + 1),
                            str(episode_return), str(episode_actions)]) + "\n")
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'epi': '%d' % (self.n_episode / 10 * i + i_episode + 1),
                            'ret': '%.3f' % np.mean(self.return_list[-10:]),
                            'r/a': '%.3f' % (np.sum(self.return_list[-10:]) / np.sum(self.n_action_list[-10:])),
                            'a/e': '%.1f' % np.mean(self.n_action_list[-10:])
                        })
                    pbar.update(1)
                    self.episode_count += 1
            if self.evaluate:
                eval_agent(i, self.agent, **self.eval_args)
            if i + 1 < self.n_checkpoint and self.save_checkpoint:
                self.agent.save(self.model_dir, i + 1)
        self.agent.save(self.model_dir)
        if self.replay_buffer is not None:
            self.replay_buffer.save(self.model_dir, compression="bz2")
        print(f'Training artifacts saved at "{os.path.abspath(self.model_dir)}"')

    def _train_off_policy_episode(self):
        episode_return, episode_actions = 0, 0
        state, info = self.env.reset()
        option = info["option"]
        done = False
        while not done:
            action = self.agent.take_action(state, option)
            next_state, reward, done, _, info = self.env.step(action)
            self.replay_buffer.add(state, action, reward, next_state, done)
            option = info["option"]
            state = info["next_player_state"]
            episode_return += reward
            episode_actions += 1
            # Train the Q network until buffer reached minimal size
            if len(self.replay_buffer) > self.minimal_size:
                b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.batch_size)
                transition_dict = {
                    "states": b_s,
                    "actions": b_a,
                    "next_states": b_ns,
                    "rewards": b_r,
                    "dones": b_d
                }
                self.agent.update(**transition_dict)
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
        transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": []}
        while not done:
            action = self.agent.take_action(state, option)
            next_state, reward, done, _, info = self.env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            option = info["option"]
            state = info["next_player_state"]
            episode_return += reward
            episode_actions += 1
        self.agent.update(**transition_dict)
        return episode_return, episode_actions

    def train_on_policy(self):
        self._train(self._train_on_policy_episode)

    def run(self):
        if self.agent.on_policy:
            self.train_on_policy()
        else:
            self.train_off_policy()

    @staticmethod
    def from_settings(settings, out_dir=".", save_settings=True):
        env = MahjongEnv()
        state_dim = len(env.game.to_numpy())
        action_dim = env.action_space.shape[0]
        print(f"State dim: {state_dim}, Action dim: {action_dim}")

        if "seed" in settings:
            seed = settings["seed"]
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            env.seed(seed)

        if settings["agent"] == "DQN":
            agent = DQN(state_dim=state_dim, action_dim=action_dim, **settings["agent_params"])
        elif settings["agent"] == "PPO":
            agent = PPO(state_dim=state_dim, action_dim=action_dim, **settings["agent_params"])
        elif settings["agent"] == "SAC":
            agent = SAC(state_dim=state_dim, action_dim=action_dim, **settings["agent_params"])
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
        trainer = Trainer(env, agent, model_dir, **settings["train_params"])
        if "seed" in settings:
            trainer._seed = settings["seed"]
        if not trainer.agent.on_policy:
            trainer.prepare_off_policy(**settings["off_policy_params"])
        if not os.path.isdir(trainer.model_dir):
            os.makedirs(trainer.model_dir)
        if save_settings:
            with open(os.path.join(out_dir, "training_settings.json"), "w") as f:
                json.dump(settings, f, indent=2)
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
