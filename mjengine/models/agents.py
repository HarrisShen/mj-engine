import json
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear
from torch.optim import Adam

from mjengine.constants import PlayerAction, GameStatus
from mjengine.models.utils import find_last_discard, game_numpy_to_dict

DQN_ALGORITHMS = ["DQN", "DoubleDQN", "DuelingDQN"]


class QNet(Module):
    """
    Source: https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.input = Linear(state_dim, hidden_dim)
        self.output = Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.output(x)
        return x


class VANet(torch.nn.Module):
    """
    Source: https://hrl.boyuai.com/chapter/2/dqn%E6%94%B9%E8%BF%9B%E7%AE%97%E6%B3%95/
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VANet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        A_mean = A.mean(-1)
        if A.dim() == 2:
            A_mean = A_mean.view(-1, 1)
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A_mean
        return Q


class Agent(ABC):
    def __init__(self, train: bool = True):
        self.train = train

    @abstractmethod
    def take_action(self, state: np.ndarray, option: np.ndarray) -> int:
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        pass

    @abstractmethod
    def save(self, model_dir: str = ".", model_name: str | None = None) -> str:
        pass

    @staticmethod
    @abstractmethod
    def restore(model_dir: str, device: torch.device, train: bool = False):
        pass


class Deterministic(Agent):
    def __init__(self, game, strategy):
        super().__init__(False)

        from mjengine.player import make_player

        self.make_player = make_player
        self.game = game

        self.strategy = strategy
        self.game.players = [make_player(strategy) for _ in range(4)]
        # from mjengine.strategy import Strategy, RandomStrategy, AnalyzerStrategy
        #
        # if isinstance(strategy, str):
        #     if strategy == "random":
        #         strategy = RandomStrategy()
        #     elif strategy == "analyzer":
        #         strategy = AnalyzerStrategy()
        #     else:
        #         strategy = AnalyzerStrategy(strategy)
        # self.strategy: Strategy = strategy

    def take_action(self, state: np.ndarray, option: np.ndarray) -> int:
        game = self.game
        last_discard = None
        if game.status == GameStatus.CHECK:
            last_discard = game.players[game.current_player].discards[-1]
        action, tile = game.players[game.acting_player].decide(
            game.option,
            last_discard,
            game.to_dict(game.acting_player))
        if action is None:
            return tile
        if action == PlayerAction.PASS:
            return 75
        if action == PlayerAction.WIN:
            return 68 if tile is None else 74
        if action == PlayerAction.KONG:
            return 34 + tile if game.option.concealed_kong else 73
        if action == PlayerAction.PONG:
            return 72
        # CHOW
        return 68 + int(action)
        # hand = state[1: 35].tolist()
        # if not option[-1]:
        #     return self.strategy.discard(hand, game_numpy_to_dict(state))[1]
        # if option[68]:  # win by self
        #     action, _ = self.strategy.win(hand, game_numpy_to_dict(state))
        #     if action == PlayerAction.WIN:
        #         return 68
        # if any(option[34: 68]):  # concealed kong
        #     tiles = [i for i, v in enumerate(option[34: 68]) if v]
        #     action, tile = self.strategy.kong(hand, game_numpy_to_dict(state), tiles)
        #     if action == PlayerAction.KONG:
        #         return 34 + tile
        # if option[74]:  # win by chuck
        #     action, _ = self.strategy.win(hand, game_numpy_to_dict(state), find_last_discard(state))
        #     if action == PlayerAction.WIN:
        #         return 74
        # if option[73]:  # kong
        #     action, _ = self.strategy.kong(hand, game_numpy_to_dict(state), [find_last_discard(state)])
        #     if action == PlayerAction.KONG:
        #         return 73
        # if option[72]:  # pong
        #     action, _ = self.strategy.pong(hand, game_numpy_to_dict(state), find_last_discard(state))
        #     if action == PlayerAction.PONG:
        #         return 72
        # if any(option[69: 72]):  # chow
        #     action, _ = self.strategy.chow(
        #         hand, game_numpy_to_dict(state),
        #         find_last_discard(state),
        #         [True] + option[69: 72].tolist())
        #     if action != PlayerAction.PASS:
        #         return 68 + action
        # return 75

    def update(self, **kwargs) -> None:
        pass

    def save(self, model_dir: str = ".", model_name: str | None = None) -> str:
        if model_name is None:
            timestamp = datetime.strftime(datetime.utcnow(), "%y%m%d%H%M%S")
            model_name = f"{self.strategy}_{timestamp}"
        model_dir = os.path.join(model_dir, model_name)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        return model_dir

    @staticmethod
    def restore(model_dir: str, device: torch.device, train=False):
        raise RuntimeError("Deterministic agent cannot be restored from files")


class DQN(Agent):
    def __init__(
            self,
            state_dim, hidden_dim, action_dim,
            lr, gamma, epsilon, target_update,
            device, algorithm="DQN", train=True):
        super().__init__(train)

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.algorithm = "DQN" if algorithm == "default" else algorithm
        if self.algorithm not in DQN_ALGORITHMS:
            raise ValueError("Invalid algorithm type for Deep Q Network model")

        if self.algorithm == "DuelingDQN":
            self.q_net = VANet(state_dim, hidden_dim, action_dim)
            self.q_net.to(device)
            self.target_q_net = VANet(state_dim, hidden_dim, action_dim)
            self.target_q_net.to(device)
        else:
            self.q_net = QNet(state_dim, hidden_dim, action_dim)
            self.q_net.to(device)
            self.target_q_net = QNet(state_dim, hidden_dim, action_dim)
            self.target_q_net.to(device)

        self.lr = lr
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state: np.ndarray, option: np.ndarray) -> int:
        if self.train and np.random.random() < self.epsilon:
            action = int(np.random.choice(np.arange(0, 76, dtype=int)[option], size=1))
        else:
            state = torch.from_numpy(state.astype(np.float32)).to(self.device)
            old_index = np.arange(0, 76, dtype=int)[option]
            max_index = self.q_net(state)[option].argmax().item()
            action = old_index[max_index].item()
        return action

    def update(self, **kwargs) -> None:
        assert self.train
        states = torch.tensor(kwargs["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(kwargs["actions"]).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            kwargs["rewards"],
            dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(
            kwargs["next_states"],
            dtype=torch.float).to(self.device)
        dones = torch.tensor(
            kwargs["dones"],
            dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        # Maximum Q for next states
        if self.algorithm == "DoubleDQN":
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # Update target network
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1

    def save(self, model_dir: str = ".", model_name: str | None = None) -> str:
        if model_name is None:
            timestamp = datetime.strftime(datetime.utcnow(), "%y%m%d%H%M%S")
            algo = "default" if self.algorithm == "DQN" else self.algorithm
            model_name = f"DQN_{self.hidden_dim}_{algo}_{timestamp}"
        model_dir = os.path.join(model_dir, model_name)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        torch.save(self.q_net.state_dict(), os.path.join(model_dir, "q_net.pt"))
        torch.save(self.target_q_net.state_dict(), os.path.join(model_dir, "target_q_net.pt"))
        with open(os.path.join(model_dir, "model_settings.json"), "w") as outf:
            json.dump({
                "state_dim": self.state_dim,
                "hidden_dim": self.hidden_dim,
                "action_dim": self.action_dim,
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "target_update": self.target_update,
                "algorithm": self.algorithm
            }, outf, indent=2)
        with open(os.path.join(model_dir, "model_state.pkl"), "wb") as outf:
            pickle.dump({"count": self.count}, outf)
        return model_dir

    @staticmethod
    def restore(model_dir, device, train: bool = False):
        with open(os.path.join(model_dir, "model_settings.json"), "r") as f:
            kwargs = json.load(f)
        kwargs["device"] = device
        obj = DQN(**kwargs)
        with open(os.path.join(model_dir, "model_state.pkl"), "rb") as f:
            attributes = pickle.load(f)
        obj.__dict__.update(attributes)
        state_dict = torch.load(os.path.join(model_dir, "q_net.pt"))
        obj.q_net.load_state_dict(state_dict)
        state_dict = torch.load(os.path.join(model_dir, "target_q_net.pt"))
        obj.target_q_net.load_state_dict(state_dict)
        return obj
