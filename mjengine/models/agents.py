import json
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Linear
from torch.optim import Adam, lr_scheduler

from mjengine.constants import PlayerAction, GameStatus

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


class VANet(Module):
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
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(-1, keepdim=True)
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


class PolicyNet(torch.nn.Module):
    """
    Source: https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95/
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class ValueNet(torch.nn.Module):
    """
    Source: https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95/
    """
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantages = np.zeros(td_delta.shape, dtype=np.float32)
    advantage = 0.0
    for i in range(len(td_delta) - 1, -1, -1):
        advantage = gamma * lmbda * advantage + td_delta[i]
        advantages[i] = advantage
    return torch.tensor(advantages, dtype=torch.float)


class PPO(Agent):
    """
    Source: https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95/
    """
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, train=True):
        super().__init__(train)

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

        self.epochs = epochs  # number of repetitions

        def adjust_lr(epoch: int) -> float:
            if epoch > 60000:
                return 0.001
            if epoch > 25000:
                return 0.003
            if epoch > 10000:
                return 0.01
            if epoch > 4000:
                return 0.03
            if epoch > 2000:
                return 0.1
            if epoch > 1000:
                return 0.3
            return 1.0

        self.actor_scheduler = lr_scheduler.LambdaLR(self.actor_optimizer, adjust_lr)
        self.critic_scheduler = lr_scheduler.LambdaLR(self.critic_optimizer, adjust_lr)

        self.gamma = gamma
        self.lmbda = lmbda

        self.eps = eps  # PPO clip range
        self.device = device

    def take_action(self, state, option):
        state = torch.from_numpy(state.astype(np.float32)).to(self.device)
        # zero out illegal actions, then normalize probs to sum 1
        option = torch.from_numpy(option.astype(np.float32)).to(self.device)
        probs = self.actor(state) * option
        prob_sum = probs.sum(dim=-1)
        if prob_sum == 0:
            probs = option / option.sum(dim=-1)
        else:
            probs = probs / prob_sum
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, **kwargs):
        assert self.train
        states = torch.tensor(np.array(kwargs["states"]),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(kwargs['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(kwargs['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(kwargs['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(kwargs['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # PPO clip
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # loss function
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            self.actor_scheduler.step()
            self.critic_scheduler.step()

    def save(self, model_dir: str = ".", model_name: str | None = None) -> str:
        if model_name is None:
            timestamp = datetime.strftime(datetime.utcnow(), "%y%m%d%H%M%S")
            model_name = f"PPO_{self.hidden_dim}_{timestamp}"
        model_dir = os.path.join(model_dir, model_name)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        torch.save(self.actor.state_dict(), os.path.join(model_dir, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(model_dir, "critic.pt"))
        with open(os.path.join(model_dir, "model_settings.json"), "w") as outf:
            json.dump({
                "state_dim": self.state_dim,
                "hidden_dim": self.hidden_dim,
                "action_dim": self.action_dim,
                "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr,
                "lmbda": self.lmbda,
                "epochs": self.epochs,
                "gamma": self.gamma,
                "eps": self.eps
            }, outf, indent=2)
        return model_dir

    @staticmethod
    def restore(model_dir: str, device: torch.device, train: bool = False):
        with open(os.path.join(model_dir, "model_settings.json"), "r") as f:
            kwargs = json.load(f)
        kwargs["device"] = device
        obj = PPO(**kwargs)
        state_dict = torch.load(os.path.join(model_dir, "actor.pt"))
        obj.actor.load_state_dict(state_dict)
        state_dict = torch.load(os.path.join(model_dir, "critic.pt"))
        obj.critic.load_state_dict(state_dict)
        return obj
