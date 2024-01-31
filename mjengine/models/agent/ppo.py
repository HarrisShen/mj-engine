import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler

from mjengine.models.agent import Agent


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


def adjust_lr(epoch: int) -> float:
    if epoch > 6000:
        return 0.001
    if epoch > 2500:
        return 0.003
    if epoch > 1000:
        return 0.01
    if epoch > 400:
        return 0.03
    if epoch > 200:
        return 0.1
    if epoch > 100:
        return 0.3
    return 1.0


class PPO(Agent):
    """
    Source: https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95/
    """
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lr_schedule,
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

        self.lr_schedule = lr_schedule
        self.actor_scheduler = lr_scheduler.LambdaLR(self.actor_optimizer, adjust_lr)
        self.critic_scheduler = lr_scheduler.LambdaLR(self.critic_optimizer, adjust_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO clip range

        self.device = device

        self.n_episode = 0

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
        actions = torch.tensor(kwargs['actions']).view(-1, 1).to(self.device)
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
        if self.lr_schedule:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
        self.n_episode += 1

    def save(self, model_dir, checkpoint=None) -> str:
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        settings_file = os.path.join(model_dir, "model_settings.json")
        if not os.path.exists(settings_file):
            with open(settings_file, "w") as outf:
                json.dump({
                    "state_dim": self.state_dim,
                    "hidden_dim": self.hidden_dim,
                    "action_dim": self.action_dim,
                    "actor_lr": self.actor_lr,
                    "critic_lr": self.critic_lr,
                    "lmbda": self.lmbda,
                    "gamma": self.gamma,
                    "eps": self.eps,
                    "epochs": self.epochs
                }, outf, indent=2)
        model_state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict(),
            "critic_scheduler": self.critic_scheduler.state_dict(),
            "n_episode": self.n_episode
        }
        filename = "model_state.pt" if checkpoint is None else f"model_state_cp_{checkpoint}.pt"
        torch.save(model_state, os.path.join(model_dir, filename))
        return model_dir

    @staticmethod
    def restore(model_dir: str, device: torch.device, train: bool = False, checkpoint: int | None = None):
        with open(os.path.join(model_dir, "model_settings.json"), "r") as f:
            kwargs = json.load(f)
        obj = PPO(train=train, device=device, **kwargs)
        state_file = "model_state.pt" if checkpoint is None else f"model_state_cp_{checkpoint}.pt"
        model_state = torch.load(os.path.join(model_dir, state_file))
        for k, v in model_state.items():
            if isinstance(v, dict):  # restore from state dict
                getattr(obj, k).load_state_dict(model_state[k])
            else:
                setattr(obj, k, model_state[k])
        return obj
