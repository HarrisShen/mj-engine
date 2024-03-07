"""
Source: https://hrl.boyuai.com/chapter/2/sac%E7%AE%97%E6%B3%95/
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from mjengine.models.agent import Agent
from mjengine.models.agent.net import PolicyNet, QNet


class SAC(Agent):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device, train=True):
        super().__init__(on_policy=False, device=device, train=train)

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QNet(state_dim, hidden_dim, action_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_lr, self.critic_lr = actor_lr, critic_lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.alpha_lr = alpha_lr
        # use log alpha could make training stable
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau

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

    # Get target Q value by calculating expectation based on probabilities from policy net
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=-1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=-1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        # Update two Q networks
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update policy network
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=-1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Alpha value
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        self.step()

    def save(self, model_dir: str, checkpoint: int | None = None) -> str:
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
                    "alpha_lr": self.alpha_lr,
                    "target_entropy": self.target_entropy,
                    "gamma": self.gamma,
                    "tau": self.tau
                }, outf, indent=2)
        model_state = {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            "n_episodes": self.n_episode
        }
        filename = "model_state.pt" if checkpoint is None else f"model_state_cp_{checkpoint}.pt"
        torch.save(model_state, os.path.join(model_dir, filename))
        return model_dir

    @staticmethod
    def restore(model_dir: str, device: torch.device, train: bool = False):
        pass
