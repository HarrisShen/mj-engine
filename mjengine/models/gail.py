"""
Implementation of generative adversarial imitation learning，GAIL
Source: https://hrl.boyuai.com/chapter/3/%E6%A8%A1%E4%BB%BF%E5%AD%A6%E4%B9%A0/
"""
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

from mjengine.models.agent.ppo import adjust_lr


class Discriminator(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))


class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr, device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.discriminator = Discriminator(state_dim, hidden_dim, action_dim).to(self.device)
        self.discriminator_lr = lr
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr)
        self.disc_scheduler = lr_scheduler.LambdaLR(self.discriminator_optimizer, adjust_lr)
        self.agent = agent

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(self.device)
        expert_actions = torch.tensor(expert_a).to(self.device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(self.device)
        agent_actions = torch.tensor(agent_a).to(self.device)
        expert_actions = F.one_hot(expert_actions.long(), num_classes=self.action_dim).float()
        agent_actions = F.one_hot(agent_actions.long(), num_classes=self.action_dim).float()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        discriminator_loss = torch.nn.BCELoss()(
            agent_prob, torch.ones_like(agent_prob)) + torch.nn.BCELoss()(
                expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        self.disc_scheduler.step()

        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': dones
        }
        self.agent.update(**transition_dict)
