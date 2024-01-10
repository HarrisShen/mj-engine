import numpy as np
import torch
from torch.nn import Module, Linear
import torch.nn.functional as F
from torch.optim import Adam


class QNet(Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.input = Linear(state_dim, hidden_dim)
        self.output = Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.output(x)
        return x


class DQN:
    def __init__(
            self,
            state_dim, hidden_dim, action_dim,
            lr, gamma, epsilon,
            target_update, device):
        self.action_dim = action_dim
        self.q_net = QNet(state_dim, hidden_dim, action_dim)
        self.q_net.to(device)
        self.target_q_net = QNet(state_dim, hidden_dim, action_dim)
        self.target_q_net.to(device)
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state: np.ndarray, option: np.ndarray):
        if np.random.random() < self.epsilon:
            action = int(np.random.choice(np.arange(0, 76, dtype=int)[option], size=1))
        else:
            state = torch.from_numpy(state.astype(np.float32)).to(self.device)
            old_index = np.arange(0, 76, dtype=int)[option]
            max_index = self.q_net(state)[option].argmax().item()
            action = old_index[max_index].item()
        return action

    def update(self, **kwargs):
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
