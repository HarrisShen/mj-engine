import torch
from torch.nn import ModuleList, Linear, functional as F, Module


class QNet(Module):
    """
    Source: https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95
    """
    def __init__(self, state_dim, hidden_dim, action_dim, hidden_layer=1):
        super(QNet, self).__init__()
        self.hidden_layer = hidden_layer
        self.layers = ModuleList([Linear(state_dim, hidden_dim)])
        self.layers.extend([Linear(hidden_dim, hidden_dim) for _ in range(self.hidden_layer - 1)])
        self.layers.append(Linear(hidden_dim, action_dim))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)


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


class PolicyNet(torch.nn.Module):
    """
    Source: https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95/
    """
    def __init__(self, state_dim, hidden_dim, action_dim, hidden_layer=1):
        super(PolicyNet, self).__init__()
        self.hidden_layer = hidden_layer
        self.layers = ModuleList([Linear(state_dim, hidden_dim)])
        self.layers.extend([Linear(hidden_dim, hidden_dim) for _ in range(self.hidden_layer - 1)])
        self.layers.append(Linear(hidden_dim, action_dim))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return F.softmax(self.layers[-1](x), dim=-1)


class ValueNet(torch.nn.Module):
    """
    Source: https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95/
    """
    def __init__(self, state_dim, hidden_dim, hidden_layer=1):
        super(ValueNet, self).__init__()
        self.hidden_layer = hidden_layer
        self.layers = ModuleList([Linear(state_dim, hidden_dim)])
        self.layers.extend([Linear(hidden_dim, hidden_dim) for _ in range(self.hidden_layer - 1)])
        self.layers.append(Linear(hidden_dim, 1))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)
