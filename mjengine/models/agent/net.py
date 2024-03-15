import torch
from torch.nn import ModuleList, Linear, functional as F, Module, Conv1d, MaxPool1d


class QNet(Module):
    """
    Source: https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95
    """
    def __init__(self, state_dim, hidden_dim, action_dim, hidden_layer=1):
        super(QNet, self).__init__()
        self.hidden_layer = hidden_layer
        # self.conv = Conv1d(state_dim, hidden_dim, kernel_size=3, padding=1)
        # self.pool = MaxPool1d(kernel_size=3)
        # self.layers = ModuleList([Linear(hidden_dim, hidden_dim)])
        self.layers = ModuleList([Linear(state_dim, hidden_dim)])
        self.layers.extend([Linear(hidden_dim, hidden_dim) for _ in range(self.hidden_layer - 1)])
        self.layers.append(Linear(hidden_dim, action_dim))

    def forward(self, x):
        # x = self.conv(x.view(392, -1))
        # x = F.relu(x)
        # x = self.pool(x)
        # x = x.view(-1, 256)
        # if x.shape[0] == 1:
        #     x = x.view(256)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)


class VANet(Module):
    """
    Source: https://hrl.boyuai.com/chapter/2/dqn%E6%94%B9%E8%BF%9B%E7%AE%97%E6%B3%95/
    """
    def __init__(self, state_dim, hidden_dim, action_dim, hidden_layer=1):
        super(VANet, self).__init__()
        self.hidden_layer = hidden_layer
        self.layers_a = ModuleList([Linear(state_dim, hidden_dim)])
        self.layers_a.extend([Linear(hidden_dim, hidden_dim) for _ in range(self.hidden_layer - 1)])
        self.layers_a.append(Linear(hidden_dim, action_dim))
        self.layers_v = ModuleList([Linear(state_dim, hidden_dim)])
        self.layers_v.extend([Linear(hidden_dim, hidden_dim) for _ in range(self.hidden_layer - 1)])
        self.layers_v.append(Linear(hidden_dim, 1))

    def forward(self, x):
        A = x
        for i in range(len(self.layers_a) - 1):
            A = F.relu(self.layers_a[i](A))
        A = self.layers_a[-1](A)
        V = x
        for i in range(len(self.layers_v) - 1):
            V = F.relu(self.layers_v[i](V))
        V = self.layers_v[-1](V)
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
