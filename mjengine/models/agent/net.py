import math

import torch
from torch.nn import ModuleList, Linear, functional as F, Module, Conv1d, MaxPool1d, Flatten, Sequential, ReLU, Softmax, \
    BatchNorm1d


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
    def __init__(self, state_dim, hidden_dim, action_dim, conv_layer=1, hidden_layer=1):
        super(PolicyNet, self).__init__()
        self.conv_layer = conv_layer
        self.hidden_layer = hidden_layer
        conv_channels = math.ceil(hidden_dim / 34)
        conv_layers = [
            Conv1d(state_dim[0], conv_channels, kernel_size=3, padding=1),
            BatchNorm1d(conv_channels, eps=1e-5, momentum=0.001),
            ReLU()
        ]
        for _ in range(conv_layer - 1):
            conv_layers += [
                Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
                BatchNorm1d(conv_channels, eps=1e-5, momentum=0.001),
                ReLU()
            ]
        fc_layers = [Linear(conv_channels * 34, hidden_dim)]
        for _ in range(hidden_layer - 1):
            fc_layers += [ReLU(), Linear(hidden_dim, hidden_dim)]
        fc_layers += [ReLU(), Linear(hidden_dim, action_dim)]
        self.net = Sequential(*conv_layers, Flatten(), *fc_layers, Softmax(dim=-1))

    def forward(self, x):
        return self.net(x)


class ValueNet(torch.nn.Module):
    """
    Source: https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95/
    """
    def __init__(self, state_dim, hidden_dim, conv_layer=1, hidden_layer=1):
        super(ValueNet, self).__init__()
        self.conv_layer = conv_layer
        self.hidden_layer = hidden_layer
        conv_channels = math.ceil(hidden_dim / 34)
        conv_layers = [
            Conv1d(state_dim[0], conv_channels, kernel_size=3, padding=1),
            BatchNorm1d(conv_channels, eps=1e-5, momentum=0.001),
            ReLU()
        ]
        for _ in range(conv_layer - 1):
            conv_layers += [
                Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
                BatchNorm1d(conv_channels, eps=1e-5, momentum=0.001),
                ReLU()
            ]
        fc_layers = [Linear(conv_channels * 34, hidden_dim)]
        for _ in range(hidden_layer - 1):
            fc_layers += [ReLU(), Linear(hidden_dim, hidden_dim)]
        fc_layers += [ReLU(), Linear(hidden_dim, 1)]
        self.net = Sequential(*conv_layers, Flatten(), *fc_layers)

    def forward(self, x):
        return self.net(x)
