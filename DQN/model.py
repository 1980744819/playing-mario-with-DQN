#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: 1980744819@qq.com
# @Date  : 2019-03-20
# @Desc  :
from torch import nn
import torch.nn.functional as F


class Deep_Q_net(nn.Module):
    def __init__(self, in_channels=4, num_actions=14):
        super(Deep_Q_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)  # 32,55,59
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)  # 64,26,28
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)  # 64,24,26
        self.fc4 = nn.Linear(in_features=24 * 26 * 64, out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class CNN(nn.Module):
    def __init__(self, in_channels=12, num_action=14):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # (4,220,240)
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=8,
                stride=4,
            ),  # 32,55,59
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )  # 16,27,29
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

        )  # 32 6 6
        self.fc4 = nn.Linear(in_features=6 * 6 * 32, out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=num_action)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc4(x.view(x.size(0), -1))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc5(x)
        return x
