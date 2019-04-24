#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : models.py
# @Author: zixiao
# @Date  : 2019-04-24
# @Desc  :
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, in_channels, num_action):
        super(Actor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc4 = nn.Linear(in_features=2 * 2 * 64, out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=num_action)

    def forward(self, x):  # 4 84 84
        x = self.conv1(x)  # 32 10,10
        x = self.conv2(x)  # 64 2 2
        # x = self.conv3(x)  # 128 1 1
        x = self.fc4(x.view(x.size(0), -1))
        x = self.fc5(x)
        x = F.softmax(x, dim=1)
        return x


class Critic(nn.Module):
    def __init__(self, in_channels):
        super(Critic, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc4 = nn.Linear(in_features=2 * 2 * 64, out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):  # 4 84 84
        x = self.conv1(x)  # 32 10,10
        x = self.conv2(x)  # 64 2 2
        # x = self.conv3(x)  # 128 1 1
        x = self.fc4(x.view(x.size(0), -1))
        x = self.fc5(x)
        return x
