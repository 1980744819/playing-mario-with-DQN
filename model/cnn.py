#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : cnn.py
# @Author: zixiao@ainirobot.com
# @Date  : 2019-03-21
# @Desc  :
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels=4, num_actions=14):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class NormalCNN(nn.Module):
    def __init__(self):
        super(NormalCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=4
            ),
        )

    def forward(self, x):
        return
