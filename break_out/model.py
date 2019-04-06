#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: zixiao
# @Date  : 2019-04-07
# @Desc  :
from torch import nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):
    def __init__(self, in_channels, num_action):
        super(CNN, self).__init__()
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
        self.fc4 = nn.Linear(in_features=2 * 1 * 128, out_features=256)
        self.fc5 = nn.Linear(in_features=256, out_features=num_action)

    def forward(self, x):  # 4 210 160
        x = self.conv1(x)  # 32 25,29
        x = self.conv2(x)  # 64 5 4
        x = self.conv3(x)  # 128 2 1
        x = self.fc4(x.view(x.size(0), -1))
        x = self.fc5(x)
        return x
