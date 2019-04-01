#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: zixiao
# @Date  : 2019-04-01
# @Desc  :
import torch
import torch.nn as nn


class DuelingCNN(nn.Module):
    def __init__(self, in_channels, num_action):
        super(DuelingCNN, self).__init__()
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
        self.Advantage_fc4 = nn.Linear(in_features=2 * 2 * 128, out_features=256)
        self.Advantage_fc5 = nn.Linear(in_features=256, out_features=num_action)

        self.Value_fc4 = nn.Linear(in_features=2 * 2 * 128, out_features=256)
        self.Value_fc5 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):  # 8 200 200
        x = self.conv1(x)  # 32 24,24
        x = self.conv2(x)  # 64 5 5
        x = self.conv3(x)  # 128 2 2
        A = self.Advantage_fc4(x.view(x.size(0), -1))
        A = self.Advantage_fc5(A)
        V = self.Value_fc4(x.view(x.size(0), -1))
        V = self.Value_fc5(V)
        A_mean = torch.mean(A, dim=1, keepdim=True)
        return V + (A - A_mean)
