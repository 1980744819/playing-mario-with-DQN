#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ppo.py
# @Author: zixiao
# @Date  : 2019-04-24
# @Desc  :
import os
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from PPO_v2.models import Actor, Critic

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

save_path = './save_pkl'


class PPO:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 100000
    batch_size = 32
    num_action = 7
    gamma = 0.99

    def __init__(self):
        self.num_channel = 4
        self.actor_net = Actor(self.num_channel, self.num_action)
        self.critic_net = Critic(self.num_channel)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def select_action(self, state):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), save_path + '/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), save_path + '/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        for i in range(int(len(self.buffer)/100)):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.mse_loss(Gt_index, V)

                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1
        del self.buffer[:]
