#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : RL_brain.py
# @Author: zixiao@ainirobot.com
# @Date  : 2019-03-23
# @Desc  :
from DQN.model import CNN
from tmp.memory import Memory
import numpy as np
from settings.action_space import Actions
import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.autograd as autograd

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


class Brian:
    def __init__(self,
                 memory_size,
                 input_args,
                 num_actions,
                 shape,
                 learning_rate,
                 reward_decay,
                 e_greedy,
                 e_greedy_increment,
                 batch_size,
                 replace_target_iter):
        self.q_eval = CNN(in_channels=input_args, num_action=num_actions).type(dtype)
        self.q_next = CNN(in_channels=input_args, num_action=num_actions).type(dtype)
        self.memory = Memory(memory_size, shape[0], shape[1])
        self.channals = input_args
        self.num_action = num_actions
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.actions = Actions()
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_count = 0

        self.op = torch.optim.Adam(self.q_eval.parameters(), lr=learning_rate)  # optimize all cnn parameters
        self.loss_func = nn.MSELoss()

    def choose_action(self, obs):
        if np.random.uniform() < self.epsilon and obs.shape[0] == self.channals:
            obs = torch.FloatTensor(obs)
            obs = obs.unsqueeze(0).type(dtype)
            out = self.q_eval(obs)
            return np.argmax(out.detach().cpu()).item()
        return np.random.randint(0, self.num_action - 1)

    def store_transition(self, obs, action, reward, obs_):
        self.memory.store_transition(obs, action, reward, obs_)

    def learn(self):
        if self.learn_step_count == self.replace_target_iter:
            self.learn_step_count = 0
            self.q_next.load_state_dict(self.q_eval.state_dict())
        obs_batch, act_batch, reward_batch, obs__batch = self.memory.get_memory(self.batch_size, self.channals)
        obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype))
        obs__batch = Variable(torch.from_numpy(obs__batch).type(dtype))
        # q_next = self.q_next(torch.FloatTensor(obs__batch))
        # q_eval = self.q_eval(torch.FloatTensor(obs_batch))
        # reward_batch = torch.FloatTensor(reward_batch)
        q_next = self.q_next(obs__batch)
        q_eval = self.q_eval(obs_batch)
        reward_batch = torch.from_numpy(reward_batch)

        if USE_CUDA:
            # act_batch = act_batch.cuda()
            reward_batch = reward_batch.cuda()

        q_target = q_eval.clone().detach()
        batch_index = np.arange(self.batch_size)
        q_target[batch_index, act_batch] = reward_batch.float() + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.loss_func(q_eval, q_target)
        self.op.zero_grad()
        loss.backward()
        self.op.step()

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_count += 1

    def get_last_memory(self):
        res = self.memory.get_last_frame(self.channals)
        return res
