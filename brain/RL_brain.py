#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : RL_brain.py
# @Author: 1980744819@qq.com
# @Date  : 2019-03-23
# @Desc  :
from model.model import CNN_2 as CNN
from Memory.memory import Memory, Memory_2
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import os
from settings.conf import *

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
        if os.path.isfile(save_q_eval_path):
            print('load q_eval ...')
            self.q_eval.load_state_dict(torch.load(save_q_eval_path))
        if os.path.isfile(save_q_next_path):
            print('load q_next ...')
            self.q_next.load_state_dict(torch.load(save_q_next_path))
        self.memory = Memory(memory_size, shape[0], shape[1])
        self.channals = input_args
        self.num_action = num_actions
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_count = 0

        self.op = torch.optim.Adam(self.q_eval.parameters(), lr=learning_rate)  # optimize all cnn parameters
        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.CrossEntropyLoss()
        self.save_step = 2000
        self.learn_step = 0

    def choose_action(self, obs):
        if np.random.uniform() < self.epsilon and obs.shape[0] == self.channals:
            obs = torch.FloatTensor(obs)
            obs = obs.unsqueeze(0).type(dtype)
            out = self.q_eval(obs / 255.0)
            return np.argmax(out.detach().cpu()).item()
        return np.random.randint(0, self.num_action - 1)

    def store_transition(self, obs, action, reward, obs_):
        self.memory.store_transition(obs, action, reward, obs_)

    def learn(self):
        if self.learn_step_count == self.replace_target_iter:
            self.learn_step_count = 0
            self.q_next.load_state_dict(self.q_eval.state_dict())
        if self.learn_step == self.save_step:
            torch.save(self.q_eval.state_dict(), save_q_eval_path)
            torch.save(self.q_next.state_dict(), save_q_next_path)
            self.learn_step = 0
        obs_batch, act_batch, reward_batch, obs__batch = self.memory.get_memory(self.batch_size, self.channals)
        obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype))
        obs__batch = Variable(torch.from_numpy(obs__batch).type(dtype))
        # q_next = self.q_next(torch.FloatTensor(obs__batch))
        # q_eval = self.q_eval(torch.FloatTensor(obs_batch))
        # reward_batch = torch.FloatTensor(reward_batch)
        q_next = self.q_next(obs__batch / 255.0)
        q_eval = self.q_eval(obs_batch / 255.0)
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
        self.learn_step += 1

    def get_last_memory(self):
        res = self.memory.get_last_frame(self.channals)
        return res


class Brain_2:
    def __init__(self,
                 memory_size,
                 input_args,
                 num_actions,
                 shape,
                 learning_rate,
                 reward_decay,
                 e_greedy,
                 e_greedy_increment,
                 e_greedy_start,
                 batch_size,
                 replace_target_iter):

        self.q_eval = CNN(in_channels=input_args, num_action=num_actions).type(dtype)
        self.q_next = CNN(in_channels=input_args, num_action=num_actions).type(dtype)
        if os.path.isfile(save_q_eval_path):
            print('load q_eval ...')
            self.q_eval.load_state_dict(torch.load(save_q_eval_path))
        if os.path.isfile(save_q_next_path):
            print('load q_next ...')
            self.q_next.load_state_dict(torch.load(save_q_next_path))
        self.memory = Memory_2(memory_size, shape[0], shape[1], input_args)
        self.channels = input_args
        self.num_action = num_actions
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = e_greedy_start
        self.learn_step_count = 0

        self.op = torch.optim.Adam(self.q_eval.parameters(), lr=learning_rate)  # optimize all cnn parameters
        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.CrossEntropyLoss()
        self.save_step = 2000
        self.learn_step = 0

    def choose_action(self, obs):
        if np.random.uniform() < self.epsilon and obs.shape[0] == self.channels:
            obs = torch.FloatTensor(obs)
            obs = obs.unsqueeze(0).type(dtype)
            out = self.q_eval(obs / 255.0)
            return np.argmax(out.detach().cpu()).item()
        return np.random.randint(0, self.num_action - 1)

    def store_transition(self, action, reward, obs_):
        self.memory.store_transition(action, reward, obs_)

    def learn(self):
        if self.learn_step_count == self.replace_target_iter:
            self.learn_step_count = 0
            self.q_next.load_state_dict(self.q_eval.state_dict())
        if self.learn_step == self.save_step:
            torch.save(self.q_eval.state_dict(), save_q_eval_path)
            torch.save(self.q_next.state_dict(), save_q_next_path)
            self.learn_step = 0
        obs_batch, act_batch, reward_batch, obs_batch_ = self.memory.get_memory(self.batch_size)
        obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype))
        obs_batch_ = Variable(torch.from_numpy(obs_batch_).type(dtype))
        # q_next = self.q_next(torch.FloatTensor(obs__batch))
        # q_eval = self.q_eval(torch.FloatTensor(obs_batch))
        # reward_batch = torch.FloatTensor(reward_batch)
        q_next = self.q_next(obs_batch_ / 255.0)
        q_eval = self.q_eval(obs_batch / 255.0)
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
        self.learn_step += 1

    def get_last_memory(self):
        res = self.memory.get_last_frame()
        return res

    def store_start_frame(self, obs):
        self.memory.store_frame(obs)

    def double_learn(self):
        if self.learn_step_count == self.replace_target_iter:
            self.learn_step_count = 0
            self.q_next.load_state_dict(self.q_eval.state_dict())
        if self.learn_step == self.save_step:
            torch.save(self.q_eval.state_dict(), 'q_eval.pkl')
            torch.save(self.q_next.state_dict(), 'q_next.pkl')
            self.learn_step = 0
        obs_batch, act_batch, reward_batch, obs_batch_ = self.memory.get_memory(self.batch_size)
        obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype))
        obs_batch_ = Variable(torch.from_numpy(obs_batch_).type(dtype))
        # q_next = self.q_next(torch.FloatTensor(obs_batch_))
        # q_eval = self.q_eval(torch.FloatTensor(obs_batch))
        # reward_batch = torch.FloatTensor(reward_batch)
        q_next = self.q_next(obs_batch_ / 255.0)
        q_eval_next = self.q_eval(obs_batch_ / 255.0)
        q_eval = self.q_eval(obs_batch / 255.0)

        reward_batch = torch.from_numpy(reward_batch)

        if USE_CUDA:
            # act_batch = act_batch.cuda()
            reward_batch = reward_batch.cuda()

        q_target = q_eval.clone().detach()
        batch_index = np.arange(self.batch_size)
        max_act_q_eval_next = torch.argmax(q_eval_next, dim=1)
        print(max_act_q_eval_next)
        select_q_next = q_next[batch_index, max_act_q_eval_next]
        q_target[batch_index, act_batch] = reward_batch.float() + self.gamma * select_q_next
        loss = self.loss_func(q_eval, q_target)
        self.op.zero_grad()
        loss.backward()
        self.op.step()

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_count += 1
        self.learn_step += 1
