#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : PER_brian.py
# @Author: zixiao
# @Date  : 2019-03-30
# @Desc  :
import numpy as np
from model.model import CNN_2 as CNN
from settings.conf import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import os

USE_CUDA = torch.cuda.is_available()
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


class AbsErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_target, q_eval):
        return torch.sum(torch.abs(q_target - q_target), dim=1)


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduce=False, reduction='none')

    def forward(self, IS_weights, q_target, q_eval):
        x = self.loss(q_target, q_eval)
        return torch.mean(IS_weights * x)


class Variable(Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


class SumTree(object):
    data_index = 0

    def __init__(self, size, frame_len, w, h):
        self.data_size = size
        self.tree_size = 2 * size - 1
        self.tree = np.zeros(self.tree_size)
        self.data_obs = np.zeros((size, frame_len, w, h), dtype=np.uint8)
        self.data_reward = np.zeros(size, dtype=np.float32)
        self.data_action = np.zeros(size, dtype=np.uint8)

    def add(self, tree_point, action, reward, obs_):
        tree_index = self.data_index + self.data_size - 1
        self.data_action[self.data_index] = action
        self.data_reward[self.data_index] = reward
        self.data_index = int((self.data_index + 1) % self.data_size)
        self.data_obs[self.data_index] = obs_
        self.update(tree_index, tree_point)

    def update(self, tree_index, pointer):
        change = pointer - self.tree[tree_index]
        self.tree[tree_index] = pointer
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    @property
    def total_weight(self):
        return self.tree[0]

    def get_leaf(self, value):
        parent_index = 0
        while True:
            left_index = 2 * parent_index + 1
            right_index = left_index + 1
            if left_index >= self.tree_size:
                break
            else:
                if value <= self.tree[left_index]:
                    parent_index = left_index
                else:
                    value -= self.tree[left_index]
                    parent_index = right_index
        leaf_index = parent_index
        data_index = leaf_index - (self.data_size - 1)
        data_index_ = (data_index + 1) % self.data_size

        return leaf_index, self.tree[leaf_index], self.data_obs[data_index], self.data_action[data_index], \
               self.data_reward[data_index], self.data_obs[data_index_]

    def store_frame(self, obs):
        self.data_obs[self.data_index] = obs

    def get_last_frame(self):
        return self.data_obs[self.data_index]


class Memory(object):
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1

    def __init__(self, size, frame_len, w, h):
        self.size = size
        self.frame_len = frame_len
        self.w = w
        self.h = h
        self.tree = SumTree(size=self.size, frame_len=self.frame_len, w=self.w, h=self.h)

    def store_transition(self, action, reward, obs_):
        max_leaf_weight = np.max(self.tree.tree[-self.tree.data_size:])
        if max_leaf_weight == 0:
            max_leaf_weight = self.abs_err_upper
        self.tree.add(max_leaf_weight, action, reward, obs_)

    def get_memory(self, batch_size):
        batch_leaf_index = np.zeros(batch_size, dtype=np.int32)
        batch_action = np.zeros(batch_size, dtype=np.uint8)
        batch_reward = np.zeros(batch_size, dtype=np.float32)
        batch_obs = np.zeros((batch_size, self.frame_len, self.w, self.h), dtype=np.uint8)
        batch_obs_ = np.zeros((batch_size, self.frame_len, self.w, self.h), dtype=np.uint8)
        IS_weights = np.zeros((batch_size, 1))

        priority_segment = self.tree.total_weight / batch_size
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])
        min_probability = np.min(self.tree.tree[-self.tree.data_size:]) / self.tree.total_weight
        for i in range(batch_size):
            low = priority_segment * i
            high = priority_segment * (i + 1)
            value = np.random.uniform(low, high)
            leaf_index, leaf_value, obs, action, reward, obs_ = self.tree.get_leaf(value)
            probability = leaf_value / self.tree.total_weight
            IS_weights[i, 0] = np.power(probability / min_probability, -self.beta)
            batch_leaf_index[i] = leaf_index

            batch_obs[i] = obs
            batch_obs_[i] = obs_
            batch_action[i] = action
            batch_reward[i] = reward
        return batch_leaf_index, IS_weights, batch_obs, batch_action, batch_reward, batch_obs_

    def batch_update(self, tree_index, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for t_index, p in zip(tree_index, ps):
            self.tree.update(t_index, p)

    def store_frame(self, obs):
        self.tree.store_frame(obs)

    def get_last_frame(self):
        return self.tree.get_last_frame()


class Brain:
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
        self.memory = Memory(size=memory_size, frame_len=input_args, w=shape[0], h=shape[1])
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
        self.loss_func = Loss()
        self.abs_errors_func = AbsErrorLoss()
        # self.loss_func = nn.CrossEntropyLoss()
        self.abs_errors = torch
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
        batch_leaf_index, IS_weights, batch_obs, batch_action, batch_reward, batch_obs_ = self.memory.get_memory(
            self.batch_size)
        batch_obs = Variable(torch.from_numpy(batch_obs).type(dtype))
        batch_obs_ = Variable(torch.from_numpy(batch_obs_).type(dtype))
        IS_weights = Variable(torch.from_numpy(IS_weights).type(dtype))
        q_next = self.q_next(batch_obs_ / 255.0)
        q_eval = self.q_eval(batch_obs / 255.0)
        reward_batch = torch.from_numpy(batch_reward)
        if USE_CUDA:
            # act_batch = act_batch.cuda()
            reward_batch = reward_batch.cuda()

        q_target = q_eval.clone().detach()
        batch_index = np.arange(self.batch_size)
        q_target[batch_index, batch_action] = reward_batch.float() + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.loss_func(IS_weights, q_eval, q_target)
        abs_error_loss = self.abs_errors_func(q_target, q_eval)
        self.memory.batch_update(batch_leaf_index, abs_error_loss.detach().cpu())
        self.op.zero_grad()
        loss.backward()
        self.op.step()

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_count += 1
        self.learn_step += 1

    def store_start_frame(self, obs):
        self.memory.store_frame(obs)

    def get_last_memory(self):
        return self.memory.get_last_frame()

    def double_learn(self):
        if self.learn_step_count == self.replace_target_iter:
            self.learn_step_count = 0
            self.q_next.load_state_dict(self.q_eval.state_dict())
        if self.learn_step == self.save_step:
            torch.save(self.q_eval.state_dict(), save_q_eval_path)
            torch.save(self.q_next.state_dict(), save_q_next_path)
            self.learn_step = 0
        batch_leaf_index, IS_weights, batch_obs, batch_action, batch_reward, batch_obs_ = self.memory.get_memory(
            self.batch_size)
        batch_obs = Variable(torch.from_numpy(batch_obs).type(dtype))
        batch_obs_ = Variable(torch.from_numpy(batch_obs_).type(dtype))
        IS_weights = Variable(torch.from_numpy(IS_weights).type(dtype))
        q_next = self.q_next(batch_obs_ / 255.0)
        q_eval_next = self.q_eval(batch_obs_ / 255.0)
        q_eval = self.q_eval(batch_obs / 255.0)
        reward_batch = torch.from_numpy(batch_reward)
        if USE_CUDA:
            # act_batch = act_batch.cuda()
            reward_batch = reward_batch.cuda()

        q_target = q_eval.clone().detach()
        batch_index = np.arange(self.batch_size)
        max_act_q_eval_next = torch.argmax(q_eval_next, dim=1)
        print(max_act_q_eval_next)
        select_q_next = q_next[batch_index, max_act_q_eval_next]
        q_target[batch_index, batch_action] = reward_batch.float() + self.gamma * select_q_next
        loss = self.loss_func(IS_weights, q_eval, q_target)
        abs_error_loss = self.abs_errors_func(q_target, q_eval)
        self.memory.batch_update(batch_leaf_index, abs_error_loss.detach().cpu())
        self.op.zero_grad()
        loss.backward()
        self.op.step()

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_count += 1
        self.learn_step += 1
