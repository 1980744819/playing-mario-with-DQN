#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : memory.py
# @Author: zixiao
# @Date  : 2019-04-01
# @Desc  :
import numpy as np


class SumTree(object):
    data_index = 0

    def __init__(self, size, frame_len, w, h):
        self.data_size = size
        self.tree_size = 2 * size - 1
        self.tree = np.zeros(self.tree_size)
        self.data_obs = np.zeros((size, w, h), dtype=np.uint8)
        self.data_reward = np.zeros(size, dtype=np.float32)
        self.data_action = np.zeros(size, dtype=np.uint8)
        self.frame_len = frame_len

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
        if leaf_index == 0:
            leaf_index = 1
        data_index = leaf_index - (self.data_size - 1)
        # data_index_ = (data_index + 1) % self.data_size
        obs_frame, obs_frame_ = self.get_frame(data_index)

        return leaf_index, self.tree[leaf_index], obs_frame, self.data_action[data_index], \
               self.data_reward[data_index], obs_frame_

    def store_obs(self, obs):
        self.data_obs[self.data_index] = obs

    def get_last_frame(self):
        start = self.data_index - self.frame_len + 1
        end = self.data_index
        if start < 0:
            start += self.data_size
            obs_frame = np.concatenate((self.data_obs[start:],
                                        self.data_obs[0:end + 1]))
        else:
            obs_frame = self.data_obs[start:end + 1]
        return obs_frame

    def get_frame(self, data_index):
        obs_start = data_index - self.frame_len + 1
        obs_end = data_index
        obs_start_ = int((data_index + 1) % self.data_size)
        obs_end_ = obs_start_ + self.frame_len - 1
        if obs_start < 0:
            obs_start += self.data_size
            obs_frame = np.concatenate((self.data_obs[obs_start:], self.data_obs[0:obs_end + 1]))
        else:
            obs_frame = self.data_obs[obs_start:obs_end + 1]
        if obs_end_ >= self.data_size:
            obs_end_ -= self.data_size
            obs_frame_ = np.concatenate((self.data_obs[obs_start_:], self.data_obs[0:obs_end_ + 1]))
        else:
            obs_frame_ = self.data_obs[obs_start_:obs_end_ + 1]
        return obs_frame, obs_frame_


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
        self.tree.store_obs(obs)

    def get_last_frame(self):
        return self.tree.get_last_frame()