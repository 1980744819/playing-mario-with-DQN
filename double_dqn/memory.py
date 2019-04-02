#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : memory.py
# @Author: zixiao
# @Date  : 2019-04-02
# @Desc  :
import numpy as np


class Memory:
    def __init__(self, size, w, h, frame_len):
        self.size = size
        self.index = 0
        self.count = 0
        self.num_in_memory = 0
        self.frame_len = frame_len
        self.obs = np.zeros((size, w, h), dtype=np.uint8)
        self.actions = np.zeros((size,), dtype=np.uint8)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.obs_shape = [w, h]
        self.w = w
        self.h = h

    def store_transition(self, action, reward, obs_):
        index = int((self.index + 1) % self.size)

        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.obs[index] = obs_
        self.index = index
        self.count += 1
        self.num_in_memory = min(self.size, self.count)

    def get_memory(self, batch_size):
        nums = np.random.choice(self.num_in_memory, size=batch_size)
        obs_batch = np.zeros((batch_size, self.frame_len, self.w, self.h))
        obs_batch_ = np.zeros((batch_size, self.frame_len, self.w, self.h))
        for i in range(len(nums)):
            obs_start = nums[i] - self.frame_len + 1
            obs_end = nums[i]
            if obs_start < 0:
                obs_start += self.num_in_memory
                obs_batch[i] = np.concatenate((self.obs[obs_start:self.num_in_memory ], self.obs[0:obs_end + 1]))
            else:
                obs_batch[i] = self.obs[obs_start:obs_end + 1]
            obs_start_ = nums[i]
            obs_end_ = nums[i] + self.frame_len - 1
            if obs_end_ >=self.num_in_memory:
                obs_end_ -= self.num_in_memory
                obs_batch_[i] = np.concatenate((self.obs[obs_start_:self.num_in_memory ], self.obs[0:obs_end_ + 1]))
            else:
                obs_batch_[i] = self.obs[obs_start_:obs_end_ + 1]
        action_batch = self.actions[nums]
        reward_batch = self.rewards[nums]
        return obs_batch, action_batch, reward_batch, obs_batch_

    def get_last_frame(self):
        start = self.index - self.frame_len + 1
        end = self.index
        if start < 0:
            start += self.num_in_memory
            obs_frame = np.concatenate((self.obs[start:self.num_in_memory + 1], self.obs[0:end + 1]))
        else:
            obs_frame = self.obs[start:end + 1]
        return obs_frame

    def store_obs(self, obs):
        self.obs[self.index] = obs
