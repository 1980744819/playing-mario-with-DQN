#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : memory.py
# @Author: zixiao@ainirobot.com
# @Date  : 2019-03-23
# @Desc  :
import numpy as np


class Memory():
    def __init__(self, size, w, h):
        self.size = size
        self.index = 0
        self.count = 0
        self.num_in_memory = 0
        self.obs = np.zeros((size, w, h))
        self.actions = np.zeros((size,))
        self.rewards = np.zeros((size,))
        self.obs_ = np.zeros((size, w, h))
        self.obs_shape = [w, h]

    def store_transition(self, obs, action, reward, obs_):
        self.obs[self.index] = obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.obs_[self.index] = obs_
        self.index += 1
        self.index = self.index % self.size
        self.count += 1
        self.num_in_memory = min(self.size, self.count)

    def get_memory(self, batch_size, frame_len):
        nums = np.random.choice(self.num_in_memory, size=batch_size)
        obs_frames = np.zeros((batch_size, frame_len, self.obs_shape[0], self.obs_shape[1]))
        obs_after_frames = np.zeros((batch_size, frame_len, self.obs_shape[0], self.obs_shape[1]))
        # action_batch = np.zeros(batch_size)
        # rewards = np.zeros(batch_size)
        # obs_s = np.zeros((batch_size, self.obs_shape[0], self.obs_shape[1]))
        for i in range(nums.shape[0]):
            start = nums[i] - frame_len + 1
            end = nums[i]
            after_start = nums[i]
            after_end = nums[i] + frame_len - 1
            if start < 0:
                start += self.num_in_memory
                obs_frames[i] = np.concatenate((self.obs[start:self.num_in_memory], self.obs[0:end + 1]))
            else:
                obs_frames[i] = self.obs[start:end + 1]
            if after_end > self.num_in_memory:
                after_end -= self.num_in_memory
                obs_after_frames[i] = np.concatenate(
                    (self.obs_[after_start:self.num_in_memory], self.obs_[0:after_end + 1]))
            else:
                print('after_start,after_end', after_start, after_end)
                obs_after_frames[i] = self.obs_[after_start:after_end + 1]
        action_batch = self.actions[nums]
        rewards = self.rewards[nums]
        obs_s = self.obs_[nums]
        return obs_frames, action_batch, rewards, obs_after_frames

    def get_last_frame(self, frame_len):
        end = self.index
        start = end - frame_len + 1
        if start < 0:
            if self.num_in_memory > self.index:
                start += self.num_in_memory
                return np.concatenate((self.obs[start:self.num_in_memory], self.obs[0:end + 1]))
            else:
                start = 0
        return self.obs[start:end + 1]
