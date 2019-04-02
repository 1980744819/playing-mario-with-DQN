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
        self.obs = np.zeros((size, frame_len, w, h), dtype=np.uint8)
        self.actions = np.zeros((size,), dtype=np.uint8)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.obs_shape = [w, h]

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
        obs_frames = self.obs[nums]
        action_batch = self.actions[nums]
        rewards = self.rewards[nums]
        nums += 1
        nums %= self.size

        obs_after_frames = self.obs[nums]

        return obs_frames, action_batch, rewards, obs_after_frames

    def get_last_frame(self):
        return self.obs[(self.index - 1) % self.size]

    def store_frame(self, obs):
        self.obs[self.index] = obs
