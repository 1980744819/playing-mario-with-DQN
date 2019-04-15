#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : memory.py
# @Author: zixiao
# @Date  : 2019-04-15
# @Desc  :
import numpy as np


class Memory:
    def __init__(self, batch_size, frame_len):
        self.log_probabilities = []
        self.rewards = []
        self.actions = []
        self.states = []
        self.masks = []
        self.values = []
        self.batch_size = batch_size
        self.frame_len = frame_len

    def clear(self):
        self.actions.clear()
        self.rewards.clear()
        self.actions.clear()
        self.states.clear()
        self.masks.clear()
        self.log_probabilities.clear()
        self.values.clear()

    def store_transition(self, state, action, reward, done, log_probability, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(done)
        self.log_probabilities.append(log_probability)
        self.values.append(value)

    def get_batch_memory(self):
        pass

    def get_last_frame(self):
        return np.asarray(self.states[-4:])
