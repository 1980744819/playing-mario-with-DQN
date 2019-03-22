#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : replay_buffer.py
# @Author: zixiao@ainirobot.com
# @Date  : 2019-03-20
# @Desc  :
import numpy as np
import random


def sample_n_unique(sampling_f, n):
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class Replay_buffer(object):
    def __init__(self, size=2000, frame_history_len=4):
        self.size = size
        self.frame_his_len = frame_history_len
        self.index = 0
        self.buffer_num = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None
        self.count = 0

    def store_frame(self, frame):
        frame = frame.transpose(2, 0, 1)
        if self.obs is None:
            self.obs = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)

        self.obs[self.index] = frame
        index = self.index
        self.index = (self.index + 1) % self.size
        self.num_in_buffer = min(self.index, self.size)
        self.count += 1

        return index

    def encoder_recent_obs(self):
        end = self.index
        start = self.index - self.frame_his_len + 1
        h, w = self.obs.shape[2], self.obs.shape[3]
        if start < 0:
            start += self.size
            ans = np.concatenate((self.obs[start:-1], self.obs[0:end]), 0).reshape(-1, h, w)
        else:
            ans = self.obs[start:end + 1].reshape(-1, h, w)
        return ans

    def store_effect(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done

    def can_sample(self, batch_size):  # return True or False
        return batch_size + 1 <= self.num_in_buffer

    def sample(self, batch_size):  # return batch of (obs, act, rew, next_obs) and done_mask
        # 先確認大小足夠能抽樣才進行抽樣
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def _encode_sample(self, idxes):
        obs_batch = np.concatenate([self._encode_observation(idx)[np.newaxis, :] for idx in idxes], 0)
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[np.newaxis, :] for idx in idxes], 0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)  # 將True, False轉為1, 0

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def _encode_observation(self, idx):  # return list of frame
        end_idx = idx + 1
        start_idx = end_idx - self.frame_his_len

        # check if it is low-dimensional obs, such as RAM
        if len(self.obs.shape) == 2: return self.obs[end_idx - 1]

        # 假如buffer裡沒有足夠的frame時
        if start_idx < 0 and self.num_in_buffer != self.size: start_idx = 0

        # 標註每個start idx
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1

        # 處理頭幾個frame，因為idx出現負數時沒有frame
        missing_context = self.frame_his_len - (end_idx - start_idx)
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            # 拿別的frame來填補
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0)
        else:
            # 底下的處理可以節約30%的計算時間
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)
