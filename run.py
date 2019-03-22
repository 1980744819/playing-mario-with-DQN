#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : run.py
# @Author: zixiao@ainirobot.com
# @Date  : 2019-03-18
# @Desc  :
import retro
import random
from settings import Actions
from DQN import learning

BATCH_SIZE = 16
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 30000
LEARNING_STARTS = 10000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGET_UPDATE_FREQ = 3000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

# actions = Actions()
# env = retro.make(game='SuperMarioBros-Nes')

if __name__ == '__main__':
    actions = Actions()
    env = retro.make(game='SuperMarioBros-Nes')
    env.reset()
    learning(
        env=env,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        frame_history_len=FRAME_HISTORY_LEN,
        num_act=14,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        target_update_freq=TARGET_UPDATE_FREQ
    )
