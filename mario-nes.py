#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : mario-nes.py
# @Author: zixiao@ainirobot.com
# @Date  : 2019-03-19
# @Desc  :
import retro
import numpy as np
import random

env = retro.make(game='SuperMarioBros-Nes')
obs = env.reset()
env.render()
while True:
    # a = env.action_space.sample()
    a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    b = [0, 4, 6, 7, 8]
    # 0 b
    # 1 noop
    # 4 up
    # 5 down
    # 6 left
    # 7 right
    # 8 a
    c = random.sample(b, random.randint(1, 3))
    # # b = random.randint(0, 8)
    a[c] = 1
    obs, re, done, info = env.step(a)
    print(a)
    print(info)
    # obs, re, done, info = env.step(0)
    env.render()
    if done:
        env.reset()
env.close()
