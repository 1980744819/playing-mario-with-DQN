#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : mario.py
# @Author: 1980744819@qq.com
# @Date  : 2019-03-21
# @Desc  :
import retro
import random

env = retro.make(game='SuperMarioBros-Nes')
actions = {
    #   0  1  2  3  4  5  6  7  8
    #   b  n        u  d  l  r  a
    0: [0, 1, 0, 0, 0, 0, 0, 0, 0],  # noop
    1: [0, 0, 0, 0, 0, 0, 1, 0, 0],  # left
    2: [0, 0, 0, 0, 0, 0, 0, 1, 0],  # right
    3: [0, 0, 0, 0, 0, 0, 0, 0, 1],  # a
    4: [0, 0, 0, 0, 0, 0, 0, 1, 1],  # right + a
    5: [1, 0, 0, 0, 0, 0, 0, 1, 0],  # right + b
    6: [1, 0, 0, 0, 0, 0, 0, 1, 1],  # right + a +b
}


def sample():
    return random.randint(0, 6)


if __name__ == '__main__':
    obs = env.reset()
    for i in range(100000):
        act_index = sample()
        obs, reward, done, info = env.step(actions[act_index])
        env.render()
        if done:
            env.reset()
    for i in range(100):
        act_index = sample()
        obs, reward, done, info = env.step(actions[act_index])
        env.render()
        if done:
            env.reset()
