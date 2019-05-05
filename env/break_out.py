#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : break_out.py
# @Author: zixiao
# @Date  : 2019-04-06
# @Desc  :
import gym
env = gym.make('Breakout-v0')
env = env.unwrapped
print(env.action_space)
# > Discrete(2)
print(env.observation_space)
if __name__ == '__main__':
    env.reset()
    for _ in range(10000):
        env.render()
        obs, re, done, info = env.step(env.action_space.sample())  # take a random action
        print(re, info)
        if done:
            env.reset()
    env.close()
