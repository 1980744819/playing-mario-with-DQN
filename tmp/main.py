#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: 1980744819@qq.com
# @Date  : 2019-03-23
# @Desc  :
from env.mario import env
from PIL import Image
import numpy as np
from utils.img import RGB_to_gary
from tmp.RL_brain import Brian
from settings.action_space import Actions

if __name__ == '__main__':
    obs = env.reset()
    actions = Actions()
    brain = Brian(memory_size=1000000,
                  input_args=8,
                  num_actions=14,
                  shape=obs.shape,
                  learning_rate=0.0001,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  e_greedy_increment=0.000001,
                  batch_size=32,
                  replace_target_iter=1000)
    obs = RGB_to_gary(obs)
    env.render()
    step = 0
    last_info = {
        'coins': 0,
        'time': 400,
        'lives': 2,
        'score': 0,
        'xscrollLo': 0
    }
    while True:
        ons_frames = brain.get_last_memory()
        action = brain.choose_action(ons_frames)
        obs_, reward, done, info = env.step(actions.mapping[action])
        re = info['coins'] - last_info['coins'] + info['time'] - last_info['time'] + (
                info['lives'] - last_info['lives']) * 10 + info['score'] - last_info['score'] + info['xscrollLo'] - \
             last_info['xscrollLo'] - 0.1
        last_info = info
        reward = re

        print(action, '\t', re, '\t', info)
        obs_ = RGB_to_gary(obs_)
        brain.store_transition(obs, action, reward, obs_)
        if step > 2000 and step % 100 == 0:
            brain.learn()
        env.render()
        if done:
            obs_ = env.reset()
            obs_ = RGB_to_gary(obs_)
        obs = obs_
        step += 1
    print(1)
