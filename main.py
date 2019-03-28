#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: 1980744819@qq.com
# @Date  : 2019-03-23
# @Desc  :
from env.gym_super_mario_bros import env
from utils.img import RGB_to_gray
from brain.RL_brain import Brian

from settings.action_space import Actions

if __name__ == '__main__':
    obs = env.reset()
    actions = Actions()
    frame_len = 8
    start_step = 1000
    brain = Brian(memory_size=1000000,
                  input_args=frame_len,
                  num_actions=7,
                  shape=obs.shape,
                  learning_rate=0.0001,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  e_greedy_increment=0.001,
                  batch_size=32,
                  replace_target_iter=1000)
    obs = RGB_to_gray(obs)
    a = env.render()
    # actions = env.action_space
    step = 0
    last_info = {
        'coins': 0,
        'time': 400,
        'lives': 2,
        'score': 0,
        'xscrollLo': 0
    }
    max_re = 0
    i = 0
    while True:
        print(i)
        i += 1
        ons_frames = brain.get_last_memory()
        action = brain.choose_action(ons_frames)
        obs_, reward, done, info = env.step(actions.mapping[12])
        re = info['coins'] - last_info['coins'] + info['time'] - last_info['time'] + (
                info['lives'] - last_info['lives']) * 10 + info['score'] - last_info['score'] + info['xscrollLo'] - \
             last_info['xscrollLo'] - 0.1
        last_info = info
        # reward = re
        # max_re = max(max_re, re)
        # print(action, '\t', re, '\t', info)
        # print(re, '\t', max_re)
        obs_ = RGB_to_gray(obs_)
        brain.store_transition(obs, action, reward, obs_)
        if step > 2000 and step % 100 == 0:
            brain.learn()
        env.render()
        if done:
            obs_ = env.reset()
            obs_ = RGB_to_gray(obs_)
        obs = obs_
        step += 1
    print(1)


def get_render(frame_len, env):
    res = []
    for i in range(frame_len):
        res.append(env.render())
    return res
