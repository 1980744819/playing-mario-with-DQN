#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : retro_dueling.py
# @Author: zixiao
# @Date  : 2019-04-04
# @Desc  :
from env.mario import env, actions, sample
from dueling.dueling_brain import Brain
import numpy as np

import PIL.Image as Image


def RGB2gray(obs):
    img = Image.fromarray(obs).crop((0, 24, 240, 224)).resize((200, 200))
    img = img.convert('L')
    # img.show()
    return np.asarray(img)


if __name__ == '__main__':
    state = env.reset()  # 224 240 3
    state = RGB2gray(state)
    frame_len = 4
    memory_size = 10000
    brain = Brain(memory_size=memory_size,
                  input_args=frame_len,
                  num_actions=7,
                  shape=state.shape,
                  learning_rate=0.00025,
                  reward_decay=0.99,
                  e_greedy=0.9,
                  e_greedy_increment=0.001,
                  e_greedy_start=0,
                  batch_size=32,
                  replace_target_iter=10000)
    last_info = {
        'time': 400,
        'lives': 2,
        'xscrollLo': 0
    }
    brain.store_start_frame(state)
    for i in range(int(memory_size / 10) + 5):
        act_index = sample()
        obs, re, done, info = env.step(actions[act_index])

        a = info['time'] - last_info['time']
        b = info['lives'] - last_info['lives']
        c = info['xscrollLo'] - last_info['xscrollLo']
        if a != 0 and a != -1:
            a = 0
        if b != 0 and b != -1:
            b = -1
        if c < -20:
            c = 0
        b *= 15
        reward = (a + b + c) / 15.0
        print(act_index, reward, i)
        if reward < -0.6:
            print(last_info)
            print(info)

        if done:
            obs = env.reset()
            info['time'] = 0
            info['lives'] = (info['lives'] - 1) % 3
            info['xscrollLo'] = 0
        last_info = info

        obs = RGB2gray(obs)
        env.render()
        brain.store_transition(action=act_index, reward=re, obs_=obs)

    step = 1
    while True:
        last_frame = brain.get_last_memory()
        # get_gif(last_frame)
        act_index = brain.choose_action(last_frame)
        obs_, re, done, info = env.step(actions[act_index])

        a = info['time'] - last_info['time']
        b = info['lives'] - last_info['lives']
        c = info['xscrollLo'] - last_info['xscrollLo']
        if a != 0 and a != -1:
            a = 0
        if b != 0 and b != -1:
            b = -1
        if c < -20:
            c = 0
        b *= 15
        reward = (a + b + c) / 15.0
        print(act_index, reward, i)
        if reward < -0.6:
            print(last_info)
            print(info)

        if done:
            obs_ = env.reset()
            info['time'] = 0
            info['lives'] = (info['lives'] - 1) % 3
            info['xscrollLo'] = 0
        obs_ = RGB2gray(obs_)
        env.render()
        print(act_index, reward, brain.epsilon, step)
        if reward < -0.6:
            print(reward)
            print(last_info)
            print(info)

        brain.store_transition(action=act_index, reward=reward, obs_=obs_)
        last_info = info
        if step % 30 == 0:
            brain.double_learn()

        step += 1
