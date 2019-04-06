#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : break_out_main.py
# @Author: zixiao
# @Date  : 2019-04-06
# @Desc  :
from env.break_out import env
import PIL.Image as Image
import numpy as np
from break_out.brain import Brain


def RGB2gray(observation):
    img = Image.fromarray(observation)
    # .crop((40, 0, 120, 210))
    # img.show()
    img = img.convert('L')
    return np.asarray(img)


if __name__ == '__main__':
    state = env.reset()  # 210 160 3
    state = RGB2gray(state)
    frame_len = 4
    memory_size = 150000

    brain = Brain(memory_size=memory_size,
                  input_args=frame_len,
                  num_actions=4,
                  shape=state.shape,
                  learning_rate=0.00025,
                  reward_decay=0.99,
                  e_greedy=0.9,
                  e_greedy_increment=0.001,
                  e_greedy_start=0,
                  batch_size=32,
                  replace_target_iter=10000)
    info = None
    brain.store_start_frame(state)
    for i in range(int(memory_size / 10) + 5):
        print(i)
        action = env.action_space.sample()
        obs, re, done, info = env.step(action)
        obs = RGB2gray(obs)
        env.render()
        # re /= 15.0
        # print(re)
        brain.store_transition(action=action, reward=re, obs_=obs)
        if done:
            env.reset()
    step = 1
    # last_info = env.unwrapped._get_info()
    last_info = info
    while True:
        last_frame = brain.get_last_memory()
        # get_gif(last_frame)
        action = brain.choose_action(last_frame)
        obs_, re, done, info = env.step(action)
        if done:
            obs_ = env.reset()
        obs_ = RGB2gray(obs_)
        env.render()
        tmp = info['ale.lives'] - last_info['ale.lives']
        if tmp != 0 and tmp != 1:
            tmp = -1
        # reward = re / 15.0
        re += tmp
        print(action, re, brain.epsilon, step)
        if re < -0.6:
            print(re)
            print(last_info)
            print(info)

        brain.store_transition(action=action, reward=re, obs_=obs_)
        last_info = info
        if step % 30 == 0:
            brain.double_learn()

        step += 1
