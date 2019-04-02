#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : double_main.py
# @Author: zixiao
# @Date  : 2019-04-02
# @Desc  :
from env.gym_super_mario_bros import env
from double_dqn.brain import Brain
import numpy as np
from utils.img import get_gif
import PIL.Image as Image


def RGB2gray(obs):
    img = Image.fromarray(obs).crop((0, 40, 256, 240)).resize((200, 200))
    img = img.convert('L')
    return np.asarray(img)


if __name__ == '__main__':
    state = env.reset()
    state = RGB2gray(state)
    frame_len = 4
    memory_size = 1500
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
    brain.store_start_frame(state)
    for i in range(int(memory_size / 3) + 5):
        print(i)
        action = env.action_space.sample()
        obs, re, done, info = env.step(action)
        obs = RGB2gray(obs)
        env.render()
        re /= 15.0
        brain.store_transition(action=action, reward=re, obs_=obs)
        if done:
            env.reset()
    step = 1
    last_info = env.unwrapped._get_info()
    while True:
        last_frame = brain.get_last_memory()
        # get_gif(last_frame)
        action = brain.choose_action(last_frame)
        obs_, re, done, info = env.step(action)
        if done:
            obs_ = env.reset()
        obs_ = RGB2gray(obs_)
        env.render()
        reward = re / 15.0
        print(action, reward, brain.epsilon, step)
        if reward < -0.6:
            print(reward)
            print(last_info)
            print(info)

        brain.store_transition(action=action, reward=reward, obs_=obs_)
        last_info = info
        if step % 30 == 0:
            brain.double_learn()

        step += 1
