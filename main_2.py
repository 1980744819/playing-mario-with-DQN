#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main_2.py
# @Author: zixiao
# @Date  : 2019-03-28
# @Desc  :
from env.gym_super_mario_bros import env
from utils.img import RGB_to_gray
from brain.RL_brain import Brain_2 as Brain
import numpy as np
from PIL import Image

from settings.action_space import Actions

if __name__ == '__main__':
    state = env.reset()
    state = RGB_to_gray(state)
    frame_len = 4
    learning_start = 2000
    brain = Brain(memory_size=100000,
                  input_args=frame_len,
                  num_actions=7,
                  shape=state.shape,
                  learning_rate=0.00025,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  e_greedy_increment=0.001,
                  e_greedy_start=0,
                  batch_size=32,
                  replace_target_iter=1000)
    obs = np.zeros((frame_len, state.shape[0], state.shape[1]))
    for i in range(frame_len):
        env.render()
        ob = env.unwrapped.screen
        ob = RGB_to_gray(ob)
        obs[i] = ob
    brain.store_start_frame(obs)
    last_info = env.unwrapped._get_info()
    for i in range(learning_start):
        action = env.action_space.sample()
        states, re, done, info = env.step(action)
        states = RGB_to_gray(states)
        obs[0] = states
        env.render()
        for j in range(1, frame_len):
            env.render()
            state = env.unwrapped.screen
            state = RGB_to_gray(state)
            obs[j] = state
        info = env.unwrapped._get_info()
        reward = info['x_pos'] - last_info['x_pos'] + info['time'] - last_info['time'] + (
                info['life'] - last_info['life']) * 15-0.1
        reward = reward / 15.0
        brain.store_transition(reward=reward, action=action, obs_=obs)
        last_info = info
        if done:
            env.reset()
    step = 1
    while True:
        last_frame = brain.get_last_memory()
        action = brain.choose_action(last_frame)
        obs_, re, done, info = env.step(action)
        obs_ = RGB_to_gray(obs_)
        obs[0] = obs_
        env.render()
        for j in range(1, frame_len):
            env.render()
            state = env.unwrapped.screen
            state = RGB_to_gray(state)
            obs[j] = state
        info = env.unwrapped._get_info()
        reward = info['x_pos'] - last_info['x_pos'] + info['time'] - last_info['time'] + (
                info['life'] - last_info['life']) * 15-0.1
        reward = reward / 15.0
        print(action, reward, brain.epsilon, step)
        brain.store_transition(reward=reward, action=action, obs_=obs)
        last_info = info
        if step % 200 == 0:
            brain.double_learn()
        if done:
            env.reset()
            last_info['time'] = 0
            last_info['x_pos'] = 0
        step += 1
