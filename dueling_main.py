#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : dueling_main.py
# @Author: zixiao
# @Date  : 2019-04-01
# @Desc  :
import copy

import PIL.Image as Image
import numpy as np

# from utils.img import RGB_to_gray
from dueling.dueling_brain import Brain
from env.gym_super_mario_bros import env
from utils.img import get_gif


def RGB2gray(obs):
    img = Image.fromarray(obs).crop((0, 40, 256, 240)).resize((200, 200))
    img = img.convert('L')
    return np.asarray(img)


if __name__ == '__main__':
    state = env.reset()
    w, h, deep = state.shape
    state = RGB2gray(state)
    frame_len = 4
    memory_size = 1500
    brain = Brain(memory_size=memory_size,
                  input_args=frame_len,
                  num_actions=7,
                  shape=state.shape,
                  learning_rate=0.00025,
                  reward_decay=0.99,
                  e_greedy=0.95,
                  e_greedy_increment=0.0001,
                  e_greedy_start=0,
                  batch_size=32,
                  replace_target_iter=10000)
    brain.store_start_frame(state)
    for i in range(memory_size + 5):
        print('\r', i)
        action = env.action_space.sample()
        obs, re, done, info = env.step(action)
        obs = RGB2gray(obs)
        env.render()
        re /= 15.0
        brain.store_transition(reward=re, action=action, obs_=obs)
        if done:
            env.reset()
    step = 1
    last_info = env.unwrapped._get_info()
    recording = []
    reward_list = []
    r = 0
    while step < 40000000:
        last_frame = brain.get_last_memory()
        # get_gif(last_frame)
        action = brain.choose_action(last_frame)
        obs_, re, done, info = env.step(action)
        recording.append(copy.deepcopy(obs_))
        if done:
            if last_info['world'] == 2:
                get_gif(recording, step)
                recording = []
            obs_ = env.reset()
            reward_list.append(r)
            r = 0
        obs_ = RGB2gray(obs_)
        env.render()
        reward = re / 15.0
        r += reward
        print(action, reward, brain.epsilon, step)
        if reward < -0.6:
            print(reward)
            print(last_info)
            print(info)

        brain.store_transition(reward=reward, action=action, obs_=copy.deepcopy(obs_))
        last_info = info
        if step % 4 == 0:
            brain.double_learn()
        if step % 10000 == 0:
            np.save('./npy/' + str(step) + '.npy', np.array(reward_list))
        step += 1
