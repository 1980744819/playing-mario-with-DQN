#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : PPO_main.py
# @Author: zixiao
# @Date  : 2019-04-15
# @Desc  :
from collections import deque
from copy import deepcopy

import PIL.Image as Image
import numpy as np
import torch
import torch.nn.functional as F

from PPO.brain import Agent as Brain
from env.gym_super_mario_bros import env

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'


def RGB2gray(obs):
    img = Image.fromarray(obs).crop((0, 40, 256, 240)).resize((84, 84))
    img = img.convert('L')
    # img.show()
    return np.asarray(img, dtype=np.uint8)


if __name__ == '__main__':
    state = env.reset()
    state = RGB2gray(state)
    num_action = 7
    input_shape = (4, 84, 84)
    frame = deque(maxlen=input_shape[0])
    input_channels = input_shape[0]
    brain = Brain(states_shape=input_shape,
                  output_size=num_action,
                  batch_size=32,
                  gamma=0.99,
                  tau=0.95,
                  init_lr=1e-4,
                  save_step=500)
    num_step = 128
    for i in range(input_channels):
        frame.append(state)
    round_num = 0
    while True:
        round_num += 1

        for _ in range(num_step):
            env.render()

            last_frame = torch.FloatTensor(frame).unsqueeze(0).to(device)
            logs, value = brain.model(last_frame)
            value = value.cpu().detach().numpy()[0][0]
            action = brain.choose_action(logs).cpu().detach().numpy()[0]
            log_probability = F.log_softmax(logs, dim=-1)
            obs, reward, done, info = env.step(action)

            print(reward, action)

            reward /= 15.0

            obs = RGB2gray(obs)

            brain.memory.store_transition(state=np.array(deepcopy(frame)),
                                          action=action,
                                          reward=reward,
                                          done=done,
                                          log_probability=log_probability,
                                          value=value)
            frame.append(obs)
            if done:
                obs = env.reset()
                obs = RGB2gray(obs)
                for i in range(input_channels):
                    frame.append(obs)

        next_state = torch.FloatTensor(frame).unsqueeze(0).to(device)
        _, next_value = brain.model(next_state)
        next_value = next_value.cpu().detach().numpy()[0][0]

        brain.update_model(next_value)
        brain.memory.clear()

        # print(1)
