#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ppo_v2_main.py
# @Author: zixiao
# @Date  : 2019-04-24
# @Desc  :
from collections import deque, namedtuple
from copy import deepcopy
from itertools import count

import numpy as np
from PIL import Image

from PPO_v2.ppo import PPO
from env.gym_super_mario_bros import env


def RGB2gray(obs):
    img = Image.fromarray(obs).crop((0, 40, 256, 240)).resize((84, 84))
    img = img.convert('L')
    # img.show()
    return np.asarray(img, dtype=np.uint8)


Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
if __name__ == '__main__':
    agent = PPO()
    render = True
    num_channel = 4
    obs = deque(maxlen=num_channel)
    for i_epoch in range(100000):
        state = env.reset()
        state = RGB2gray(state)
        for i in range(num_channel):
            obs.append(deepcopy(state))
        if render:
            env.render()
        for t in count():
            action, action_prob = agent.select_action(obs)

            next_state, reward, done, info = env.step(action)
            next_state = RGB2gray(next_state)
            reward /= 15.0
            trans = Transition(deepcopy(obs), action, action_prob, reward, deepcopy(next_state))
            agent.store_transition(trans)
            if render:
                env.render()
            obs.append(deepcopy(next_state))
            if done:
                print("done")
                if len(agent.buffer) >= agent.batch_size:
                    print("buffer length is ", len(agent.buffer))
                    agent.update(i_epoch)
                break
