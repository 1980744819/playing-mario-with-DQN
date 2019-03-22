#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : learn.py
# @Author: zixiao@ainirobot.com
# @Date  : 2019-03-20
# @Desc  :
from settings import Actions
from DQN.model import Deep_Q_net as net
import torch.optim as optim
from DQN.replay_buffer import Replay_buffer
import torch
from itertools import count
from common import LinearSchedule
import torch.autograd as autograd
import random
import numpy as np

LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
exploration_schedule = LinearSchedule(1000000, 0.1)


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


def learning(
        env,
        replay_buffer_size=1000000,
        batch_size=32,
        frame_history_len=4,
        num_act=14,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        target_update_freq=10000
):
    img_h, img_w, img_c = env.observation_space.shape
    input_arg = frame_history_len * img_c
    actions = Actions()
    num_actions = num_act

    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration_schedule.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            return model(Variable(obs, volatile=True)).data.max(1)[1].view(1, 1)
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    q_eval = net(input_arg, num_actions).type(dtype)
    q_target = net(input_arg, num_actions).type(dtype)
    op = optim.RMSprop(q_eval.parameters(), lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
    replay_buffer = Replay_buffer(replay_buffer_size, frame_history_len)

    last_obs = env.reset()
    num_param_updates = 0
    LOG_EVERY_N_STEPS = 10000
    last_info = {
        'coins': 0,
        'time': 400,
        'lives': 2,
        'score': 0,
        'xscrollLo': 0
    }

    for t in count():
        last_idx = replay_buffer.store_frame(last_obs)
        # print(last_idx)

        if t > learning_starts:
            print('net')
            recent_obs = replay_buffer.encoder_recent_obs()
            index = select_epilson_greedy_action(q_eval, recent_obs, t)[0, 0]
            index = int(index)
        else:
            index = random.randrange(num_actions)
        action = np.array(actions.mapping[index])
        obs, reward, done, info = env.step(action)
        re = info['coins'] - last_info['coins'] + info['time'] - last_info['time'] + (
                info['lives'] - last_info['lives']) * 10 + info['score'] - last_info['score'] + info['xscrollLo'] - \
             last_info['xscrollLo']
        last_info = info
        reward = re
        print(reward)
        env.render()
        replay_buffer.store_effect(last_idx, index, reward, done)
        if done:
            obs = env.reset()
        last_obs = obs
        if t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size):
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)

            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
            act_batch = Variable(torch.from_numpy(act_batch).long())
            rew_batch = Variable(torch.from_numpy(rew_batch))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(
                dtype)  # 如果下一個state是episode中的最後一個，則done_mask = 1

            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()

            # 從抽出的batch observation中得出現在的Q值
            current_Q_values = q_eval(obs_batch).gather(1, act_batch.unsqueeze(1))
            # 用next_obs_batch計算下一個Q值，detach代表將target network從graph中分離，不去計算它的gradient
            next_max_q = q_target(next_obs_batch).detach().max(1)[0]
            next_Q_values = not_done_mask * next_max_q
            # TD value
            target_Q_values = rew_batch + (gamma * next_Q_values)
            # Compute Bellman error
            bellman_error = target_Q_values - current_Q_values
            # clip the bellman error between [-1, 1]
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            # 要 * -1 才是正確的gradient，why?
            d_error = clipped_bellman_error * -1.0

            print(d_error)

            # backward & update
            op.zero_grad()
            # current_Q_values.backward(d_error.data.unsqueeze(1))
            current_Q_values.backward(d_error.data)

            op.step()
            num_param_updates += 1

            # 每隔一段時間才更新target network
            if num_param_updates % target_update_freq == 0:
                q_target.load_state_dict(q_eval.state_dict())
