#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ppo.py
# @Author: zixiao
# @Date  : 2019-04-15
# @Desc  :
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def get_batch(batch_size, states, actions, log_probabilities, returns, advantages):
    len_states = len(states)
    ids = np.random.randint(0, len_states, batch_size)
    return states[ids], actions[ids], log_probabilities[ids], returns[ids], advantages[ids]


def ppo_update(model,
               optimizer,
               ppo_epochs,
               batch_size,
               states,
               actions,
               log_probabilities,
               returns,
               advantages,
               clip_param):
    for i in range(ppo_epochs):
        state, action, old_log_probabilities, return_, advantage = get_batch(batch_size,
                                                                             states,
                                                                             actions,
                                                                             log_probabilities,
                                                                             returns,
                                                                             advantages)
        state = torch.FloatTensor(state).to(device)
        action = torch.LongTensor(action)
        pi, value = model(state)
        probability = F.softmax(pi, dim=-1)
        log_probability = F.log_softmax(pi, dim=-1)
        batch_index = np.arange(batch_size)
        action_prob = probability[batch_index, action]
        # old_log_probabilities = torch.FloatTensor(old_log_probabilities).to(device)
        action_prob_old = old_log_probabilities.exp()[batch_index, action]
        ratio = action_prob / (action_prob_old + 1e-10)

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
        advantage = torch.FloatTensor(advantage)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, min=1. - clip_param, max=1. + clip_param) * advantage

        policy_loss = -torch.min(surr1, surr2).mean()
        return_ = torch.FloatTensor(return_)
        value_loss = (return_ - value).pow(2).mean()
        entropy_loss = (probability * log_probability).sum(1).mean()
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
