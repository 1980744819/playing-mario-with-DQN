#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : brain.py
# @Author: zixiao
# @Date  : 2019-04-15
# @Desc  :
import torch.optim as optim

from PPO.memory import Memory
from PPO.model import Net
from PPO.ppo import *

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'

save_q_eval_path = 'PPO/save_pkl/'


class Agent:
    def __init__(self,
                 states_shape,
                 output_size,
                 batch_size=32,
                 gamma=0.99,
                 tau=0.95,
                 init_lr=1e-4,
                 save_step=500,
                 max_update_times=1e7):
        self.model = Net(states_shape, output_size)
        self.memory = Memory(batch_size=batch_size, frame_len=states_shape[0])
        self.current_update_times = 0
        self.gamma = gamma
        self.tau = tau,
        self.init_lr = init_lr,
        self.save_step = save_step
        self.max_update_times = max_update_times
        self.batch_size = batch_size
        self.ppo_epochs = 3

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.init_lr[0])

    @staticmethod
    def choose_action(logits, noise_action=True):
        u = torch.rand(logits.size()).to(device)
        if noise_action:
            _, action = torch.max(logits.detach() - (-u.log()).log(), 1)
        else:
            logits = F.softmax(logits, dim=-1)
            action = np.argmax(logits.cpu().detach().numpy(), axis=-1)[0]

        return action

    def update_model(self, next_value):
        returns = compute_gae(next_value=next_value,
                              rewards=self.memory.rewards,
                              masks=self.memory.masks,
                              values=self.memory.values,
                              gamma=self.gamma,
                              tau=self.tau[0])
        # returns = torch.cat(returns).detach()
        returns = np.asarray(returns)
        log_probabilities = torch.cat(self.memory.log_probabilities).detach()
        values = self.memory.values
        values = np.asarray(values)
        advantages = returns - values

        clip_p = 0.2 * (1 - self.current_update_times / self.max_update_times)

        # log_probabilities = np.asarray(self.memory.log_probabilities)
        states = np.asarray(self.memory.states)
        actions = np.asarray(self.memory.actions)
        ppo_update(self.model,
                   self.optimizer,
                   ppo_epochs=self.ppo_epochs,
                   batch_size=self.batch_size,
                   states=states,
                   actions=actions,
                   log_probabilities=log_probabilities, returns=returns, advantages=advantages, clip_param=clip_p)

        if self.current_update_times % self.save_step == 0:
            torch.save(self.model.state_dict(), save_q_eval_path + str(self.current_update_times) + '.pkl')

        adjust_learning_rate(self.optimizer, initial_lr=self.init_lr[0], max_update_times=self.max_update_times,
                             current_update_times=self.current_update_times)
        self.current_update_times += 1


def adjust_learning_rate(optimizer, initial_lr, max_update_times, current_update_times):
    lr = initial_lr * (1 - 1.0 * current_update_times / max_update_times)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
