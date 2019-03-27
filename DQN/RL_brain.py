#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : RL_brain.py
# @Author: 1980744819@qq.com
# @Date  : 2019-03-20
# @Desc  :

class Brain():
    def __init__(self,
                 input_channel=12,
                 action_size=14,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None, ):
        return
