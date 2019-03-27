#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : action_space.py
# @Author: 1980744819@qq.com
# @Date  : 2019-03-20
# @Desc  :
import numpy as np
import random


class Actions():

    def __init__(self):
        self.mapping = {
            #   0  1  2  3  4  5  6  7  8
            #   b  n        u  d  l  r  a
            0: [0, 1, 0, 0, 0, 0, 0, 0, 0],  # noop
            1: [0, 0, 0, 0, 1, 0, 0, 0, 0],  # up
            2: [0, 0, 0, 0, 0, 1, 0, 0, 0],  # down
            3: [0, 0, 0, 0, 0, 0, 1, 0, 0],  # left
            4: [0, 0, 0, 0, 0, 0, 0, 1, 0],  # right
            5: [0, 0, 0, 0, 0, 0, 0, 0, 1],  # a
            6: [0, 0, 0, 0, 0, 0, 1, 0, 1],  # left + a
            7: [1, 0, 0, 0, 0, 0, 1, 0, 0],  # left + b
            8: [1, 0, 0, 0, 0, 0, 1, 0, 1],  # left + a + b
            9: [1, 0, 0, 0, 0, 0, 0, 0, 0],  # b
            10: [0, 0, 0, 0, 0, 0, 0, 1, 1],  # right + a
            11: [1, 0, 0, 0, 0, 0, 0, 1, 0],  # right + b
            12: [1, 0, 0, 0, 0, 0, 0, 1, 1],  # right + a +b
            13: [1, 0, 0, 0, 0, 0, 0, 0, 1],  # a + b

        }
        self.n = len(self.mapping)

    def sample(self):
        return np.array(self.mapping[random.randint(0, 13)])
