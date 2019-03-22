#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: zixiao@ainirobot.com
# @Date  : 2019-03-21
# @Desc  :

from settings.action_space import Actions
from env.mario import env
from itertools import count


def get_reward(info, record):
    return


if __name__ == '__main__':
    actions = Actions()
    env.reset()
    for t in count():
        env.render()
