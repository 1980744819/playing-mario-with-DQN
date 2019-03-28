#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : reward.py
# @Author: zixiao
# @Date  : 2019-03-28
# @Desc  :


def get_reward(info, last_info):
    re = info['coins'] - last_info['coins'] + info['time'] - last_info['time'] + (
            info['lives'] - last_info['lives']) * 10 + info['score'] - last_info['score'] + info['xscrollLo'] - \
         last_info['xscrollLo'] - 0.1
    return re / 1000.0
