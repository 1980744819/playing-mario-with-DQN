#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : schedule.py
# @Author: zixiao@ainirobot.com
# @Date  : 2019-03-20
# @Desc  :



class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
