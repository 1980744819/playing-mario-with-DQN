#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : mario.py
# @Author: 1980744819@qq.com
# @Date  : 2019-03-18
# @Desc  :
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
