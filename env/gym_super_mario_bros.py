#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : gym_super_mario_bros.py
# @Author: zixiao
# @Date  : 2019-03-28
# @Desc  :
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

