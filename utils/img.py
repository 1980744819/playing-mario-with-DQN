#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : img.py
# @Author: zixiao@ainirobot.com
# @Date  : 2019-03-23
# @Desc  :
from PIL import Image
import numpy as np


def RGB_to_gary(obs):
    img = Image.fromarray(obs)
    img = img.convert('L')
    return np.asarray(img)
