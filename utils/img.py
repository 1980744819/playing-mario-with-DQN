#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : img.py
# @Author: 1980744819@qq.com
# @Date  : 2019-03-23
# @Desc  :
from PIL import Image
import numpy as np


def RGB_to_gray(obs):
    img = Image.fromarray(obs)
    img = img.convert('L')
    return np.asarray(img)
