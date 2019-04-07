#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : img.py
# @Author: 1980744819@qq.com
# @Date  : 2019-03-23
# @Desc  :
import numpy as np
from PIL import Image, ImageSequence


def RGB_to_gray(obs):
    img = Image.fromarray(obs).crop((0, 40, 256, 240)).resize((200, 200))
    img = img.convert('L')
    return np.asarray(img)


def get_gif(ims, name):
    sequence = []
    for item in ims:
        sequence.append(Image.fromarray(item))
    sequence[0].save(str(name) + '.gif', save_all=True, append_images=sequence[1:])
