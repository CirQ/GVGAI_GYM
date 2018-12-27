#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: dell
# Created Time: 2018-12-27 17:12:01

from itertools import product

from PIL import Image
import numpy as np


def convolution(gamemap):  # perform a convolution on the game map
    # kernal = np.ones(shape=(self.factor, self.factor, 3), dtype=np.uint16)
    conv = np.zeros(shape=(90,100), dtype=np.uint8)
    for i, j in product(range(9), range(10)):
        row = slice(i*10, (i+1)*10)
        col = slice(j*10, (j+1)*10)
        area = gamemap[row, col]
        conv[row, col] = np.sum(area)  # equivalent to all kernal elements are 1s
    return conv


img = Image.open('gameStateByBytes.png')
img.save('stateObs.png')

arr = np.array(img.getdata()).reshape(img.size[1], img.size[0], -1)

red = arr[:,:,0].astype(np.uint8)
Image.fromarray(red, 'L').save('red.png')

c = convolution(red)
Image.fromarray(c, 'L').save('conv.png')



pass
