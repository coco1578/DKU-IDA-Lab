# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2020.02.17"


"""
import numpy as np


class TransPosition:

    def __init__(self, times, sizes):
        self.times = times
        self.sizes = sizes
        self.features = list()

    def getTransPosFeature(self):

        count = 0
        temp = list()
        for i in range(0, len(self.sizes)):
            if self.sizes[i] > 0:
                count += 1
                self.features.append(i)
                temp.append(i)
            if count == 300:
                break

        for i in range(count, 300):
            self.features.append(0)

        self.features.append(np.std(temp))
        self.features.append(np.mean(temp))

        count = 0
        temp = list()
        for i in range(0, len(self.sizes)):
            if self.sizes[i] < 0:
                count += 1
                self.features.append(i)
                temp.append(i)
            if count == 300:
                break
        for i in range(count, 300):
            self.features.append(0)

        self.features.append(np.std(temp))
        self.features.append(np.mean(temp))

        return self.features
