# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2020.02.14"


"""
import numpy as np


class PktDistributiin:

    def __init__(self, times, sizes):
        self.times = times
        self.sizes = sizes
        self.features = list()

    def getPktDistFeature(self):

        count = 0
        temp = list()

        for i in range(0, min(len(self.sizes), 6000)):
            if self.sizes[i] > 0:
                count += 1
            if i % 30 == 29:
                self.features.append(count)
                temp.append(count)
                count = 0

        for i in range(int(len(self.sizes) / 30), 200):
            self.features.append(0)
            temp.append(0)

        self.features.append(np.std(temp))
        self.features.append(np.mean(temp))
        self.features.append(np.median(temp))
        self.features.append(np.max(temp))

        num_bucket = 20
        bucket = [0] * num_bucket

        for i in range(0, 200):
            ib = int(i / (200 / num_bucket))
            bucket[ib] = bucket[ib] + temp[i]

        self.features.extend(bucket)
        self.features.append(np.sum(bucket))

        return self.features
