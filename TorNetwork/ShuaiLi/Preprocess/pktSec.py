# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2020.02.14"


"""
import numpy as np

class PktSec:

    def __init__(self, times, sizes, howlong):
        self.times = times
        self.sizes = sizes
        self.features = list()
        self.howlong = int(howlong)

    def getPktSecFeature(self):
        count = [0] * self.howlong
        for i in range(len(self.sizes)):
            t = int(np.floor(self.times[i]))
            if t < self.howlong:
                count[t] = count[t] + 1
        self.features.extend(count)

        self.features.append(np.mean(count))
        self.features.append(np.std(count))
        self.features.append(np.min(count))
        self.features.append(np.max(count))
        self.features.append(np.median(count))

        bucket_num = 20
        bucket = [0] * bucket_num
        for i in range(0, self.howlong):
            ib = int(i / (self.howlong / bucket_num))
            bucket[ib] = bucket[ib] + count[i]

        self.features.extend(bucket)
        self.features.append(np.sum(bucket))

        return self.features