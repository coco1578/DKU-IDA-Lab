# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2020.02.17"


"""
import numpy as np


class Time:

    def __init__(self, times, sizes):
        self.times = times
        self.sizes = sizes
        self.features = list()

    def interTimeStats(self, times):

        res = list()

        for i in range(1, len(times)):
            prev = times[i - 1]
            cur = times[i]
            res.append(cur - prev)

        if len(res) == 0:
            return [0, 0, 0, 0]
        else:
            return [np.max(res), np.mean(res), np.std(res), np.percentile(res, 75)]

    def transTimeStats(self, times):

        return [np.percentile(times, 25), np.percentile(times, 50), np.percentile(times, 75), np.percentile(times, 100)]

    def getTimeFeature(self):

        self.features.extend(self.interTimeStats(self.times))

        times_out = list()
        for i in range(0, len(self.sizes)):
            if self.sizes[i] > 0:
                times_out.append(self.times[i])
        self.features.extend(self.interTimeStats(times_out))

        times_in = list()
        for i in range(0, len(self.sizes)):
            if self.sizes[i] < 0:
                times_in.append(self.times[i])
        self.features.extend(self.interTimeStats(times_in))

        self.features.extend(self.transTimeStats(self.times))
        self.features.extend(self.transTimeStats(times_out))
        self.features.extend(self.transTimeStats(times_in))

        return self.features
