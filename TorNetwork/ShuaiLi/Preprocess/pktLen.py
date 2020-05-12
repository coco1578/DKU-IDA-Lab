# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2020.02.14"


"""
class PktLen:

    def __init__(self, times, sizes):
        self.times = times
        self.sizes = sizes
        self.features = list()

    def getPktLenFeature(self):

        for i in range(-1500, 1500):
            if i in self.sizes:
                self.features.append(1)
            else:
                self.features.append(0)
