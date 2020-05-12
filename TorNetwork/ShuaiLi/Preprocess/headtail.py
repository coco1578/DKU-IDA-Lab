# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2020.02.14"


"""
class HeadTail:

    def __init__(self, times, sizes):
        self.times = times
        self.sizes = sizes
        self.features = list()

    def getFirst20(self):

        for i in range(20):
            try:
                self.features.append(self.sizes[i] + 1500)
            except:
                self.features.append(0)

        return self.features

    def getFirst30PktNum(self):

        out_count = 0
        in_count = 0

        for i in range(30):
            if i < len(self.sizes):
                if self.sizes[i] > 0:
                    out_count += 1
                else:
                    in_count += 1

        self.features.append(out_count)
        self.features.append(in_count)

        return self.features

    def getLast30PktNum(self):

        out_count = 0
        in_count = 0

        for i in range(1, 31):

            if i <= len(self.sizes):
                if self.sizes[-i] > 0:
                    out_count += 1
                else:
                    in_count += 1

        self.features.append(out_count)
        self.features.append(in_count)

        return self.features