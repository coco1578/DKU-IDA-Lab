# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2020.02.14"


"""
def roundArbitraty(x, base):
    return int(base * round(float(x) / base))


class PktNumFeature:

    def __init__(self, times, sizes):
        self.times = times
        self.sizes = sizes
        self.features = list()

    def getPktNumFeature(self):

        total = len(self.times)
        self.features.append(total)

        count = 0

        for x in self.sizes:
            if x > 0:
                count += 1

        self.features.append(count)
        self.features.append(total - count)

        out_total = float(count) / total
        in_total = float(total - count) / total
        self.features.append(out_total * 100)
        self.features.append(in_total * 100)

        self.features.append(roundArbitraty(total, 15))
        self.features.append(roundArbitraty(count, 15))
        self.features.append(roundArbitraty(total - count, 15))

        self.features.append(roundArbitraty(out_total * 100, 5))
        self.features.append(roundArbitraty(in_total * 100, 5))

        self.features.append(total * 512)
        self.features.append(count * 512)
        self.features.append((total - count) * 512)

        return self.features
