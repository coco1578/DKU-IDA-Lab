# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2020.02.14"


"""
import itertools
import numpy as np


class Cumul:

    def __init__(self, packets, featureCount):
        self.packets = packets
        self.featureCount = int(featureCount)

    def getCumulFeatures(self):

        separateClassifier = True

        features = list()

        total = list()
        cum = list()
        pos = list()
        neg = list()
        inSize = 0
        outSize = 0
        inCount = 0
        outCount = 0

        for packetsize in itertools.islice(self.packets, None):
            packetsize = -packetsize

            if packetsize > 0:
                inSize += packetsize
                inCount += 1

                if len(cum) == 0:
                    cum.append(packetsize)
                    total.append(packetsize)
                    pos.append(packetsize)
                    neg.append(0)
                else:
                    cum.append(cum[-1] + packetsize)
                    total.append(total[-1] + abs(packetsize))
                    pos.append(pos[-1] + packetsize)
                    neg.append(neg[-1] + 0)

            if packetsize < 0:
                outSize += abs(packetsize)
                outCount += 1

                if len(cum) == 0:
                    cum.append(packetsize)
                    total.append(abs(packetsize))
                    pos.append(0)
                    neg.append(abs(packetsize))
                else:
                    cum.append(cum[-1] + packetsize)
                    total.append(total[-1] + abs(packetsize))
                    pos.append(pos[-1] + 0)
                    neg.append(neg[-1] + abs(packetsize))

        features.append(inCount)
        features.append(outCount)
        features.append(outSize)
        features.append(inSize)

        if separateClassifier:
            posFeatures = np.interp(np.linspace(total[0], total[-1], int(self.featureCount / 2)), total, pos)
            negFeatures = np.interp(np.linspace(total[0], total[-1], int(self.featureCount / 2)), total, neg)
            for el in itertools.islice(posFeatures, None):
                features.append(el)
            for el in itertools.islice(negFeatures, None):
                features.append(el)
        else:
            cumFeatures = np.interp(np.linspace(total[0], total[-1], self.featureCount + 1), total, cum)
            for el in itertools.islice(cumFeatures, 1, None):
                features.append(el)

        return features