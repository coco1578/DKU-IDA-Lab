# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2020.02.14"


"""
class Ngram:

    def __init__(self, sizes, NGRAM):
        self.sizes = sizes
        self.NGRAM = NGRAM

    def NgramLocator(self, sample, Ng):

        index = 0
        for i in range(Ng):
            if sample[i] == 1:
                bit = 1
            else:
                bit = 0
            index = index + bit * (2**(Ng-i-1))

        return index

    def getNgram(self):

        counter = 0
        buckets = [0] * (2 ** self.NGRAM)

        for i in range(len(self.sizes) - self.NGRAM + 1):
            index = self.NgramLocator(self.sizes[i:i + self.NGRAM], self.NGRAM)
            buckets[index] = buckets[index] + 1
            counter += 1

        return buckets