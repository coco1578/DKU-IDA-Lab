# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2020.02.14"


"""
class Interval:

    def __init__(self, sizes, Category):
        self.sizes = sizes
        self.features = list()
        self.Category = Category

    def getIntervalFeature(self):

        if self.Category == 'KNN':

            count = 0
            prevloc = 0

            for i in range(len(self.sizes)):
                if self.sizes[i] > 0:
                    count += 1
                    self.features.append(i - prevloc)
                    prevloc = i
                if count == 300:
                    break

            for i in range(count, 300):
                self.features.append(0)

            count = 0
            prevloc = 0

            for i in range(len(self.sizes)):
                if self.sizes[i] < 0:
                    count += 1
                    self.features.append(i - prevloc)
                    prevloc = i
                if count == 300:
                    break

            for i in range(count, 300):
                self.features.append(0)

        if self.Category == "ICICS" or self.Category == "WPES11":

            MAX_INTERVAL = 300

            count = 0
            prevloc = 0
            interval_freq_in = [0] * (MAX_INTERVAL + 1)
            for i in range(len(self.sizes)):
                if self.sizes[i] > 0:
                    inv = i - prevloc - 1
                    prevloc = i

                    if inv > MAX_INTERVAL:
                        inv = MAX_INTERVAL

                    interval_freq_in[inv] += 1

            count = 0
            prevloc = 0
            interval_freq_out = [0] * (MAX_INTERVAL + 1)
            for i in range(len(self.sizes)):
                if self.sizes[i] < 0:
                    inv = i - prevloc - 1
                    prevloc = i

                    if inv > MAX_INTERVAL:
                        inv = MAX_INTERVAL
                    interval_freq_out[inv] += 1

            if self.Category == 'ICICS':
                self.features.extend(interval_freq_in)
                self.features.extend(interval_freq_out)

            if self.Category == 'WPES11':
                self.features.extend(interval_freq_in[0:3])
                self.features.extend(interval_freq_in[3:6])
                self.features.extend(interval_freq_in[6:9])
                self.features.extend(interval_freq_in[9:14])
                self.features.extend(interval_freq_in[14:])

                self.features.extend(interval_freq_out[0:3])
                self.features.extend(interval_freq_out[3:6])
                self.features.extend(interval_freq_out[6:9])
                self.features.extend(interval_freq_out[9:14])
                self.features.extend(interval_freq_out[14:])

        return self.features
