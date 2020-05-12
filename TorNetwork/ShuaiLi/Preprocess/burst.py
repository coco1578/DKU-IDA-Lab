# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2020.02.14"


"""
class Burst:

	def __init__(self, sizes):
		self.sizes = sizes
		self.features = list()

	def getBurstFeature(self):

		bursts = list()
		curburst = 0
		stopped = 0

		for size in self.sizes:
			if size > 0:
				stopped = 0
				curburst += size
			if size < 0 and stopped == 0:
				stopped = 1
			elif size < 0 and stopped == 1:
				stopped = 0
				if curburst != 0:
					bursts.append(curburst)
				curburst = 0
			else:
				pass

		if len(bursts) != 0:
			self.features.append(max(bursts))
			self.features.append(sum(bursts) / len(bursts))
			self.features.append(len(bursts))
		else:
			self.features.append(0)
			self.features.append(0)
			self.features.append(0)

		counts = [0, 0, 0]
		for burst in bursts:
			if burst > 5:
				counts[0] += 1
			if burst > 10:
				counts[1] += 1
			if burst > 15:
				counts[2] += 1

		self.features.extend(counts)

		for i in range(5):
			try:
				self.features.append(bursts[i])
			except:
				self.features.append(0)

		return self.features
