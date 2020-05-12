# -*- coding: utf-8 -*-
"""


__author__ = "coco"
__email__ = "rlawnsgh0826@gmail.com"
__date__ = "2019. 12. 26."


"""
import numpy as np

from .dataSet import Dataset
from scipy.sparse import coo_matrix
from sklearn.datasets import load_svmlight_file


class DataReader:

    def __init__(self, file_name=None):
        super(DataReader, self).__init__()
        assert (file_name is not None), 'File name is None.'

        self.file_name = file_name
        self.X = None
        self.y = None

        if Dataset.check_file_format(self.file_name) == Dataset.SPARSE:
            self.__load_svmlight_file()
        elif Dataset.check_file_format(self.file_name) == Dataset.NORMAL:
            self.__load_normal_file()
        else:
            raise Exception('Invalid file format.')

        self.dataset = Dataset(X=self.X, y=self.y)

    def __load_svmlight_file(self):
        X, self.y = load_svmlight_file(self.file_name)
        self.X = np.array(coo_matrix(X, dtype=np.float).todense())

    def __load_normal_file(self):
        X = np.loadtxt(self.file_name, delimiter=',')

        row_size, col_size = X.shape
        self.X = X[:, 1:col_size]
        self.y = X[:, 0]

    def get_dataset(self):
        return self.dataset