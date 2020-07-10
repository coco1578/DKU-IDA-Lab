import torch
import torch.utils.data

from torch.autograd import Variable
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self, dataset=None, transform=None, resize_shape=None, scale_factor=None
    ):

        super(Dataset, self).__init__()

        self.dataset = dataset
        self.X, self.y = self.setData(self.dataset, resize_shape, scale_factor)
        self.len = self.X.shape[0]

        # TODO: Transform
        self.transform = transform

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

    def setData(self, dataset, resize_shape, scale_factor):

        from sklearn.preprocessing import MinMaxScaler

        X, y = dataset[:, 1:], dataset[:, 0]
        if scale_factor is not None:
            scaler = MinMaxScaler(feature_range=(scale_factor[0], scale_factor[1]))
        else:
            scaler = MinMaxScaler()

        X = scaler.fit_transform(X)
        X = Variable(torch.from_numpy(X).float())

        if resize_shape is not None:
            if len(resize_shape) == 2:
                X = X.reshape(X.shape[0], resize_shape[0], resize_shape[1])
            else:
                X = X.reshape(
                    X.shape[0], resize_shape[0], resize_shape[1], resize_shape[2]
                )
        y = Variable(torch.from_numpy(y).long())

        return X, y
