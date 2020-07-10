import torch
import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, config):

        super(Generator, self).__init__()
        self.config = config

        self.fc1 = nn.Sequential(
            nn.Linear(self.config["latent_dim"] + self.config["n_classes"], 128),
            nn.LeakyReLU(0.2),
        )

        self.fc2 = nn.Sequential(nn.Linear(128, 256), nn.LeakyReLU(0.2))

        self.fc3 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.2))

        self.fc4 = nn.Sequential(
            nn.Linear(512, int(np.prod((1, self.config["img_shape"])))), nn.Tanh()
        )

    def forward(self, noise, label):

        x = torch.cat((label, noise), -1)
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = out.view(out.size(0), 1, self.config["img_shape"])

        return out


class Discriminator(nn.Module):
    def __init__(self, config):

        super(Discriminator, self).__init__()
        self.config = config

        self.fc1 = nn.Sequential(
            nn.Linear(
                self.config["n_classes"] + int(np.prod((1, self.config["img_shape"]))),
                64,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, 128), nn.Dropout(0.25), nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, 256), nn.Dropout(0.25), nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc4 = nn.Linear(256, 1)

    def forward(self, x, label):

        x = torch.cat((label, x.view(x.size(0), -1)), -1)
        x = x.view(x.size(0), 1, -1)
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = out.view(out.size(0), -1)

        return out
