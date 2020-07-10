import os
import argparse

from tqdm import tqdm
from scipy.sparse import coo_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from torch.autograd import Variable
from torch.utils.data import DataLoader

from TorNetwork.GAN.torch_library.neural_network_architecture import *
from TorNetwork.GAN.torch_library.utils.dataSet import *


def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument(
        "--b1",
        type=float,
        default=0.9,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of second order momentum of gradient",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Logging interval of the training process",
    )

    args = parser.parse_args()

    return args


def main():

    args = argument_parser()

    cuda = True if torch.cuda.is_available() else False

    X, y = load_svmlight_file()
    X = np.array(coo_matrix(X, dtype=np.float).todense())
    num_of_classes = len(set(y))

    dataset = CustomDataset(
        dataset=np.hstack((y.reshape(-1, 1), X)), transform=None, scale_factor=(-1, 1)
    )
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    config = dict()
    config["latent_dim"] = args.latent_dim
    config["num_of_classes"] = num_of_classes
    config["img_shape"] = ()

    generator = Generator(config=config)
    discriminator = Discriminator(config=config)

    adv_loss = nn.MSELoss()
    loss = nn.BCELoss()

    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
    )

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adv_loss.cuda()
        loss.cuda()

    for epoch in tqdm(range(args.n_epochs)):
        for i, (traffic, label) in enumerate(dataLoader):

            traffic = traffic.cuda()
            label = label.cuda()
            batch_size = traffic.shape[0]

            # Adversarial ground truths
            valid = Variable(
                torch.cuda.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False
            )
            fake = Variable(
                torch.cuda.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False
            )

            # Configure input
            real_data = Variable(traffic.type(torch.cuda.FloatTensor))
            real_data = real_data.view(batch_size, 1, config["img_shape"])
            label = Variable(label.type(torch.cuda.LongTensor))

            # one hot encoding
            label_one_hot = Variable(
                torch.cuda.FloatTensor(batch_size, config["num_of_classes"])
            )
            label_one_hot.zero_()
            label_one_hot.scatter_(1, label.view(label.size(0), 1), 1)

            optimizer_g.zero_grad()

            # Sample noise and labels as generator input
            noise = Variable(
                torch.cuda.FloatTensor(
                    np.random.normal(0, 1, (batch_size, config["latent_dim"]))
                )
            )
            fake_label = Variable(
                torch.cuda.LongTensor(
                    np.random.randint(0, config["num_of_classes"], batch_size)
                )
            )

            # one hot encoding
            fake_label_one_hot = Variable(
                torch.cuda.FloatTensor(batch_size, config["num_of_classes"])
            )
            fake_label_one_hot.zero_()
            fake_label_one_hot.scatter_(1, fake_label.view(fake_label.size(0), -1), 1)

            # Generator a batch of traffic data
            fake_traffic = generator(noise, fake_label_one_hot)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(fake_traffic, fake_label_one_hot)
            g_loss = loss(validity, valid)
            g_loss.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()

            # Loss for real traffic data
            validity_real = discriminator(real_data, label_one_hot)
            d_real_loss = loss(validity_real, valid)

            # Loss for fake traffic data
            validity_fake = discriminator(fake_traffic.detach(), fake_label_one_hot)
            d_fake_loss = loss(validity_fake, fake)

            # Total discriminative loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

        if i % args.log_interval == 0:
            print(
                "[INFO] [EPOCH %d/%d - BATCH %d/%d - D loss: %.4f - G loss: %.4f]"
                % (
                    epoch,
                    args.n_epochs,
                    i,
                    len(dataLoader),
                    d_loss.item(),
                    g_loss.item(),
                )
            )

        # model checkpoint
        torch.save(generator.state_dict())
        torch.save(discriminator.state_dict())
