"""
Convert mnist dataset into 3d graph dataset
Adapted from: https://github.com/rkansal47/MPGAN/blob/main/mnist/mnist_dataset.py
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

import logging


class MNISTGraphDataset(Dataset):
    def __init__(self, data_dir, num_thresholded, train=True, intensities=True, num=-1, generate=False):
        if train:
            dataset = np.loadtxt(data_dir + "/mnist_train.csv", delimiter=",", dtype=np.float32)
        else:
            dataset = np.loadtxt(data_dir + "/mnist_test.csv", delimiter=",", dtype=np.float32)

        logging.info("MNIST CSV Loaded")

        if isinstance(num, list):
            map1 = list(map(lambda x: x in num, dataset[:, 0]))
            dataset = dataset[map1]
        elif num > -1:
            dataset = dataset[dataset[:, 0] == num]

        logging.debug(f"{dataset.shape = }")

        X_pre = (dataset[:, 1:] - 127.5) / 255.0

        imrange = np.linspace(-0.5, 0.5, num=28, endpoint=False)

        xs, ys = np.meshgrid(imrange, imrange)

        xs = xs.reshape(-1)
        ys = ys.reshape(-1)

        self.X = np.array(list(map(lambda x: np.array([xs, ys, x]).T, X_pre)))

        if not intensities:
            self.X = np.array(
                list(map(lambda x: x[x[:, 2].argsort()][-num_thresholded:, :2], self.X))
            )
        else:
            self.X = np.array(list(map(lambda x: x[x[:, 2].argsort()][-num_thresholded:], self.X)))

        self.X = torch.FloatTensor(self.X)


        self.generate = generate
        if self.generate:
            self.m = self.X.mean(axis=1).reshape(-1, 1, 3)
            self.std = self.X.std(axis=1).reshape(-1, 1, 3)

        logging.debug(f"{self.X.shape = }")
        # print(self.X[0])
        logging.info("Data Processed")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.generate:
            m = self.m[idx]
            std = self.std[idx]

            return {
                'test_points': self.X[idx],
                'idx': idx,
                'mean': m,
                'std': std
            }
        else:
            return {
                'train_points': self.X[idx],
                'idx': idx
            }

def load_mnist_graph(data_dir, num_particles, num, dataset_size=None, generate=False):
    # TODO: utilize dataset_size param if necessary (prob not)
    X_train = MNISTGraphDataset(
        data_dir=data_dir, num_thresholded=num_particles, train=True, num=num, generate=generate
    )

    X_test = MNISTGraphDataset(
        data_dir=data_dir, num_thresholded=num_particles, train=False, num=num, generate=generate
    )

    return X_train, X_test