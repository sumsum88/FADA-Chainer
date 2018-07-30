import os
import numpy as np
from chainer.dataset import dataset_mixin
from scipy import io
from chainer.datasets import get_mnist
from scipy.misc import imresize
from collections import defaultdict


dataset_path = os.path.join(os.getenv('DATASET_PATH'), 'SVHN_MNIST')


class SVHNDataset(dataset_mixin.DatasetMixin):
    img_size = (28, 28)

    def __init__(self, root=dataset_path, src='train', size=1000, k=None):
        if src == 'train':
            mat = io.loadmat(os.path.join(root, 'train_32x32.mat'))
        elif src == 'test':
            mat = io.loadmat(os.path.join(root, 'test_32x32.mat'))
        else:
            raise ValueError

        matx = mat['X'].transpose(2, 3, 0, 1).mean(axis=0)
        maty = mat['y'][:, 0].astype(np.int8)
        if k is None:
            self.x = []
            for x in matx[:size]:
                self.x.append(imresize(x, self.img_size)[np.newaxis, ...])

            self.x = np.array(self.x, dtype=np.float32)
            self.y = maty[:size]

        else:
            self.x, self.y = [], []
            counter = defaultdict(int)

            n, i = 0, 0
            while n < k * 10:
                x = imresize(matx[n], self.img_size)[np.newaxis, ...]
                y = maty[n]
                if counter[y] < k:
                    self.x.append(x)
                    self.y.append(y)
                    n += 1
                i += 1
            self.x = np.array(self.x, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def get_example(self, i):
        return self.x[i], self.y[i]


# class MNISTSVHNDataset(dataset_mixin.DatasetMixin):
#     img_size = (28, 28)
#
#     def __init__(self, svhn_path=dataset_path, n_svhn=50):
#         mat = io.loadmat(os.path.join(svhn_path, 'train_32x32.mat'))
#         self.svhn_x = mat['X'][..., :n_svhn].transpose(2, 0, 1, 3).mean(axis=0)
#         self.svhn_y = mat['y'][:n_svhn, 0]
#         train, test = get_mnist(ndim=3)
#
#         self.mnist = train
#         self.n_svhn = n_svhn
#
#     def __len__(self):
#         return 100#`len(self.mnist)
#
#     def get_example(self, index):
#         mnist_x, mnist_y = self.mnist[index]
#         if np.random.rand() >= 0.5:
#             # S * S
#             j = np.random.randint(len(self.mnist))
#             mnist_x_, mnist_y_ = self.mnist[j]
#             if mnist_y == mnist_y_:
#                 d = 0
#             else:
#                 d = 2
#             return (mnist_x, mnist_x_), (mnist_y, mnist_y_), d
#         else:
#             # S * T
#             j = np.random.randint(self.n_svhn)
#             svhn_x = self.svhn_x[..., j]
#             svhn_y = self.svhn_y[..., j]
#             svhn_x = imresize(svhn_x, self.img_size)[np.newaxis, ...]
#             if mnist_y == svhn_y:
#                 d = 1
#             else:
#                 d = 3
#
#             return (mnist_x, svhn_x), (mnist_y, svhn_y), d


class MNISTSVHNDataset(dataset_mixin.DatasetMixin):
    img_size = (28, 28)
    n_classes = 10

    def __init__(self, svhn_path=dataset_path, n_mnist=1000, n_svhn=5):
        mat = io.loadmat(os.path.join(svhn_path, 'train_32x32.mat'))
        svhn_x = mat['X'].transpose(2, 0, 1, 3).mean(axis=0)
        svhn_y = mat['y'][:, 0]
        train, test = get_mnist(ndim=3)

        self.svhn = {c: [] for c in range(self.n_classes)}
        self.mnist = {c: [] for c in range(self.n_classes)}

        n, i = 0, 0
        while n < n_mnist * 10:
            x, y = train[i]
            if len(self.mnist[y]) < n_mnist:
                self.mnist[y].append(x)
                n += 1
            i += 1

        n, i = 0, 0
        while n < n_svhn * 10:
            x = svhn_x[..., i]
            y = svhn_y[i] % 10
            if len(self.svhn[y]) < n_svhn:
                xr = imresize(x, self.img_size)[np.newaxis, ...]
                self.svhn[y].append(xr)
                n += 1
            i += 1

        self.n_mnist = n_mnist
        self.n_svhn = n_svhn

    def __len__(self):
        return self.n_mnist

    def get_example(self, index, verbose=False):
        c = np.random.randint(self.n_classes)

        mnist_x = self.mnist[c][index]

        if np.random.rand() >= 0.5:
            # S * S
            if np.random.rand() >= 0.5:
                # same class
                j = np.random.randint(self.n_mnist)
                mnist_x_ = self.mnist[c][j]
                d = 0
                if verbose:
                    print('S * S same')
                return (mnist_x, mnist_x_), (c, c), d
            else:
                # diff class
                c_ = np.random.randint(self.n_classes)
                while c_ == c:
                    c_ = np.random.randint(self.n_classes)

                j = np.random.randint(self.n_mnist)
                mnist_x_ = self.mnist[c_][j]
                d = 2
                if verbose:
                    print('S * S diff')

                return (mnist_x, mnist_x_), (c, c_), d
        else:
            # S * T
            if np.random.rand() >= 0.5:
                # same class
                j = np.random.randint(self.n_svhn)
                svhn_x = self.svhn[c][j]
                d = 1
                if verbose:
                    print('S * T same')
                return (mnist_x, svhn_x), (c, c), d
            else:
                # diff class
                c_ = np.random.randint(self.n_classes)
                while c_ == c:
                    c_ = np.random.randint(self.n_classes)

                j = np.random.randint(self.n_svhn)
                svhn_x = self.svhn[c_][j]
                d = 3
                if verbose:
                    print('S * T diff')
                return (mnist_x, svhn_x), (c, c_), d
