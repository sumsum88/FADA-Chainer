import os
import glob
import argparse
import random

dataset_path = os.getenv('DATASET_PATH')
output_path = os.getenv('OUTPUT_PATH')


import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
from dataset import *
from updater import FADAUpdater

from model import *


def v(x):
    return Variable(np.asarray(x, dtype=np.float32))

def vi(x):
    return Variable(np.asarray(x, dtype=np.int32))


class Concat(chainer.Chain):
    def __init__(self, g, h):
        super(Concat, self).__init__()
        self.g = g
        self.h = h

    def __call__(self, x):
        return self.h(self.g(x))


def main(args):
    print('GPU: {}'.format(args.gpu))

    g = VGG()
    gcl = L.Classifier(g)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()  # Make a specified GPU current
        g.to_gpu(args.gpu)  # Copy the models to the GPU
        gcl.to_gpu(args.gpu)  # Copy the models to the GPU

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer

    # g_path = os.path.join(args.output_path, 'g.npz')
    g_path = os.path.join(output_path, '0614FADA', 'g.npz')
    if os.path.exists(g_path):
        print('g.npz detected')
        try:
            serializers.load_npz(g_path, g)
        except:
            serializers.load_npz(g_path, gcl)
    else:
        exit()

    ######################################################################
    print('# Train')
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    opt_gcl = make_optimizer(gcl)
    opt_gcl.add_hook(DelGradient([
        'block1_1', 'block1_2', 'block2_1', 'block2_2',
        'block3_1', 'block3_2', 'block3_3', 'block4_1', 'block4_2', 'block4_3',
        'block5_1', 'block5_2', 'block5_3'
    ]))
    train = SVHNDataset(src='train', k=args.k_shots)
    test = SVHNDataset(src='test', size=1000)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, opt_gcl, device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.output_path)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(test_iter, gcl, device=args.gpu))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'main/accuracy','validation/main/loss', 'validation/main/accuracy', 'elapsed_time'
    ]))
    trainer.extend(
        extensions.snapshot(
        filename='snapshot_g'),
        trigger=(100, 'epoch')
    )
    trainer.extend(extensions.ProgressBar())
    # trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    e = extensions.Evaluator(test_iter, gcl, device=args.gpu)
    print('# Test')
    print(e())


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # dataset io
    p.add_argument('-o', '--output_path', metavar='PATH', type=str, default='test',
                   help='output_path (default: ./output)')
    # p.add_argument('-d', '--dataset', type=str, default='CS',
    #                help='MSRC or DSD or CS')
    # p.add_argument('-m', '--model', type=str, default='U',
    #                help='U or Refine')

    # train
    p.add_argument('-b', '--batchsize', metavar='N', type=int, default=1,
                   help='batch size (default: 128)')
    p.add_argument('-w', '--weight_decay', metavar='N', type=float, default=0.0001,
                   help='weight decay coefficient (default 0.00001)')
    p.add_argument('-e', '--epoch', metavar='N', type=int, default=100,
                   help='number of epochs (default: 100)')
    p.add_argument('-g', '--gpu', metavar='N', type=int, default=0,
                   help='gpu id (-1 if use cpu)')
    p.add_argument('-k', '--k_shots', metavar='N', type=int, default=5,
                   help='number of target')
    p.add_argument('-r', '--resume', dest='resume', action='store_true')

    args = p.parse_args()
    print('dataset_path: ', dataset_path)
    print('output_path: ', output_path)
    args.output_path = os.path.join(output_path, args.output_path)
    os.makedirs(args.output_path, exist_ok=True)

    main(args)