#!/usr/bin/env python
import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np


def v(x):
    return Variable(np.asarray(x, dtype=np.float32))

def vi(x):
    return Variable(np.asarray(x, dtype=np.int32))


class FADAUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.g, self.dcd = kwargs.pop('models')
        super(FADAUpdater, self).__init__(*args, **kwargs)

    def loss_fada_g(self, y1_out, y2_out, d_out, y1, y2, d):
        loss_1 = F.softmax_cross_entropy(y1_out, y1)
        loss_2 = F.softmax_cross_entropy(y2_out, y2)
        loss_g = loss_1 + loss_2
        chainer.report({'acc_s': F.accuracy(y1_out, y1)}, self.g)

        if d == 1:   # work only if batchsize == 1 (TODO)
            v_0 = vi([0])
            v_0.to_gpu(0)
            loss_d = F.softmax_cross_entropy(d_out, v_0)
            chainer.report({'acc_t': F.accuracy(y2_out, y2)}, self.g)
        elif d == 3:
            v_2 = vi([2])
            v_2.to_gpu(0)
            loss_d = F.softmax_cross_entropy(d_out, v_2)
            chainer.report({'acc_t': F.accuracy(y2_out, y2)}, self.g)
        else:
            chainer.report({'loss': loss_g}, self.g)
            return loss_g

        chainer.report({'loss': loss_g + loss_d}, self.g)
        return loss_g + loss_d

    def loss_fada_d(self, d_out, d):
        loss_d = F.softmax_cross_entropy(d_out, d)
        chainer.report({'loss': loss_d}, self.dcd)
        chainer.report({'acc': F.accuracy(d_out, d)}, self.dcd)
        return loss_d

    def update_core(self):
        """
        enc, dec, dis

        :return:
        """
        g_optimizer = self.get_optimizer('g')
        dcd_optimizer = self.get_optimizer('dcd')

        g, dcd = self.g, self.dcd
        xp = g.xp

        batch = self.get_iterator('main').next()
        # list([x, y])

        batchsize = len(batch)
        in_ch = 1
        out_ch = 10

        w_in = 28

        x_1 = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
        x_2 = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
        y_1 = xp.zeros(batchsize).astype("i")
        y_2 = xp.zeros(batchsize).astype("i")
        d = xp.zeros(batchsize).astype("i")

        for i in range(batchsize):
            x_1[i] = xp.asarray(batch[i][0][0])
            x_2[i] = xp.asarray(batch[i][0][1])
            y_1[i] = xp.asarray(batch[i][1][0])
            y_2[i] = xp.asarray(batch[i][1][1])
            d[i] = xp.asarray(batch[i][2])

        x_1 = Variable(x_1)
        x_2 = Variable(x_2)

        # z1 = g.extract([x1], layers=['pool5'])['pool5']  # 1 * 512 * 7 * 7
        # z2 = g.extract([x2], layers=['pool5'])['pool5']  # 1 * 512 * 7 * 7
        with chainer.using_config('train', False):
            z1 = g(x_1, feature=True)
            z2 = g(x_2, feature=True)
            y1_out, y2_out = g(x_1), g(x_2)

        d_out = dcd(F.concat([z1, z2], axis=1))  # 1 * 512 * 14 * 7 -> 1 * 4

        # update g and h
        g_optimizer.update(self.loss_fada_g, y1_out, y2_out, d_out, y_1, y_2, d)

        d_out = dcd(F.concat([z1, z2], axis=1))  # 1 * 512 * 14 * 7 -> 1 * 4

        # update DCD
        z1.unchain_backward()
        z2.unchain_backward()
        dcd_optimizer.update(self.loss_fada_d, d_out, d)


class FTUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.g = kwargs.pop('models')
        super(FTUpdater, self).__init__(*args, **kwargs)

    def loss_fada_g(self, y1_out, y2_out, d_out, y1, y2, d):
        loss_1 = F.softmax_cross_entropy(y1_out, y1)
        loss_2 = F.softmax_cross_entropy(y2_out, y2)
        loss_g = loss_1 + loss_2
        chainer.report({'acc_s': F.accuracy(y1_out, y1)}, self.g)

        if d == 1:   # work only if batchsize == 1
            v_0 = vi([0])
            v_0.to_gpu(0)
            loss_d = F.softmax_cross_entropy(d_out, v_0)
            chainer.report({'acc_t': F.accuracy(y2_out, y2)}, self.g)
        elif d == 3:
            v_2 = vi([2])
            v_2.to_gpu(0)
            loss_d = F.softmax_cross_entropy(d_out, v_2)
            chainer.report({'acc_t': F.accuracy(y2_out, y2)}, self.g)
        else:
            chainer.report({'loss': loss_g}, self.g)
            return loss_g

        chainer.report({'loss': loss_g + loss_d}, self.g)
        return loss_g + loss_d

    def loss_fada_d(self, d_out, d):
        loss_d = F.softmax_cross_entropy(d_out, d)
        chainer.report({'loss': loss_d}, self.dcd)
        chainer.report({'acc': F.accuracy(d_out, d)}, self.dcd)
        return loss_d

    def update_core(self):
        """
        enc, dec, dis

        :return:
        """
        g_optimizer = self.get_optimizer('g')
        dcd_optimizer = self.get_optimizer('dcd')

        g = self.g
        xp = g.xp

        batch = self.get_iterator('main').next()
        # list([x, y])

        batchsize = len(batch)
        in_ch = 1
        out_ch = 10

        w_in = 28

        x_1 = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
        x_2 = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
        y_1 = xp.zeros(batchsize).astype("i")
        y_2 = xp.zeros(batchsize).astype("i")
        d = xp.zeros(batchsize).astype("i")

        for i in range(batchsize):
            x_1[i] = xp.asarray(batch[i][0][0])
            x_2[i] = xp.asarray(batch[i][0][1])
            y_1[i] = xp.asarray(batch[i][1][0])
            y_2[i] = xp.asarray(batch[i][1][1])
            d[i] = xp.asarray(batch[i][2])

        x_1 = Variable(x_1)
        x_2 = Variable(x_2)

        # z1 = g.extract([x1], layers=['pool5'])['pool5']  # 1 * 512 * 7 * 7
        # z2 = g.extract([x2], layers=['pool5'])['pool5']  # 1 * 512 * 7 * 7
        with chainer.using_config('train', False):
            z1 = g(x_1, feature=True)
            z2 = g(x_2, feature=True)
            z1.unchain_backward()
            z2.unchain_backward()
            y1_out, y2_out = g.from_feature(z1), g.from_feature(z2)

        # update g and h
        g_optimizer.update(self.loss_fada_g, y1_out, y2_out, d_out, y_1, y_2, d)