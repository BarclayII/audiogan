# -*- coding: utf8 -*-
import tensorflow as TF
import modeltf as model

import numpy as NP
import numpy.random as RNG

import h5py

import argparse
import sys

from timer import Timer

parser = argparse.ArgumentParser()
parser.add_argument('--critic_iter', default=5, type=int)
parser.add_argument('--cnn', action='store_true')
parser.add_argument('--rnn', action='store_true')
parser.add_argument('--old', action='store_true', help='use ReLU instead of LeakyReLU + Scaling')
parser.add_argument('--framesize', type=int, default=200, help='# of amplitudes to generate at a time for RNN')
parser.add_argument('--noisesize', type=int, default=100, help='noise vector size')
parser.add_argument('--statesize', type=int, default=100, help='RNN state size')
parser.add_argument('modelname', type=str)

args = parser.parse_args()

print args.modelname

batch_size = 32

if args.cnn:
    g = model.Conv1DGenerator([
        (128, 33, 4),
        (256, 33, 4),
        (512, 33, 5),
        (512, 33, 1),
#        (1024, 33, 2),
#        (2048, 33, 5),
        ], old=args.old)
    z = TF.placeholder(TF.float32, shape=(None, None))
    z_fixed = RNG.randn(batch_size, args.noisesize)
elif args.rnn:
    g = model.RNNGenerator(frame_size=args.framesize, noise_size=args.noisesize, state_size=args.statesize)
    z = TF.placeholder(TF.float32, shape=(None, None, None))
    z_fixed = RNG.randn(batch_size, 8000 // args.framesize, args.noisesize)
    z_fixed[:, 1:-1] = 0
else:
    print 'Specify either --cnn or --rnn'
    sys.exit(1)
d = model.Conv1DDiscriminator([
    (128, 33, 5),
    (256, 33, 4),
    (512, 33, 4),
#    (1024, 33, 2),
#    (2048, 33, 2),
    ])
x_real = TF.placeholder(TF.float32, shape=(None, None))
lambda_ = TF.placeholder(TF.float32, shape=())
x_fake = g.generate(batch_size=batch_size, length=8000)
comp, d_real, d_fake, pen = d.compare(x_real, x_fake, lambda_=lambda_)
loss_d = TF.reduce_mean(comp)
loss_g = TF.reduce_mean(-d.discriminate(x_fake))
score_fake = -loss_g
score_real = TF.reduce_mean(d.discriminate(x_real))

x = g.generate(z=z)

opt_g = TF.train.AdamOptimizer()
opt_d = TF.train.AdamOptimizer()
train_g = opt_g.apply_gradients(opt_g.compute_gradients(loss_g, var_list=g.model.trainable_weights))
train_d = opt_d.apply_gradients(opt_d.compute_gradients(loss_d, var_list=d.model.trainable_weights))

dataset = h5py.File('dataset.h5')
data = dataset['data']
nsamples = data.shape[0]
n_train_samples = nsamples // 10 * 9
def _dataloader(batch_size, data, lower, upper):
    epoch = 1
    batch = 0
    idx = RNG.permutation(range(lower, upper))
    cur = 0

    while True:
        indices = []
        for i in range(batch_size):
            if cur == len(idx):
                cur = 0
                idx = RNG.permutation(range(lower, upper))
                epoch += 1
                batch = 0
            indices.append(idx[cur])
            cur += 1
        sample = data[sorted(indices)]
        yield epoch, batch, NP.array(sample)
        batch += 1

dataloader = _dataloader(batch_size, data, 0, n_train_samples)
dataloader_val = _dataloader(batch_size, data, n_train_samples, nsamples)

i = 0
epoch = 1
l = 10

if __name__ == '__main__':
    s = TF.Session()
    s.run(TF.global_variables_initializer())
    x_gen = s.run(x, feed_dict={z: z_fixed})
    assert x_gen.shape[0] == batch_size
    assert x_gen.shape[1] == 8000
    while True:
        _epoch = epoch
        i += 1
        for _ in range(args.critic_iter):
            with Timer.new('load', print_=False):
                epoch, batch_id, real_data = dataloader.next()
            with Timer.new('train_d', print_=False):
                _, loss, c_real, c_fake, dr, df, p, xf = s.run([train_d, loss_d, score_real, score_fake, d_real, d_fake, pen, x_fake], feed_dict={x_real: real_data, lambda_: l})
            print 'D', epoch, batch_id, loss, c_real, c_fake, \
                    dr.mean(), dr.std(), df.mean(), df.std(), \
                    p.mean(), p.std(), Timer.get('load'), Timer.get('train_d')
            print xf.min(), xf.max(), xf.mean(), xf.std()
        _, _, real_data = dataloader_val.next()
        loss, c_real, c_fake = s.run([loss_d, score_real, score_fake], feed_dict={x_real: real_data, lambda_: l})
        print 'D-valid', loss, c_real, c_fake
        with Timer.new('train_g', print_=False):
            _, loss, c_fake = s.run([train_g, loss_g, score_fake])
        print 'G', i, loss, c_fake, Timer.get('train_g')
        _ = s.run(x_fake)
        print _.min(), _.max(), _.mean(), _.std()
        if i % 1000 == 0:
            print 'Saving...'
            x_gen = s.run(x, feed_dict={z: z_fixed})
            NP.save('%s%05d.npy' % (args.modelname, i), x_gen)
            g.save('%s-gen-%05d' % (args.modelname, i))
            d.save('%s-dis-%05d' % (args.modelname, i))
