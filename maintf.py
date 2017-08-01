# -*- coding: utf8 -*-
import tensorflow as TF
import modeltf as model
import utiltf as util

import numpy as NP
import numpy.random as RNG

import h5py

import argparse
import sys

from timer import Timer

parser = argparse.ArgumentParser()
parser.add_argument('--critic_iter', default=5, type=int)
parser.add_argument('--cnng', action='store_true')
parser.add_argument('--rnng', action='store_true')
parser.add_argument('--cnnd', action='store_true')
parser.add_argument('--rnnd', action='store_true')
parser.add_argument('--framesize', type=int, default=200, help='# of amplitudes to generate at a time for RNN')
parser.add_argument('--amplitudes', type=int, default=8000, help='# of amplitudes to generate')
parser.add_argument('--noisesize', type=int, default=100, help='noise vector size')
parser.add_argument('--statesize', type=int, default=100, help='RNN state size')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--dgradclip', type=float, default=0.1)
parser.add_argument('--ggradclip', type=float, default=0.0)
parser.add_argument('modelname', type=str)
parser.add_argument('--log_train_d', type=str, default='train-d', help='log directory for D training')
parser.add_argument('--log_valid_d', type=str, default='valid-d', help='log directory for D validation')
parser.add_argument('--log_train_g', type=str, default='train-g', help='log directory for G training')

args = parser.parse_args()

print args.modelname

batch_size = args.batchsize

if args.cnng:
    g = model.Conv1DGenerator([
        (128, 5, 1),
        (128, 5, 1),
        (128, 5, 2),
        (128, 5, 1),
        (128, 5, 1),
        (128, 5, 2),
        (256, 5, 1),
        (256, 5, 1),
        (256, 5, 2),
        (256, 5, 1),
        (256, 5, 1),
        (256, 5, 2),
        (512, 5, 1),
        (512, 5, 1),
        (512, 5, 5),
#        (1024, 33, 2),
#        (2048, 33, 5),
        ])
    z = TF.placeholder(TF.float32, shape=(None, None))
    z_fixed = RNG.randn(batch_size, args.noisesize)
elif args.rnng:
    g = model.RNNGenerator(frame_size=args.framesize, noise_size=args.noisesize, state_size=args.statesize)
    z = TF.placeholder(TF.float32, shape=(None, None, None))
    z_fixed = RNG.randn(batch_size, args.amplitudes // args.framesize, args.noisesize)
    z_fixed[:, 1:-1] = 0
else:
    print 'Specify either --cnng or --rnng'
    sys.exit(1)

if args.cnnd:
    d = model.Conv1DDiscriminator([
        (128, 5, 5),
        (256, 5, 2),
        (256, 5, 2),
        (512, 5, 2),
        (512, 5, 2),
#        (1024, 33, 2),
#        (2048, 33, 2),
        ])
elif args.rnnd:
    d = model.RNNDiscriminator()
else:
    print 'Specify either --cnnd or --rnnd'
    sys.exit(1)

x_real = TF.placeholder(TF.float32, shape=(None, None))
lambda_ = TF.placeholder(TF.float32, shape=())
x_fake = g.generate(batch_size=batch_size, length=args.amplitudes)
comp, d_real, d_fake, pen = d.compare(x_real, x_fake, lambda_=lambda_)
loss_d = TF.reduce_mean(comp)
loss_g = TF.reduce_mean(-d.discriminate(x_fake))
score_fake = -loss_g
score_real = TF.reduce_mean(d.discriminate(x_real))

x = g.generate(z=z)

opt_g = TF.train.AdamOptimizer()
opt_d = TF.train.AdamOptimizer()
grad_g, vars_g = zip(*opt_g.compute_gradients(loss_g, var_list=g.model.trainable_weights))
grad_d, vars_d = zip(*opt_d.compute_gradients(loss_d, var_list=d.model.trainable_weights))
if args.ggradclip:
    grad_g = [TF.clip_by_norm(_g, args.ggradclip) for _g in grad_g]
if args.dgradclip:
    grad_d = [TF.clip_by_norm(_g, args.dgradclip) for _g in grad_d]
train_g = opt_g.apply_gradients(zip(grad_g, vars_g))
train_d = opt_d.apply_gradients(zip(grad_d, vars_d))

# Summaries
d_summaries = TF.summary.merge([
    util.summarize_var(comp, 'comp', mean=True),
    TF.summary.histogram('d_real', d_real),
    TF.summary.histogram('d_fake', d_fake),
    util.summarize_var(pen, 'pen', mean=True, std=True),
    TF.summary.histogram('x_real', x_real),
    TF.summary.histogram('x_fake', x_fake),
    ])
g_summaries = TF.summary.merge([
    TF.summary.histogram('d_fake_g', d_fake),
    TF.summary.histogram('x_fake_g', x_fake),
    ])
audio_gen = TF.summary.audio('sample', x, 8000)
d_train_writer = TF.summary.FileWriter(args.log_train_d)
d_valid_writer = TF.summary.FileWriter(args.log_valid_d)
g_writer = TF.summary.FileWriter(args.log_train_g)

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
        yield epoch, batch, NP.array(sample)[:, :args.amplitudes]
        batch += 1

dataloader = _dataloader(batch_size, data, 0, n_train_samples)
dataloader_val = _dataloader(batch_size, data, n_train_samples, nsamples)

i = 0
epoch = 1
l = 10

if __name__ == '__main__':
    s = TF.Session()
    d_train_writer.add_graph(s.graph)
    g_writer.add_graph(s.graph)
    s.run(TF.global_variables_initializer())
    x_gen = s.run(x, feed_dict={z: z_fixed})
    assert x_gen.shape[0] == batch_size
    assert x_gen.shape[1] == args.amplitudes
    while True:
        _epoch = epoch

        for j in range(args.critic_iter):
            with Timer.new('load', print_=False):
                epoch, batch_id, real_data = dataloader.next()
            with Timer.new('train_d', print_=False):
                _, loss, d_sum = s.run([train_d, loss_d, d_summaries], feed_dict={x_real: real_data, lambda_: l})
            print 'D', epoch, batch_id, loss, Timer.get('load'), Timer.get('train_d')
            d_train_writer.add_summary(d_sum, i * args.critic_iter + j + 1)
        i += 1

        _, _, real_data = dataloader_val.next()
        loss, d_sum = s.run([loss_d, d_summaries], feed_dict={x_real: real_data, lambda_: l})
        print 'D-valid', loss
        d_valid_writer.add_summary(d_sum, i * args.critic_iter)

        with Timer.new('train_g', print_=False):
            _, loss, g_sum = s.run([train_g, loss_g, g_summaries])
        print 'G', i, loss, Timer.get('train_g')
        g_writer.add_summary(g_sum, i * args.critic_iter)

        _ = s.run(x_fake)
        if NP.any(NP.isnan(_)):
            print 'NaN generated'
            sys.exit(0)
        if i % 50 == 0:
            print 'Saving...'
            x_gen, x_sum = s.run([x, audio_gen], feed_dict={z: z_fixed})
            g_writer.add_summary(x_sum, i * args.critic_iter)
            if i % 1000 == 0:
                NP.save('%s%05d.npy' % (args.modelname, i), x_gen)
                g.save('%s-gen-%05d' % (args.modelname, i))
                d.save('%s-dis-%05d' % (args.modelname, i))
