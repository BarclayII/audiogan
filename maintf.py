# -*- coding: utf8 -*-
import tensorflow as TF
import modeltf as model
import utiltf as util
from keras import backend as K

import numpy as NP
import numpy.random as RNG

import h5py

import argparse
import sys

from timer import Timer

parser = argparse.ArgumentParser()
parser.add_argument('--critic_iter', default=5, type=int)
parser.add_argument('--cnng', action='store_true', default=True)
parser.add_argument('--rnng', action='store_true',default=False)
parser.add_argument('--cnnd', action='store_true',default=True)
parser.add_argument('--rnnd', action='store_true',default=False)
parser.add_argument('--resnet', action='store_true',default=True)
parser.add_argument('--framesize', type=int, default=200, help='# of amplitudes to generate at a time for RNN')
parser.add_argument('--amplitudes', type=int, default=8000, help='# of amplitudes to generate')
parser.add_argument('--noisesize', type=int, default=100, help='noise vector size')
parser.add_argument('--statesize', type=int, default=100, help='RNN state size')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--dgradclip', type=float, default=0.1)
parser.add_argument('--ggradclip', type=float, default=0.0)
parser.add_argument('modelname', type=str)
parser.add_argument('--log_train_d', type=str, default='train-d_global', help='log directory for D training')
parser.add_argument('--log_valid_d', type=str, default='valid-d_global', help='log directory for D validation')
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
        ], resnet = args.resnet)
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
    d_global = model.Conv1DDiscriminator([
        (128, 5, 5),
        (256, 5, 2),
        (256, 5, 2),
        (512, 5, 2),
        (512, 5, 2),
        ])
    d_local = model.Conv1DDiscriminator([
        (64, 5, 2),
        (128, 5, 2),
        (128, 5, 2),
        ])
elif args.rnnd:
    d_global = model.RNNDiscriminator()
    d_local = model.RNNDiscriminator()
else:
    print 'Specify either --cnnd or --rnnd'
    sys.exit(1)

x_global_real = TF.placeholder(TF.float32, shape=(None, None))
lambda_ = TF.placeholder(TF.float32, shape=())
x_global_fake = g.generate(batch_size=batch_size, length=args.amplitudes)
comp_global, d_global_real, d_global_fake, pen_global = d_global.compare(x_global_real, x_global_fake, lambda_=lambda_)
loss_d_global = TF.reduce_mean(comp_global)
loss_g_global = TF.reduce_mean(-d_global.discriminate(x_global_fake))
score_fake_global = -loss_g_global
score_real_global = TF.reduce_mean(d_global.discriminate(x_global_real))

local_size = TF.Variable(TF.random_uniform([1],minval=50,maxval=300,dtype=TF.int32)[0])
x_local_real = TF.random_crop(
    x_global_real,
    size=[batch_size, local_size]
)
x_local_fake = TF.random_crop(
    x_global_fake,
    size=[batch_size, local_size]
)
comp_local, d_local_real, d_local_fake, pen_local = d_local.compare(x_local_real, x_local_fake, lambda_=lambda_)
loss_d_local = TF.reduce_mean(comp_local)
loss_g_local = TF.reduce_mean(-d_local.discriminate(x_local_fake))
score_fake_local = -loss_g_local
score_real_local = TF.reduce_mean(d_local.discriminate(x_local_real))


x = g.generate(z=z)

opt_g = TF.train.AdamOptimizer()
opt_d = TF.train.AdamOptimizer()
grad_g, vars_g = zip(*opt_g.compute_gradients(loss_g_global, var_list=g.model.trainable_weights))
grad_d, vars_d = zip(*opt_d.compute_gradients(
    loss_d_global, var_list=d_global.model.trainable_weights + d_local.model.trainable_weights))
if args.ggradclip:
    grad_g = [TF.clip_by_norm(_g, args.ggradclip) for _g in grad_g]
if args.dgradclip:
    grad_d = [TF.clip_by_norm(_g, args.dgradclip) for _g in grad_d if _g is not None]
train_g = opt_g.apply_gradients(zip(grad_g, vars_g))
train_d = opt_d.apply_gradients(zip(grad_d, vars_d))

# Summaries
d_summaries = TF.summary.merge([
    util.summarize_var(comp_global, 'comp_global', mean=True),
    TF.summary.histogram('d_global_real', d_global_real),
    TF.summary.histogram('d_global_fake', d_global_fake),
    util.summarize_var(pen_global, 'pen_global', mean=True, std=True),
    TF.summary.histogram('x_global_real', x_global_real),
    TF.summary.histogram('x_global_fake', x_global_fake),
    util.summarize_var(comp_local, 'comp_local', mean=True),
    TF.summary.histogram('d_local_real', d_local_real),
    TF.summary.histogram('d_local_fake', d_local_fake),
    util.summarize_var(pen_local, 'pen_local', mean=True, std=True),
    TF.summary.histogram('x_local_real', x_local_real),
    TF.summary.histogram('x_local_fake', x_local_fake),
    ])
g_summaries = TF.summary.merge([
    TF.summary.histogram('d_fake_g_global', d_global_fake),
    TF.summary.histogram('x_fake_g_global', x_global_fake),
    TF.summary.histogram('d_fake_g_local', d_local_fake),
    TF.summary.histogram('x_fake_g_local', x_local_fake),
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
    x_gen = s.run(x, feed_dict={z: z_fixed,
                                K.learning_phase():0})
    assert x_gen.shape[0] == batch_size
    assert x_gen.shape[1] == args.amplitudes
    while True:
        _epoch = epoch

        for j in range(args.critic_iter):
            with Timer.new('load', print_=False):
                epoch, batch_id, real_data = dataloader.next()
            with Timer.new('train_d', print_=False):
                _, loss, d_sum = s.run([train_d, loss_d_global, d_summaries], 
                                       feed_dict={x_global_real: real_data, lambda_: l,
                                                  K.learning_phase():1})
            print 'D', epoch, batch_id, loss, Timer.get('load'), Timer.get('train_d')
            d_train_writer.add_summary(d_sum, i * args.critic_iter + j + 1)
        i += 1

        _, _, real_data = dataloader_val.next()
        loss, d_sum = s.run([loss_d_global, d_summaries],
                            feed_dict={x_global_real: real_data, lambda_: l,
                                       K.learning_phase():0})
        print 'D-valid', loss
        d_valid_writer.add_summary(d_sum, i * args.critic_iter)

        with Timer.new('train_g', print_=False):
            _, loss, g_sum = s.run([train_g, loss_g_global, g_summaries],
                                    {K.learning_phase():1})
        print 'G', i, loss, Timer.get('train_g')
        g_writer.add_summary(g_sum, i * args.critic_iter)

        _ = s.run(x_global_fake, {K.learning_phase():0})
        if NP.any(NP.isnan(_)):
            print 'NaN generated'
            sys.exit(0)
        if i % 50 == 0:
            print 'Saving...'
            x_gen, x_sum = s.run([x, audio_gen], feed_dict={z: z_fixed,
                                                            K.learning_phase(): 0})
            g_writer.add_summary(x_sum, i * args.critic_iter)
            if i % 1000 == 0:
                NP.save('%s%05d.npy' % (args.modelname, i), x_gen)
                g.save('%s-gen-%05d' % (args.modelname, i))
                d_global.save('%s-dis-%05d' % (args.modelname, i))
