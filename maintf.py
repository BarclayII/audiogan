import tensorflow as TF
import modeltf as model
import utiltf as util
from keras import backend as K

import numpy as NP
import numpy.random as RNG

import argparse
import sys
import datetime
import os

from timer import Timer
import dataset
from computation_graph import UnconditionalGAN

default_cnng_config = [
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
        ]
default_cnnd_config = [
        (128, 5, 5),
        (256, 5, 2),
        (256, 5, 2),
        (512, 5, 2),
        (512, 5, 2),
        ]

default_cnng_config = ' '.join(['-'.join(map(str, c)) for c in default_cnng_config])
default_cnnd_config = ' '.join(['-'.join(map(str, c)) for c in default_cnnd_config])

parser = argparse.ArgumentParser()
parser.add_argument('--critic_iter', default=5, type=int)
parser.add_argument('--cnng', action='store_true')
parser.add_argument('--rnng', action='store_true')
parser.add_argument('--rnncg', action='store_true')
parser.add_argument('--cnnd', action='store_true')
parser.add_argument('--multid', action='store_true')
parser.add_argument('--rnnd', action='store_true')
parser.add_argument('--rnng_layers', type=int, default=1)
parser.add_argument('--rnnd_layers', type=int, default=1)
parser.add_argument('--resnet', action='store_true')
parser.add_argument('--rnntd', action='store_true')
parser.add_argument('--rnntd_precise', action='store_true')
parser.add_argument('--cnng_config', type=str, default=default_cnng_config)
parser.add_argument('--cnnd_config', type=str, default=default_cnnd_config)
parser.add_argument('--framesize', type=int, default=200, help='# of amplitudes to generate at a time for RNN')
parser.add_argument('--amplitudes', type=int, default=8000, help='# of amplitudes to generate')
parser.add_argument('--noisesize', type=int, default=100, help='noise vector size')
parser.add_argument('--gstatesize', type=int, default=100, help='RNN state size')
parser.add_argument('--dstatesize', type=int, default=100, help='RNN state size')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--dgradclip', type=float, default=0.0)
parser.add_argument('--ggradclip', type=float, default=0.0)
parser.add_argument('--local', type=int, default=1000)
parser.add_argument('--modelname', type=str, default = '')
parser.add_argument('--modelnamesave', type=str, default='')
parser.add_argument('--modelnameload', type=str, default='')
parser.add_argument('--just_run', type=str, default='')
parser.add_argument('--loaditerations', type=int)
parser.add_argument('--logdir', type=str, default='.', help='log directory')
parser.add_argument('--subset', type=int, default=0)
parser.add_argument('--metric', type=str, default='l2_loss')
parser.add_argument('--dataset', type=str, default='dataset.h5')
parser.add_argument('--conditional', action='store_true')
parser.add_argument('--embedsize', type=int, default=100)
parser.add_argument('--constraint', type=str, default='gp', help='none, gp, wc or noise')

args = parser.parse_args()

if args.just_run not in ['', 'gen', 'dis']:
    print('just run should be empty string, gen, or dis. Other values not accepted')
    sys.exit(0)

if len(args.modelname) > 0:
    modelnamesave = args.modelname
    modelnameload = None
else:
    modelnamesave = args.modelnamesave
    modelnameload = args.modelnameload

print modelnamesave
print args

batch_size = args.batchsize

# Log directories
log_train_d, log_valid_d, log_train_g = util.logdirs(args.logdir, modelnamesave)

# Build generator and discriminator
if args.cnng:
    cls = model.Conv1DGenerator if not args.resnet else model.ResNetGenerator
    g = cls([map(int, c.split('-')) for c in args.cnng_config.split()])
    noise_size = args.amplitudes // g.multiplier
    z = TF.placeholder(TF.float32, shape=(None, noise_size))
    z_fixed = RNG.randn(batch_size, noise_size)
elif args.rnng:
    g = model.RNNGenerator(
            frame_size=args.framesize, noise_size=args.noisesize,
            state_size=args.gstatesize, num_layers=args.rnng_layers)
    nframes = args.amplitudes // args.framesize
    z = TF.placeholder(TF.float32, shape=(None, nframes, args.noisesize))
    z_fixed = RNG.randn(batch_size, nframes, args.noisesize)
elif args.rnncg:
    g = model.RNNConvGenerator(
            frame_size=args.framesize, noise_size=args.noisesize,
            state_size=args.gstatesize, num_layers=args.rnng_layers)
    nframes = args.amplitudes // args.framesize
    z = TF.placeholder(TF.float32, shape=(None, nframes, args.noisesize))
    z_fixed = RNG.randn(batch_size, nframes, args.noisesize)
else:
    print 'Specify either --cnng or --rnng'
    sys.exit(1)

if args.cnnd:
    d = model.Conv1DDiscriminator(
            [map(int, c.split('-')) for c in args.cnnd_config.split()],
            constraint=args.constraint,
            )
elif args.rnnd:
    cls = model.RNNDiscriminator if not args.rnntd else model.RNNTimeDistributedDiscriminator
    d = cls(frame_size=args.framesize,
            state_size=args.dstatesize,
            length=args.amplitudes,
            approx=not args.rnntd_precise,
            num_layers=args.rnnd_layers,
            constraint=args.constraint,
            metric=args.metric,
            )
elif args.multid:
    cls = model.RNNDiscriminator if not args.rnntd else model.RNNTimeDistributedDiscriminator
    drnn = cls(frame_size=args.framesize,
               state_size=args.dstatesize,
               length=args.amplitudes,
               approx=not args.rnntd_precise,
               num_layers=args.rnnd_layers,
               metric=args.metric,
               constraint=args.constraint,
               )
    dcnn = model.Conv1DDiscriminator(
            [map(int, c.split('-')) for c in args.cnnd_config.split()],
            constraint=args.constraint,
            metric=args.metric
            )
    d_local = model.LocalDiscriminatorWrapper(drnn, length=args.local,metric=args.metric)
    drnn2 = cls(
            frame_size=args.framesize//2,
            length=args.amplitudes,
            approx=not args.rnntd_precise,
            num_layers=args.rnnd_layers,
            metric=args.metric,
            constraint=args.constraint,
            )
    d_local2 = model.LocalDiscriminatorWrapper(drnn2, length=args.local*2,metric=args.metric)
    d = model.ManyDiscriminator(d_list =[drnn, dcnn, d_local2, d_local])
else:
    print 'Specify either --cnnd --rnnd --multid'
    sys.exit(1)

if args.conditional:
    embed = model.CharRNNEmbedder(embed_size=args.embedsize)
else:
    gan = UnconditionalGAN(args, d, g, z)

d_train_writer = TF.summary.FileWriter(log_train_d)
d_valid_writer = TF.summary.FileWriter(log_valid_d)
g_writer = TF.summary.FileWriter(log_train_g)

dataloader, dataloader_val = dataset.dataloader(batch_size, args)

i = 0
epoch = 1
l = 10
if __name__ == '__main__':
    s = model.start()
    d_train_writer.add_graph(s.graph)
    g_writer.add_graph(s.graph)
    s.run(TF.global_variables_initializer())

    if modelnameload:
        if len(modelnameload) > 0:
            d.load('%s-dis-%05d' % (modelnameload, args.loaditerations))
            g.load('%s-gen-%05d' % (modelnameload, args.loaditerations))
    
    while True:
        _epoch = epoch

        for j in range(args.critic_iter):
            with Timer.new('load', print_=False):
                epoch, batch_id, real_data = dataloader.next()
            with Timer.new('train_d', print_=False):
                _, cmp, d_sum = s.run([gan.train_d, gan.comp, gan.d_summaries], 
                                       feed_dict={gan.x_real: real_data, gan.lambda_: l,
                                                  K.learning_phase():1})
            print 'D', epoch, batch_id, cmp, Timer.get('load'), Timer.get('train_d')
            d_train_writer.add_summary(d_sum, i * args.critic_iter + j + 1)
        i += 1

        _, _, real_data = dataloader_val.next()
        _, _, real_data2 = dataloader_val.next()
        cmp, cmp_ver, d_ver1, d_ver2, d_sum = s.run(
                [gan.comp, gan.comp_verify, gan.d_verify_1, gan.d_verify_2, gan.d_valid_summaries],
                feed_dict={gan.x_real: real_data,
                           gan.x_real2: real_data2,
                           gan.lambda_: l,
                           K.learning_phase():0})
        print 'D-valid', cmp, cmp_ver, d_ver1.mean(), d_ver2.mean()
        d_valid_writer.add_summary(d_sum, i * args.critic_iter)

        with Timer.new('train_g', print_=False):
            _, loss, g_sum = s.run([gan.train_g, gan.loss_g, gan.g_summaries],
                                    {K.learning_phase():1})
        print 'G', i, loss, Timer.get('train_g')
        g_writer.add_summary(g_sum, i * args.critic_iter)

        _ = s.run(gan.x_fake, {K.learning_phase():0})
        if NP.any(NP.isnan(_)):
            print 'NaN generated'
            sys.exit(0)
        if i % 50 == 0:
            print 'Saving...'
            x_gen, x_sum = s.run([gan.x, gan.audio_gen],
                                 feed_dict={gan.z: z_fixed, K.learning_phase(): 0})
            util.plot_waves(s, gan, x_gen, g_writer, i * args.critic_iter)
            g_writer.add_summary(x_sum, i * args.critic_iter)
            if i % 1000 == 0:
                NP.save('%s%05d.npy' % (modelnamesave, i), x_gen)
                g.save('%s-gen-%05d' % (modelnamesave, i))
                d.save('%s-dis-%05d' % (modelnamesave, i))
