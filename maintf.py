import tensorflow as TF
import modeltf as model
import utiltf as util
from keras import backend as K
import scipy.misc

import numpy as NP
import numpy.random as RNG

import argparse
import sys
import datetime
import os

from timer import Timer
import dataset

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
logdir = args.logdir + '/%s-%s' % \
        (modelnamesave, datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S'))
if not os.path.exists(logdir):
    os.mkdir(logdir)
elif not os.path.isdir(logdir):
    raise IOError('%s is not a directory' % logdir)
log_train_d = '%s/train-d' % logdir
log_valid_d = '%s/valid-d' % logdir
log_train_g = '%s/train-g' % logdir
for subdir in [log_train_d, log_valid_d, log_train_g]:
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    elif not os.path.isdir(subdir):
        raise IOError('%s exists and is not a directory' % subdir)

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
    d = model.Conv1DDiscriminator([map(int, c.split('-')) for c in args.cnnd_config.split()])
elif args.rnnd:
    cls = model.RNNDiscriminator if not args.rnntd else model.RNNTimeDistributedDiscriminator
    d = cls(frame_size=args.framesize, state_size=args.dstatesize, length=args.amplitudes,
            approx=not args.rnntd_precise, num_layers=args.rnnd_layers)
elif args.multid:
    cls = model.RNNDiscriminator if not args.rnntd else model.RNNTimeDistributedDiscriminator
    drnn = cls(frame_size=args.framesize, state_size=args.dstatesize, length=args.amplitudes,
            approx=not args.rnntd_precise, num_layers=args.rnnd_layers, metric=args.metric)
    dcnn = model.Conv1DDiscriminator([map(int, c.split('-')) for c in args.cnnd_config.split()],metric=args.metric)
    d_local = model.LocalDiscriminatorWrapper(drnn, length=args.local,metric=args.metric)
    drnn2 = cls(frame_size=args.framesize//2, length=args.amplitudes,
            approx=not args.rnntd_precise, num_layers=args.rnnd_layers,metric=args.metric)
    d_local2 = model.LocalDiscriminatorWrapper(drnn2, length=args.local*2,metric=args.metric)
    d = model.ManyDiscriminator(d_list =[drnn, dcnn, d_local2, d_local])
else:
    print 'Specify either --cnnd --rnnd --multid'
    sys.exit(1)


# Computation graph
x_real = TF.placeholder(TF.float32, shape=(None, args.amplitudes))
x_real2 = TF.placeholder(TF.float32, shape=(None, args.amplitudes))
lambda_ = TF.placeholder(TF.float32, shape=())

x_fake = g.generate(batch_size=batch_size, length=args.amplitudes)
comp, d_real, d_fake, pen, _, _ = d.compare(x_real, x_fake)
comp_verify, d_verify_1, d_verify_2, pen_verify, _, _ = d.compare(x_real, x_real2)

loss_d = comp + lambda_ * TF.reduce_mean(pen)
if args.metric == 'Wasserstein':
    loss_g = TF.reduce_mean(-d_fake)
elif args.metric == 'l2_loss':
    loss_g = TF.reduce_mean(TF.square(d_fake - 0))
else:
    print('not an eligible loss function. Use Wasserstein or l2_loss')

x = g.generate(z=z)         # Sample audio from fixed noise

# Summaries
d_train_writer = TF.summary.FileWriter(log_train_d)
d_valid_writer = TF.summary.FileWriter(log_valid_d)
g_writer = TF.summary.FileWriter(log_train_g)

d_summaries = [
        util.summarize_var(comp, 'comp', mean=True),
        util.summarize_var(d_real, 'd_real', mean=True),
        util.summarize_var(d_fake, 'd_fake', mean=True),
        util.summarize_var(pen, 'pen', mean=True, std=True),
        TF.summary.histogram('x_real', x_real),
        TF.summary.histogram('x_fake', x_fake),
        ]
g_summaries = [
        util.summarize_var(d_fake, 'd_fake_g', mean=True),
        TF.summary.histogram('x_fake_g', x_fake),
        ]
audio_gen = TF.summary.audio('sample', x, 8000, max_outputs=batch_size)

d_valid_summaries = d_summaries + [
        util.summarize_var(comp_verify, 'comp_verify', mean=True),
        util.summarize_var(d_verify_1, 'd_verify_1', mean=True),
        util.summarize_var(d_verify_2, 'd_verify_2', mean=True),
        util.summarize_var(d_verify_2 - d_verify_1, 'd_verify_diff', mean=True),
        util.summarize_var(pen_verify, 'pen_verify', mean=True, std=True),
        ]

# Optimizer
opt_g = TF.train.AdamOptimizer()
opt_d = TF.train.AdamOptimizer()
with TF.control_dependencies(TF.get_collection(TF.GraphKeys.UPDATE_OPS)):
    grad_g = opt_g.compute_gradients(loss_g, var_list=g.get_trainable_weights())
    grad_d = opt_d.compute_gradients(loss_d, var_list=d.get_trainable_weights())
if args.ggradclip:
    pre_clipped_grad_g = [TF.norm(g_) for g_, v in grad_g if g_ is not None]
    grad_g = [(TF.clip_by_norm(_g, args.ggradclip), _v) for _g, _v in grad_g if _g is not None]
    g_summaries.append(
        util.summarize_var(pre_clipped_grad_g, 'pre_clipped_grad_g', min_=True, max_=True, mean=True, std=True)
        )
if args.dgradclip:
    pre_clipped_grad_d = [TF.norm(g_) for g_, v in grad_d if g_ is not None]
    grad_d = [(TF.clip_by_norm(_g, args.dgradclip), _v) for _g, _v in grad_d if _g is not None]
    d_summaries.append(
        util.summarize_var(pre_clipped_grad_d, 'pre_clipped_grad_d', min_=True, max_=True, mean=True, std=True)
        )
train_g = opt_g.apply_gradients(grad_g)
train_d = opt_d.apply_gradients(grad_d)
if args.just_run == 'gen':
    train_d = 0
if args.just_run == 'dis':
    train_g = 0

d_summaries = TF.summary.merge(d_summaries)
d_valid_summaries = TF.summary.merge(d_valid_summaries)
g_summaries = TF.summary.merge(g_summaries)

dataloader, dataloader_val = dataset.dataloader(batch_size, args)

i = 0
epoch = 1
l = 10
wavedir = 'wave_' + modelnamesave
if __name__ == '__main__':
    try:
        os.mkdir(wavedir)
    except:
        pass
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
                _, cmp, d_sum = s.run([train_d, comp, d_summaries], 
                                       feed_dict={x_real: real_data, lambda_: l,
                                                  K.learning_phase():1})
            print 'D', epoch, batch_id, cmp, Timer.get('load'), Timer.get('train_d')
            d_train_writer.add_summary(d_sum, i * args.critic_iter + j + 1)
        i += 1

        _, _, real_data = dataloader_val.next()
        _, _, real_data2 = dataloader_val.next()
        cmp, cmp_ver, d_ver1, d_ver2, d_sum = s.run(
                [comp, comp_verify, d_verify_1, d_verify_2, d_valid_summaries],
                feed_dict={x_real: real_data,
                           x_real2: real_data2,
                           lambda_: l,
                           K.learning_phase():0})
        print 'D-valid', cmp, cmp_ver, d_ver1.mean(), d_ver2.mean()
        d_valid_writer.add_summary(d_sum, i * args.critic_iter)

        with Timer.new('train_g', print_=False):
            _, loss, g_sum = s.run([train_g, loss_g, g_summaries],
                                    {K.learning_phase():1})
        print 'G', i, loss, Timer.get('train_g')
        g_writer.add_summary(g_sum, i * args.critic_iter)

        _ = s.run(x_fake, {K.learning_phase():0})
        if NP.any(NP.isnan(_)):
            print 'NaN generated'
            sys.exit(0)
        if i % 50 == 0:
            print 'Saving...'
            x_gen, x_sum = s.run([x, audio_gen], feed_dict={z: z_fixed,
                                                            K.learning_phase(): 0})
            #Doesnt work with Py2
            #scipy.misc.imsave(wavedir + '/global_picture_generated_%05d.jpg' % (i),x_gen[j,:].astype('int16'))
            util.plot_waves(x_gen[:10,:], 5, 2, 
                        fig_name= wavedir + '/global_picture_generated_%05d' % (i))
            g_writer.add_summary(x_sum, i * args.critic_iter)
            if i % 1000 == 0:
                NP.save('%s%05d.npy' % (modelnamesave, i), x_gen)
                g.save('%s-gen-%05d' % (modelnamesave, i))
                d.save('%s-dis-%05d' % (modelnamesave, i))
