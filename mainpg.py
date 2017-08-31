
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
from computation_graph import DynamicGAN

parser = argparse.ArgumentParser()
parser.add_argument('--critic_iter', default=5, type=int)
parser.add_argument('--rnng_layers', type=int, default=1)
parser.add_argument('--rnnd_layers', type=int, default=1)
parser.add_argument('--framesize', type=int, default=200, help='# of amplitudes to generate at a time for RNN')
parser.add_argument('--noisesize', type=int, default=100, help='noise vector size')
parser.add_argument('--gstatesize', type=int, default=100, help='RNN state size')
parser.add_argument('--dstatesize', type=int, default=100, help='RNN state size')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--dgradclip', type=float, default=0.0)
parser.add_argument('--ggradclip', type=float, default=0.0)
parser.add_argument('--modelname', type=str, default = '')
parser.add_argument('--modelnamesave', type=str, default='')
parser.add_argument('--modelnameload', type=str, default='')
parser.add_argument('--just_run', type=str, default='')
parser.add_argument('--loaditerations', type=int)
parser.add_argument('--logdir', type=str, default='.', help='log directory')
parser.add_argument('--metric', type=str, default='l2_loss')
parser.add_argument('--dataset', type=str, default='dataset.h5')
parser.add_argument('--embedsize', type=int, default=100)
parser.add_argument('--constraint', type=str, default='gp', help='none, gp, wc or noise')
parser.add_argument('--minwordlen', type=int, default=1)

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

args.conditional = True

batch_size = args.batchsize

maxlen, dataloader, dataloader_val = dataset.dataloader(batch_size, args, frame_size=args.framesize)
log_train_d, log_valid_d, log_train_g = util.logdirs(args.logdir, modelnamesave)

g = model.RNNDynamicGenerator(
        frame_size=args.framesize, noise_size=args.noisesize,
        state_size=args.gstatesize, num_layers=args.rnng_layers)
nframes = util.div_roundup(maxlen, args.framesize)
z = TF.placeholder(TF.float32, shape=(None, nframes, args.noisesize))
z_fixed = RNG.randn(batch_size, nframes, args.noisesize)

e_g = model.CharRNNEmbedder(args.embedsize)
e_d = model.CharRNNEmbedder(args.embedsize)
cseq = TF.placeholder(TF.int32, shape=(None, None))
clen = TF.placeholder(TF.int32, shape=(None,))
_, cseq_fixed, clen_fixed = dataset.pick_words(batch_size, args)

d = model.RNNTimeDistributedDynamicDiscriminator(
        frame_size=args.framesize,
        state_size=args.dstatesize,
        length=maxlen,
        num_layers=args.rnnd_layers,
        constraint=args.constraint,
        metric=args.metric,
        )

gan = DynamicGAN(args, maxlen, d, g, z, e_g=e_g, e_d=e_d, cseq=cseq, clen=clen)

d_train_writer = TF.summary.FileWriter(log_train_d)
d_valid_writer = TF.summary.FileWriter(log_valid_d)
g_writer = TF.summary.FileWriter(log_train_g)

i = 0
epoch = 1
l = 10
alpha = 0.1
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
                epoch, batch_id, real_data, real_len, _, cs, cl, _, csw, clw = dataloader.next()

            feed_dict = {
                    gan.x_real: real_data,
                    gan.length_real: real_len,
                    gan.lambda_: l,
                    gan.alpha: alpha,
                    K.learning_phase(): 1,
                    gan.char_seq: cs,
                    gan.char_seq_len: cl,
                    gan.char_seq_wrong: csw,
                    gan.char_seq_wrong_len: clw,
                    }
            with Timer.new('train_d', print_=False):
                _, cmp, d_sum = s.run([gan.train_d, gan.comp, gan.d_summaries], 
                                       feed_dict=feed_dict)
            print 'D', epoch, batch_id, cmp, Timer.get('load'), Timer.get('train_d')
            d_train_writer.add_summary(d_sum, i * args.critic_iter + j + 1)
        i += 1

        _, cs, cl = dataset.pick_words(batch_size, args)
        feed_dict = {
                K.learning_phase(): 1,
                gan.char_seq: cs,
                gan.char_seq_len: cl,
                }
        with Timer.new('train_g', print_=False):
            _, xf, loss, g_sum = s.run([gan.train_g, gan.x_fake, gan.loss_g, gan.g_summaries], feed_dict=feed_dict)
        print 'G', i, loss, Timer.get('train_g')
        g_writer.add_summary(g_sum, i * args.critic_iter)
        if NP.any(NP.isnan(xf)):
            print 'NaN generated'
            sys.exit(0)

        if i % 50 == 0:
            print 'Saving...'
            _, cs, cl = dataset.pick_words(batch_size, args)
            feed_dict = {
                    gan.z: z_fixed,
                    K.learning_phase(): 1,
                    gan.cseq: cs,
                    gan.clen: cl,
                    }
            x_gen, x_sum = s.run([gan.x, gan.audio_gen],
                                 feed_dict=feed_dict)
            util.plot_waves(s, gan, x_gen, g_writer, i * args.critic_iter)
            g_writer.add_summary(x_sum, i * args.critic_iter)
            if i % 1000 == 0:
                NP.save('%s%05d.npy' % (modelnamesave, i), x_gen)
                g.save('%s-gen-%05d' % (modelnamesave, i))
                d.save('%s-dis-%05d' % (modelnamesave, i))
