
import torch as T
import torch.nn as NN
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as NP
import numpy.random as RNG
import tensorflow as TF     # for Tensorboard

import argparse
import sys
import datetime
import os

from timer import Timer
import dataset
import utiltf as util

def tovar(*arrs):
    tensors = [T.Tensor(a).cuda() for a in arrs]
    vars_ = [T.autograd.Variable(t) for t in tensors]
    return vars_[0] if len(vars_) == 1 else vars_


def tonumpy(*vars_):
    arrs = [v.data.cpu().numpy() for v in vars_]
    return arrs[0] if len(arrs) == 1 else arrs


def div_roundup(x, d):
    return (x + d - 1) // d
def roundup(x, d):
    return (x + d - 1) // d * d


def log_sigmoid(x):
    return -F.softplus(-x)
def log_one_minus_sigmoid(x):
    y_neg = T.log(1 - F.sigmoid(x))
    y_pos = -x - T.log(1 + T.exp(-x))
    x_sign = (x > 0).float()
    return x_sign * y_pos + (1 - x_sign) * y_neg


def dynamic_rnn(rnn, seq, length, initial_state):
    length_sorted, length_sorted_idx = T.sort(length)
    _, length_inverse_idx = T.sort(length_sorted_idx)
    rnn_in = pack_padded_sequence(
            seq[:, length_sorted_idx],
            to_numpy(length_sorted),
            )
    rnn_out, rnn_last_state = rnn(rnn_in, initial_state)
    rnn_out = pad_packed_sequence(rnn_out)[0]
    out = rnn_out[:, length_inverse_idx]
    if isinstance(rnn_last_state, tuple):
        state = tuple(s[:, length_inverse_idx] for s in rnn_last_state)
    else:
        state = s[:, length_inverse_idx]

    return out, state


def check_grad(params):
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.data
        anynan = (g != g).long().sum()
        anybig = (g.abs() > 1e+5).long().sum()
        assert anynan == 0
        assert anybig == 0


class Embedder(NN.Module):
    def __init__(self,
                 output_size=100,
                 char_embed_size=50,
                 num_layers=1,
                 num_chars=256,
                 ):
        self._output_size = output_size
        self._char_embed_size = char_embed_size
        self._num_layers = num_layers

        self.embed = NN.Embedding(num_chars, char_embed_size)
        self.rnn = NN.LSTM(
                char_embed_size,
                output_size,
                num_layers,
                bidirectional=True,
                )

    def forward(self, chars, length):
        num_layers = self._num_layers
        batch_size = chars.size()[0]
        output_size = self._output_size

        embed_seq = self.embed(chars).permute(1, 0, 2)
        initial_state = (
                tovar(T.zeros(num_layers * 2, batch_size, state_size)),
                tovar(T.zeros(num_layers * 2, batch_size, state_size)),
                )
        embed, (h, c) = dynamic_rnn(self.rnn, embed_seq, length, initial_state)
        h = h.permute(1, 0, 2)
        return h[:, -2:].view(batch_size, output_size * 2)


class Generator(NN.Module):
    def __init__(self,
                 frame_size=200,
                 embed_size=200,
                 noise_size=100,
                 state_size=1024,
                 num_layers=1,
                 ):
        self._frame_size = frame_size
        self._noise_size = noise_size
        self._state_size = state_size
        self._embed_size = embed_size
        self._num_layers = num_layers

        self.rnn = NN.ModuleList()
        self.rnn.append(NN.LSTMCell(frame_size + embed_size + noise_size, state_size))
        for _ in range(1, num_layers):
            self.rnn.append(NN.LSTMCell(state_size, state_size))
        self.proj = NN.Linear(state_size, frame_size)
        self.stopper = NN.Linear(state_size, 1)

    def forward(self, batch_size=None, length=None, z=None, c=None):
        frame_size = self._frame_size
        noise_size = self._noise_size
        state_size = self._state_size
        embed_size = self._embed_size
        num_layers = self._num_layers

        if z is None:
            nframes = div_roundup(length, frame_size)
            z = tovar(T.randn(batch_size, nframes, noise_size))
        else:
            batch_size, nframes, _ = z.size()

        c = c.unsqueeze(1).expand(batch_size, nframes, embed_size)
        z = T.cat([z, c], 2)

        lstm_h = [tovar(T.zeros(batch_size, state_size)) for _ in range(num_layers)]
        lstm_c = [tovar(T.zeros(batch_size, state_size)) for _ in range(num_layers)]
        x_t = tovar(T.zeros(batch_size, frame_size))
        generating = T.ones(batch_size).byte()
        length = T.zeros(batch_size).long()

        x_list = []
        s_list = []
        log_action_list = []
        for t in range(nframes):
            z_t = z[:, t]
            c_t = c[:, t]
            _x = T.cat([x_t, z_t, c_t], 1)
            lstm_h[0], lstm_c[0] = self.rnn(_x, (lstm_h, lstm_c))
            for i in range(1, num_layers):
                lstm_h[i], lstm_c[i] = self.rnn(lstm_h[i-1], (lstm_h[i], lstm_c[i]))
            x = self.proj(lstm_h[-1]).tanh_()
            logit_s = self.stopper(lstm_h)
            s = log_sigmoid(logit_s)
            s1 = log_one_minus_sigmoid(logit_s)

            logp = T.cat([s1, s], 1)
            p = logp.exp()
            stop = p.multinomial()
            log_action = logp.gather(1, stop) * generating
            length += generating.long()

            x_list.append(x)
            s_list.append(s)
            log_action_list.append(log_action)

            generating *= (stop == 0)
            if generating.data.sum() == 0:
                break

        x = T.cat(x, 1)
        s = T.stack(s, 1)
        log_action = T.stack(log_action_list, 1).sum(1)

        return x, s, log_action, tovar(length * frame_size)


class Discriminator(NN.Module):
    def __init__(self,
                 frame_size=200,
                 state_size=1024,
                 embed_size=200,
                 num_layers=1):
        self._frame_size = frame_size
        self._state_size = state_size
        self._embed_size = embed_size
        self._num_layers = num_layers

        self.rnn = NN.LSTM(
                frame_size + embed_size,
                state_size,
                num_layers,
                bidirectional=True,
                )
        self.classifier = NN.Sequential(
                NN.Linear(state_size, state_size // 2),
                NN.LeakyReLU(),
                NN.Linear(state_size // 2, 1),
                )

    def forward(self, x, length, c):
        frame_size = self._frame_size
        state_size = self._state_size
        num_layers = self._num_layers
        embed_size = self._embed_size
        batch_size, maxlen = x.size()
        max_nframes = div_roundup(maxlen, frame_size)
        nframes = length // frame_size

        x = x.view(batch_size, max_nframes, frame_size)
        c = c.unsqueeze(1).expand(batch_size, max_nframes, embed_size)
        x = T.cat([x, c], 2).permute(1, 0, 2)

        initial_state = (
                tovar(T.zeros(num_layers * 2, batch_size, state_size)),
                tovar(T.zeros(num_layers * 2, batch_size, state_size)),
                )

        lstm_out, (lstm_h, lstm_c) = dynamic_rnn(self.rnn, x, nframes, initial_state)
        lstm_out = lstm_out[nframes_inverse_idx].permute(1, 0, 2)

        classifier_in = lstm_out.view(batch_size * max_nframes, state_size)
        classifier_out = self.classifier(classifier_in).view(batch_size, max_nframes)

        return classifier_out


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
parser.add_argument('--maxlen', type=int, default=0)

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

maxlen, dataloader, dataloader_val = dataset.dataloader(batch_size, args, maxlen=args.maxlen, frame_size=args.framesize)
log_train_d, _, log_train_g = util.logdirs(args.logdir, modelnamesave)

g = Generator(
        frame_size=args.framesize,
        noise_size=args.noisesize,
        state_size=args.gstatesize,
        embed_size=args.embedsize,
        num_layers=args.rnng_layers,
        ).cuda()
nframes = div_roundup(maxlen, args.framesize)
z_fixed = tovar(RNG.randn(batch_size, nframes, args.noisesize))

e_g = Embedder(args.embedsize).cuda()
e_d = Embedder(args.embedsize).cuda()
_, cseq_fixed, clen_fixed = dataset.pick_words(batch_size, args)

d = Discriminator(
        frame_size=args.framesize,
        state_size=args.statesize,
        embed_size=args.embedsize,
        num_layers=args.rnnd_layers,
        ).cuda()

d_train_writer = TF.summary.FileWriter(log_train_d)
g_writer = TF.summary.FileWriter(log_train_g)

i = 0
epoch = 1
l = 10
alpha = 0.1
baseline = 0.

param_g = list(g.parameters()) + list(e_g.parameters())
param_d = list(d.parameters()) + list(e_d.parameters())

opt_g = T.optim.RMSprop(param_g)
opt_d = T.optim.RMSprop(param_d)
if __name__ == '__main__':
    if moddelnameload:
        if len(modelnameload) > 0:
            T.load(d, '%s-dis-%05d' % (modelnameload, args.loaditerations))
            T.load(g, '%s-gen-%05d' % (modelnameload, args.loaditerations))

    while True:
        _epoch = epoch

        for p in param_g:
            p.requires_grad = False
        for j in range(args.critic_iter):
            with Timer.new('load', print_=False):
                epoch, batch_id, real_data, real_len, _, cs, cl, _, csw, clw = dataloader.next()

            with Timer.new('train_d', print_=False):
                real_data = tovar(real_data)
                real_len = tovar(real_len).long()
                cs = tovar(cs).long()
                cl = tovar(cl).long()

                embed_d = e_d(cs, cl)
                cls = d(real_data, real_len, embed_d)
                target = tovar(T.ones(*(cls.size())))
                loss = F.binary_cross_entropy_with_logits(cls, target)

                _, cs, cl = dataset.pick_words(batch_size, args)
                cs = tovar(cs).long()
                cl = tovar(cl).long()
                embed_g = e_g(cs, cl)
                embed_d = e_d(cs, cl)
                fake_data, _, _, fake_len = g(batch_size=args.batchsize, length=maxlen, c=embed_g)
                cls = d(fake_data, fake_len, embed_d)
                target = tovar(T.zeros(*(cls.size())))
                loss += F.binary_cross_entropy_with_logits(cls, target)

                opt_d.zero_grad()
                loss.backward()
                check_grad(param_d)
                opt_d.step()

            print 'D', epoch, batch_id, to_numpy(loss), Timer.get('load'), Timer.get('train_d')

        i += 1
        for p in param_g:
            p.requires_grad = True

        with Timer.new('train_g', print_=False):
            _, cs, cl = dataset.pick_words(batch_size, args)
            cs = tovar(cs).long()
            cl = tovar(cl).long()
            embed_g = e_g(cs, cl)
            embed_d = e_d(cs, cl)
            fake_data, fake_stop, fake_logprob, fake_len = g(batch_size=args.batchsize, length=maxlen, c=embed_g)
            cls = d(fake_data, fake_len, embed_d)
            target = tovar(T.ones(*(cls.size())))
            loss = F.binary_cross_entropy_with_logits(cls, target)
            reward = -to_numpy(loss)[0]
            baseline = baseline * 0.999 + reward * 0.001
            cost_action = F.cross_entropy(fake_logprob, fake_stop)
            loss += alpha * (reward - baseline) * cost_action
            opt_g.zero_grad()
            loss.backward()
            opt_g.step()

        print 'G', i, loss, Timer.get('train_g')
