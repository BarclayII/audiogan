
from torch.nn import Parameter
from functools import wraps

import torch as T
import torch.nn as NN
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import weight_norm as torch_weight_norm

import numpy as NP
import numpy.random as RNG
import tensorflow as TF     # for Tensorboard

import argparse
import sys
import datetime
import os

from timer import Timer
import dataset

import matplotlib
from librosa import feature
from mailbox import _create_carefully
matplotlib.use('Agg')
import matplotlib.pyplot as PL

from PIL import Image
import librosa


## Weight norm is now added to pytorch as a pre-hook, so use that instead :)
class WeightNorm(NN.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = T.norm(w)
            v = w/g.expand_as(w)
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/T.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

def weight_norm(m, names):
    for name in names:
        m = torch_weight_norm(m, name)
    return m

class LayerNorm(NN.Module):
    def __init__(self, features, eps=1e-6):
        NN.Module.__init__(self)
        self.gamma = NN.Parameter(T.ones(features))
        self.beta = NN.Parameter(T.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

def tovar(*arrs):
    tensors = [(T.Tensor(a.astype('float32')) if isinstance(a, NP.ndarray) else a).cuda() for a in arrs]
    vars_ = [T.autograd.Variable(t) for t in tensors]
    return vars_[0] if len(vars_) == 1 else vars_


def tonumpy(*vars_):
    arrs = [(v.data.cpu().numpy() if isinstance(v, T.autograd.Variable) else
             v.cpu().numpy() if T.is_tensor(v) else v) for v in vars_]
    return arrs[0] if len(arrs) == 1 else arrs


def div_roundup(x, d):
    return (x + d - 1) / d
def roundup(x, d):
    return (x + d - 1) / d * d


def log_sigmoid(x):
    return -F.softplus(-x)
def log_one_minus_sigmoid(x):
    y_neg = T.log(1 - F.sigmoid(x))
    y_pos = -x - T.log(1 + T.exp(-x))
    x_sign = (x > 0).float()
    return x_sign * y_pos + (1 - x_sign) * y_neg


def binary_cross_entropy_with_logits_per_sample(input, target, weight=None):
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    return loss.sum(1)


def advanced_index(t, dim, index):
    return t.transpose(dim, 0)[index].transpose(dim, 0)


def length_mask(size, length):
    length = tonumpy(length)
    batch_size = size[0]
    weight = T.zeros(*size)
    for i in range(batch_size):
        weight[i, :length[i]] = 1.
    weight = tovar(weight)
    return weight


def dynamic_rnn(rnn, seq, length, initial_state):
    length_sorted, length_sorted_idx = T.sort(length, descending=True)
    _, length_inverse_idx = T.sort(length_sorted_idx)
    rnn_in = pack_padded_sequence(
            advanced_index(seq, 1, length_sorted_idx),
            tonumpy(length_sorted),
            )
    rnn_out, rnn_last_state = rnn(rnn_in, initial_state)
    rnn_out = pad_packed_sequence(rnn_out)[0]
    out = advanced_index(rnn_out, 1, length_inverse_idx)
    if isinstance(rnn_last_state, tuple):
        state = tuple(advanced_index(s, 1, length_inverse_idx) for s in rnn_last_state)
    else:
        state = advanced_index(s, 1, length_inverse_idx)

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


def clip_grad(params, clip_norm):
    if clip_norm == 0:
        return
    norm = 0
    for p in params:
        if p.grad is not None:
            _norm = T.norm(p.grad.data)
            norm += _norm
            if _norm > clip_norm:
                p.grad.data /= (_norm / clip_norm)
    return norm

def init_lstm(lstm):
    for name, param in lstm.named_parameters():
        if name.startswith('weight_ih'):
            s = NP.sqrt(6. / (param.size()[0] + param.size()[1]))
            p = T.Tensor(RNG.uniform(low=-s, high=s, size=param.size())).cuda()
            param.data.copy_(p)
        elif name.startswith('weight_hh'):
            shape = (param.size()[0], param.size()[1])
            a = RNG.normal(0, 1, shape)
            u, _, v = NP.linalg.svd(a, full_matrices=False)
            q = u if u.shape == shape else v
            q = q.reshape(shape)
            p = T.Tensor(q).cuda()
            param.data.copy_(p)
        elif name.startswith('bias'):
            param.data.fill_(0)


class Residual(NN.Module):
    def __init__(self,size):
        NN.Module.__init__(self)
        self.size = size
        self.linear = NN.Linear(size, size)
        self.relu = NN.LeakyReLU()

    def forward(self, x):
        return self.relu(self.linear(x) + x)


class Embedder(NN.Module):
    def __init__(self,
                 output_size=128,
                 char_embed_size=128,
                 num_layers=1,
                 num_chars=256,
                 ):
        NN.Module.__init__(self)
        self._output_size = output_size
        self._char_embed_size = char_embed_size
        self._num_layers = num_layers

        self.embed = NN.DataParallel(NN.Embedding(num_chars, char_embed_size))
        self.rnn = NN.LSTM(
                char_embed_size,
                output_size // 2,
                num_layers,
                bidirectional=True,
                )
        init_lstm(self.rnn)

    def forward(self, chars, length):
        num_layers = self._num_layers
        batch_size = chars.size()[0]
        output_size = self._output_size

        embed_seq = self.embed(chars).permute(1, 0, 2)
        initial_state = (
                tovar(T.zeros(num_layers * 2, batch_size, output_size // 2)),
                tovar(T.zeros(num_layers * 2, batch_size, output_size // 2)),
                )
        embed, (h, c) = dynamic_rnn(self.rnn, embed_seq, length, initial_state)
        h = h.permute(1, 0, 2)
        return h[:, -2:].contiguous().view(batch_size, output_size)

def calc_dists(hidden_states, hidden_state_lengths):
    def kurt(x, dim):
        return (((x - x.mean(dim, True)) ** 4).sum(dim) / x.size()[dim]) ** 0.25
    means_d = []
    stds_d = []
    #kurts_d = []
    for h, l in zip(hidden_states, hidden_state_lengths):
        mask = length_mask((h.size()[0], h.size()[2]), l)
        m = h.sum(2) / l.unsqueeze(1).float()
        s = T.sqrt(((h - m.unsqueeze(2) * mask.unsqueeze(1).float()) ** 2).sum(2) / l.unsqueeze(1).float())
        #k = T.pow(((h - m.unsqueeze(2) * mask.unsqueeze(1).float()) ** 4).sum(2) / l.unsqueeze(1).float(), 0.25)
        means_d.append((m.mean(0),m.std(0)))
        #means_d.append((s.mean(0),s.std(0)))
        #means_d.append((k.mean(0), k.std(0)))
        #stds_d.append((m.std(0),m.std(0)))
        #stds_d.append((s.std(0),s.std(0)))
        #stds_d.append((k.std(0),k.std(0)))
        #kurts_d.append((kurt(m, 0), m.std(0)))
        #kurts_d.append((kurt(s, 0), s.std(0)))
        #kurts_d.append((kurt(k, 0), k.std(0)))
    return means_d + stds_d# + kurts_d

class Generator(NN.Module):
    def __init__(self,
                 frame_size=200,
                 embed_size=200,
                 noise_size=100,
                 state_size=1024,
                 num_layers=1,
                 ):
        NN.Module.__init__(self)
        self._frame_size = frame_size
        self._noise_size = noise_size
        self._state_size = state_size
        self._embed_size = embed_size
        self._num_layers = num_layers

        self.rnn = NN.ModuleList()
        lstm = NN.LSTMCell(frame_size + embed_size + noise_size, state_size)
        init_lstm(lstm)
        self.rnn.append(NN.DataParallel(lstm))
        for _ in range(1, num_layers):
            lstm = NN.LSTMCell(state_size, state_size)
            init_lstm(lstm)
            self.rnn.append(NN.DataParallel(lstm))
        self.proj = NN.DataParallel(NN.Linear(state_size, frame_size))
        self.stopper = NN.DataParallel(NN.Linear(state_size, 1))

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
        generating = T.ones(batch_size).long()
        length = T.zeros(batch_size).long()

        x_list = []
        s_list = []
        stop_list = []
        for t in range(nframes):
            z_t = z[:, t]
            _x = T.cat([x_t, z_t], 1)
            lstm_h[0], lstm_c[0] = self.rnn[0](_x, (lstm_h[0], lstm_c[0]))
            for i in range(1, num_layers):
                lstm_h[i], lstm_c[i] = self.rnn[i](lstm_h[i-1], (lstm_h[i], lstm_c[i]))
            x_t = self.proj(lstm_h[-1]).tanh_()
            logit_s_t = self.stopper(lstm_h[-1])
            s_t = log_sigmoid(logit_s_t)
            s1_t = log_one_minus_sigmoid(logit_s_t)

            logp_t = T.cat([s1_t, s_t], 1)
            p_t = logp_t.exp()
            stop_t = p_t.multinomial()
            length += generating

            x_list.append(x_t)
            s_list.append(logit_s_t.squeeze())
            stop_list.append(stop_t)

            stop_t = stop_t.squeeze()
            generating *= (stop_t.data == 0).long().cpu()
            if generating.sum() == 0:
                break

        x = T.cat(x_list, 1)
        s = T.stack(s_list, 1)

        return x, s, stop_list, tovar(length * frame_size)

_cnn_struct = [
        [7, 2, 128],
        [7, 2, 128],
        [7, 2, 128],
        [7, 2, 256],
        [7, 2, 256],
        [7, 2, 256],
        [7, 2, 512],
        [7, 2, 512],
        [7, 2, 512],
        ]

class Discriminator(NN.Module):
    def __init__(self,
                 state_size=1024,
                 embed_size=200,
                 num_layers=1,
                 cnn_struct=_cnn_struct):
        NN.Module.__init__(self)
        self._state_size = state_size
        self._embed_size = embed_size
        self._num_layers = num_layers
        self._cnn_struct = cnn_struct
        
        self.cnn = NN.ModuleList()
        self.cnn_struct = cnn_struct
        infilters = 1
        for idx, layer in enumerate(cnn_struct):
            kernel, stride, outfilters = layer[0],layer[1],layer[2]

            conv = NN.Conv1d(infilters, outfilters, kernel, stride=stride, padding=(kernel - 1) // 2)
            self.cnn.append(NN.DataParallel(conv))

            infilters = outfilters
        frame_size = outfilters
        self.frame_size = frame_size
        self._frame_size = frame_size
        self.rnn = NN.LSTM(
                frame_size + embed_size,
                state_size // 2,
                num_layers,
                bidirectional=True,
                )
        init_lstm(self.rnn)
        self.residual_net = NN.DataParallel(NN.Sequential(
                Residual(state_size),
                Residual(state_size),
                Residual(state_size),
                Residual(state_size)
            ))
        self.classifier_word = NN.DataParallel(NN.Sequential(
                NN.Linear(state_size, state_size // 2),
                NN.LeakyReLU(),
                NN.Linear(state_size // 2, 1),
                ))
        self.classifier = NN.DataParallel(NN.Sequential(
                NN.Linear(state_size, state_size // 2),
                NN.LeakyReLU(),
                NN.Linear(state_size // 2, 1),
                ))

    def forward(self, x, length, c, percent_used = 0.1):
        frame_size = self._frame_size
        state_size = self._state_size
        num_layers = self._num_layers
        embed_size = self._embed_size
        batch_size, maxlen = x.size()

        xold = x

        initial_state = (
                tovar(T.zeros(num_layers * 2, batch_size, state_size // 2)),
                tovar(T.zeros(num_layers * 2, batch_size, state_size // 2)),
                )
        cnn_outputs = []
        cnn_output_lengths = []
        cnn_output = xold.unsqueeze(1)
        nframes = length
        cnn_output = cnn_output * length_mask((batch_size, cnn_output.size()[2]), nframes).unsqueeze(1)
        for cnn_layer, (_, stride, _) in zip(self.cnn, self.cnn_struct):
            cnn_output = F.leaky_relu(cnn_layer(cnn_output))
            nframes = (nframes + stride - 1) / stride
            cnn_output = cnn_output * length_mask((batch_size, cnn_output.size()[2]), nframes).unsqueeze(1)
            cnn_outputs.append(cnn_output)
            cnn_output_lengths.append(nframes)
        x = cnn_output
        x = x.permute(0, 2, 1)
        #x = x.view(32, nframes_max, frame_size)
        max_nframes = x.size()[1]
        c = c.unsqueeze(1).expand(batch_size, max_nframes, embed_size)
        x2 = T.cat([x, c], 2).permute(1,0,2)
        lstm_out, (_, _) = dynamic_rnn(self.rnn, x2, nframes, initial_state)
        lstm_out = lstm_out.permute(1, 0, 2)
        max_nframes = lstm_out.size()[1]

        conv_out = lstm_out.view(batch_size * max_nframes, state_size)
        res_out = self.residual_net(conv_out)
        classifier_out = self.classifier(res_out).view(batch_size, max_nframes)
        classifier_word = self.classifier_word(res_out).view(batch_size, max_nframes)
        
        return classifier_out, classifier_word, cnn_outputs, cnn_output_lengths, nframes

parser = argparse.ArgumentParser()
parser.add_argument('--critic_iter', default=100, type=int)
parser.add_argument('--rnng_layers', type=int, default=1)
parser.add_argument('--rnnd_layers', type=int, default=1)
parser.add_argument('--framesize', type=int, default=200, help='# of amplitudes to generate at a time for RNN')
parser.add_argument('--noisesize', type=int, default=100, help='noise vector size')
parser.add_argument('--gstatesize', type=int, default=1024, help='RNN state size')
parser.add_argument('--dstatesize', type=int, default=1024, help='RNN state size')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--dgradclip', type=float, default=1)
parser.add_argument('--ggradclip', type=float, default=0.1)
parser.add_argument('--dlr', type=float, default=1e-4)
parser.add_argument('--glr', type=float, default=1e-4)
parser.add_argument('--modelname', type=str, default = '')
parser.add_argument('--modelnamesave', type=str, default='')
parser.add_argument('--modelnameload', type=str, default='')
parser.add_argument('--just_run', type=str, default='')
parser.add_argument('--loaditerations', type=int, default=0)
parser.add_argument('--logdir', type=str, default='.', help='log directory')
parser.add_argument('--dataset', type=str, default='dataset.h5')
parser.add_argument('--embedsize', type=int, default=100)
parser.add_argument('--minwordlen', type=int, default=1)
parser.add_argument('--maxlen', type=int, default=40000, help='maximum sample length (0 for unlimited)')
parser.add_argument('--noisescale', type=float, default=0.1)
parser.add_argument('--g_optim', default = 'boundary_seeking')
parser.add_argument('--require_acc', type=float, default=0.5)
parser.add_argument('--lambda_pg', type=float, default=0.1)
parser.add_argument('--pretrain_d', type=int, default=0)

args = parser.parse_args()
args.conditional = True
if args.just_run not in ['', 'gen', 'dis']:
    print('just run should be empty string, gen, or dis. Other values not accepted')
    sys.exit(0)
lambda_fp = 1
if len(args.modelname) > 0:
    modelnamesave = args.modelname
    modelnameload = None
else:
    modelnamesave = args.modelnamesave
    modelnameload = args.modelnameload

print modelnamesave
print args

batch_size = args.batchsize

dataset_h5, maxlen, dataloader, dataloader_val, keys_train, keys_val = \
        dataset.dataloader(batch_size, args, maxlen=args.maxlen, frame_size=args.framesize)
maxcharlen_train = max(len(k) for k in keys_train)

def logdirs(logdir, modelnamesave):
    logdir = (
            logdir + '/%s-%s' % 
            (modelnamesave, datetime.datetime.strftime(
                datetime.datetime.now(), '%Y%m%d%H%M%S')
                )
            )
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    elif not os.path.isdir(logdir):
        raise IOError('%s is not a directory' % logdir)
    return logdir
log_train_d = logdirs(args.logdir, modelnamesave)
png_file = '%s/temp.png' % log_train_d
wav_file = '%s/temp.wav' % log_train_d


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

d = Discriminator(
        state_size=args.dstatesize,
        embed_size=args.embedsize,
        num_layers=args.rnnd_layers,
        ).cuda()

def add_waveform_summary(writer, word, sample, gen_iter, tag='plot'):
    PL.plot(sample)
    PL.savefig(png_file)
    PL.close()
    with open(png_file, 'rb') as f:
        imgbuf = f.read()
    img = Image.open(png_file)
    summary = TF.Summary.Image(
            height=img.height,
            width=img.width,
            colorspace=3,
            encoded_image_string=imgbuf
            )
    summary = TF.Summary.Value(tag='%s/%s' % (tag, word), image=summary)
    writer.add_summary(TF.Summary(value=[summary]), gen_iter)

def add_audio_summary(writer, word, sample, length, gen_iter, tag='audio'):
    librosa.output.write_wav(wav_file, sample, sr=8000)
    with open(wav_file, 'rb') as f:
        wavbuf = f.read()
    summary = TF.Summary.Audio(
            sample_rate=8000,
            num_channels=1,
            length_frames=length,
            encoded_audio_string=wavbuf,
            content_type='audio/wav'
            )
    summary = TF.Summary.Value(tag='%s/%s' % (tag, word), audio=summary)
    d_train_writer.add_summary(TF.Summary(value=[summary]), gen_iter)

d_train_writer = TF.summary.FileWriter(log_train_d)

# Add real waveforms
_, _, samples, lengths, cseq, cseq_fixed, clen_fixed = dataloader_val.next()
for i in range(batch_size):
    add_waveform_summary(d_train_writer, cseq[i], samples[i, :lengths[i]], 0, 'real_plot')
    add_audio_summary(d_train_writer, cseq[i], samples[i, :lengths[i]], lengths[i], 0, 'real_audio')

cseq_fixed = NP.array(cseq_fixed)
clen_fixed = NP.array(clen_fixed)
cseq_fixed, clen_fixed = tovar(cseq_fixed, clen_fixed)
cseq_fixed = cseq_fixed.long()
clen_fixed = clen_fixed.long()

gen_iter = 0
dis_iter = 0
epoch = 1
l = 10
alpha = 0.1
baseline = None

param_g = list(g.parameters()) + list(e_g.parameters())
param_d = list(d.parameters()) + list(e_d.parameters())

opt_g = T.optim.Adam(param_g, lr=args.glr)
opt_d = T.optim.Adam(param_d, lr=args.dlr)

def discriminate(d, data, length, embed, target, target_w, real, real_w):
    cls, cls_w, _, _, nframes = d(data, length, embed)
    target = tovar(T.ones(*(cls.size())) * target)
    target_w = tovar(T.ones(*(cls_w.size())) * target_w)
    weight = length_mask(cls.size(), nframes)
    loss_c = binary_cross_entropy_with_logits_per_sample(cls, target, weight=weight) / nframes.float()
    loss_c = loss_c.mean()
    loss_w = binary_cross_entropy_with_logits_per_sample(cls_w, target_w, weight=weight) / nframes.float()
    loss_w = loss_w.mean()
    correct = ((cls.data > 0) if real else (cls.data < 0)).float() * weight.data
    correct = correct.sum()
    correct_w = ((cls_w.data > 0) if real_w else (cls_w.data < 0)).float() * weight.data
    correct_w = correct_w.sum()
    num = weight.data.sum()
    acc = correct / num
    acc_w = correct_w / num
    return cls, cls_w, nframes, target, target_w, weight, loss_c, loss_w, loss_c + loss_w, acc, acc_w

if __name__ == '__main__':
    if modelnameload:
        if len(modelnameload) > 0:
            d = T.load('%s-dis-%05d' % (modelnameload, args.loaditerations))
            g = T.load('%s-gen-%05d' % (modelnameload, args.loaditerations))
            e_g = T.load('%s-eg-%05d' % (modelnameload, args.loaditerations))
            e_d = T.load('%s-ed-%05d' % (modelnameload, args.loaditerations))

#    if not args.pretrain_d and not modelnameload:
#        print 'Pretraining D'
#        for p in param_g:
#            p.requires_grad = False
#        for p in param_d:
#            p.requires_grad = True
#        for j in range(args.critic_iter):
#            with Timer.new('load', print_=False):
#                epoch, batch_id, _real_data, _real_len, _, _cs, _cl = dataloader.next()
#                _, _, _wrong_data, _wrong_len, _, _, _ = dataloader.next()
#                ck2, _cs2, _cl2, _, _ = dataset.pick_words(
#                        batch_size, maxlen, dataset_h5, keys_train, maxcharlen_train, args, skip_samples=True)
#
#            dis_iter += 1
#
#            with Timer.new('train_d', print_=False):
#                real_data = tovar(_real_data)
#                real_len = tovar(_real_len).long()
#                wrong_data = tovar(_wrong_data)
#                wrong_len = tovar(_wrong_len).long()
#                cs = tovar(_cs).long()
#                cl = tovar(_cl).long()
#                embed_d = e_d(cs, cl)
#                cs2 = tovar(_cs2).long()
#                cl2 = tovar(_cl2).long()
#                embed_d2 = e_d(cs2, cl2)
#
#                _, cls_d, _, _, _, _, _, loss_d, _, _, acc_d = discriminate(d, real_data, real_len, embed_d, 1, 1, True, True)
#                _, cls_d_wrong, _, _, _, _, _, loss_d_wrong, _, _, acc_d_wrong = discriminate(d, wrong_data, wrong_len, embed_d2, 0, 0, False, False)
#
#                loss = loss_d + loss_d_wrong
#                opt_d.zero_grad()
#                loss.backward()
#                check_grad(param_d)
#                e_d_grad_norm = sum(T.norm(p.grad.data) ** 2 for p in e_d.parameters() if p.grad is not None) ** 0.5
#                d_grad_norm = clip_grad(param_d, args.dgradclip)
#                opt_d.step()
#
#            loss_d, loss, cls_d, cls_d_wrong = tonumpy(loss_d, loss, cls_d, cls_d_wrong)
#
#            d_train_writer.add_summary(
#                    TF.Summary(
#                        value=[
#                            TF.Summary.Value(tag='loss_d', simple_value=loss_d),
#                            TF.Summary.Value(tag='loss', simple_value=loss),
#                            TF.Summary.Value(tag='cls_d/mean', simple_value=cls_d.mean()),
#                            TF.Summary.Value(tag='cls_d/std', simple_value=cls_d.std()),
#                            TF.Summary.Value(tag='cls_d_wrong/mean', simple_value=cls_d_wrong.mean()),
#                            TF.Summary.Value(tag='cls_d_wrong/std', simple_value=cls_d_wrong.std()),
#                            TF.Summary.Value(tag='acc_d', simple_value=acc_d),
#                            TF.Summary.Value(tag='acc_d_wrong', simple_value=acc_d_wrong),
#                            TF.Summary.Value(tag='d_grad_norm', simple_value=d_grad_norm),
#                            ]
#                        ),
#                    dis_iter
#                    )
#
#            # Validation
#            epoch, batch_id, _real_data, _real_len, _, _cs, _cl = dataloader.next()
#            _, _, _wrong_data, _wrong_len, _, _, _ = dataloader.next()
#            ck2, _cs2, _cl2, _, _ = dataset.pick_words(
#                    batch_size, maxlen, dataset_h5, keys_train, maxcharlen_train, args, skip_samples=True)
#            real_data = tovar(_real_data)
#            real_len = tovar(_real_len).long()
#            wrong_data = tovar(_wrong_data)
#            wrong_len = tovar(_wrong_len).long()
#            cs = tovar(_cs).long()
#            cl = tovar(_cl).long()
#            embed_d = e_d(cs, cl)
#            cs2 = tovar(_cs2).long()
#            cl2 = tovar(_cl2).long()
#            embed_d2 = e_d(cs2, cl2)
#
#            _, cls_d, _, _, _, _, _, loss_d_val, _, _, acc_d_val = discriminate(d, real_data, real_len, embed_d, 1, 1, True, True)
#            _, cls_d_wrong, _, _, _, _, _, loss_d_wrong_val, _, _, acc_d_wrong_val = discriminate(d, wrong_data, wrong_len, embed_d2, 0, 0, False, False)
#            loss_val = tonumpy(loss_d_val + loss_d_wrong_val)[0]
#
#            d_train_writer.add_summary(
#                    TF.Summary(
#                        value=[
#                            TF.Summary.Value(tag='loss_val', simple_value=loss_val),
#                            TF.Summary.Value(tag='acc_d_val', simple_value=acc_d_val),
#                            TF.Summary.Value(tag='acc_d_wrong_val', simple_value=acc_d_wrong_val),
#                            ]
#                        ),
#                    dis_iter
#                    )
#
#            print 'D', epoch, batch_id, loss, loss_val, 'Train Acc:', acc_d, acc_d_wrong, 'Val Acc:', acc_d_val, acc_d_wrong_val, 'Grad Norms:', e_d_grad_norm, d_grad_norm, 'Time:', Timer.get('load'), Timer.get('train_d')
#            if dis_iter % 10000 == 0:
#                T.save(d, '%s-dis-pretrain-%05d' % (modelnamesave, dis_iter))
#                T.save(e_d, '%s-ed-pretrain-%05d' % (modelnamesave, dis_iter))
#    else:
#        if not modelnameload:
#            d_pretrain = T.load('%s-dis-pretrain-%05d' % (modelnamesave, args.pretrain_d))
#            e_d = T.load('%s-ed-pretrain-%05d' % (modelnamesave, args.pretrain_d))
#            d.state_dict().update({k: v for k, v in d_pretrain.state_dict().items() if not k.startswith('cnn')})

    while True:
        _epoch = epoch

        for p in param_g:
            p.requires_grad = False
        for p in param_d:
            p.requires_grad = True
        for j in range(args.critic_iter):
            dis_iter += 1
            with Timer.new('load', print_=False):
                epoch, batch_id, _real_data, _real_len, _, _cs, _cl = dataloader.next()
                _, _cs2, _cl2, _, _ = dataset.pick_words(
                        batch_size, maxlen, dataset_h5, keys_train, maxcharlen_train, args, skip_samples=True)

            with Timer.new('train_d', print_=False):
                noise = tovar(RNG.randn(*_real_data.shape) * args.noisescale)
                real_data = tovar(_real_data) + noise
                real_len = tovar(_real_len).long()
                cs = tovar(_cs).long()
                cl = tovar(_cl).long()
                cs2 = tovar(_cs2).long()
                cl2 = tovar(_cl2).long()

                embed_d = e_d(cs, cl)
                embed_g2 = e_g(cs2, cl2)
                embed_d2 = e_d(cs2, cl2)

                cls_d, cls_d_w, _, _, _, _, _, _, loss_d, acc_d, acc_d_w = \
                        discriminate(d, real_data, real_len, embed_d, 0.9, 1, True, True)
                cls_d_x, cls_d_w_x, _, _, _, _, _, _, loss_d_x, acc_d_x, acc_d_w_x = \
                        discriminate(d, real_data, real_len, embed_d2, 0.9, 0, True, False)

                fake_data, _, _, fake_len = g(batch_size=batch_size, length=maxlen, c=embed_g2)
                noise = tovar(T.randn(*fake_data.size()) * args.noisescale)
                fake_data = tovar((fake_data + noise).data)
                fake_data.requires_grad = True
                cls_g, cls_g_w, _, _, _, _, _, _, loss_g, acc_g, acc_g_w = \
                        discriminate(d, fake_data, fake_len, embed_d2, 0, 1, False, True)
                cls_g_x, cls_g_w_x, _, _, _, _, _, _, loss_g_x, acc_g_x, acc_g_w_x = \
                        discriminate(d, fake_data, fake_len, embed_d, 0, 0, False, False)

                loss = loss_d + loss_d_x + loss_g + loss_g_x
                opt_d.zero_grad()
                loss.backward()
                check_grad(param_d)
                d_grad_norm = clip_grad(param_d, args.dgradclip)
                opt_d.step()

            loss_d, loss_d_x, loss_g, loss_g_x, loss, cls_d, cls_d_x, cls_g, cls_g_x = \
                    tonumpy(loss_d, loss_d_x, loss_g, loss_g_x, loss, cls_d, cls_d_x, cls_g, cls_g_x)
            d_train_writer.add_summary(
                    TF.Summary(
                        value=[
                            TF.Summary.Value(tag='loss_d', simple_value=loss_d),
                            TF.Summary.Value(tag='loss_d_x', simple_value=loss_d_x),
                            TF.Summary.Value(tag='loss_g', simple_value=loss_g),
                            TF.Summary.Value(tag='loss_g_x', simple_value=loss_g_x),
                            TF.Summary.Value(tag='loss', simple_value=loss),
                            TF.Summary.Value(tag='cls_d/mean', simple_value=cls_d.mean()),
                            TF.Summary.Value(tag='cls_d/std', simple_value=cls_d.std()),
                            TF.Summary.Value(tag='cls_d_x/mean', simple_value=cls_d_x.mean()),
                            TF.Summary.Value(tag='cls_d_x/std', simple_value=cls_d_x.std()),
                            TF.Summary.Value(tag='cls_g/mean', simple_value=cls_g.mean()),
                            TF.Summary.Value(tag='cls_g/std', simple_value=cls_g.std()),
                            TF.Summary.Value(tag='cls_g_x/mean', simple_value=cls_g_x.mean()),
                            TF.Summary.Value(tag='cls_g_x/std', simple_value=cls_g_x.std()),
                            TF.Summary.Value(tag='acc_d', simple_value=acc_d),
                            TF.Summary.Value(tag='acc_d_x', simple_value=acc_d_x),
                            TF.Summary.Value(tag='acc_g', simple_value=acc_g),
                            TF.Summary.Value(tag='acc_g_x', simple_value=acc_g_x),
                            TF.Summary.Value(tag='d_grad_norm', simple_value=d_grad_norm),
                            ]
                        ),
                    dis_iter
                    )

            accs = [acc_d, acc_d_x, acc_g, acc_g_x, acc_d_w, acc_d_w_x, acc_g_w, acc_g_w_x]
            print 'D', epoch, batch_id, loss, ';'.join('%.03f' % a for a in accs), Timer.get('load'), Timer.get('train_d')

            if acc_d > 0.5 and acc_g > 0.5 and acc_d_x > 0.5 and acc_g_x > 0.5:
                break

        gen_iter += 1
        for p in param_g:
            p.requires_grad = True
        for p in param_d:
            p.requires_grad = False

        _, cs, cl, _, _ = dataset.pick_words(
                batch_size, maxlen, dataset_h5, keys_train, maxcharlen_train, args, skip_samples=True)
        with Timer.new('train_g', print_=False):
            cs = tovar(cs).long()
            cl = tovar(cl).long()
            embed_g = e_g(cs, cl)
            embed_d = e_d(cs, cl)
            fake_data, fake_s, fake_stop_list, fake_len = g(batch_size=batch_size, length=maxlen, c=embed_g)
            noise = tovar(T.randn(*fake_data.size()) * args.noisescale)
            fake_data += noise
            
            cls_g, cls_g_w, hidden_states_g, hidden_states_length_g, nframes_g = d(fake_data, fake_len, embed_d)
            
            #dists_d = calc_dists(hidden_states_d, hidden_states_length_d)
            #dists_g = calc_dists(hidden_states_g, hidden_states_length_g)
            #feature_penalty = 0
            #dists are (object, std) pairs.
            #penalizing z-scores of gen from real distribution
            #Note that the model could not do anything to r[1] by optimizing G.
            #for r, f in zip(dists_d, dists_g):
            #    feature_penalty += T.pow((r[0] - f[0]), 2).mean() / batch_size

            if args.g_optim == 'boundary_seeking':
                target = tovar(T.ones(*(cls_g.size())) * 0.5)   # TODO: add logZ estimate, may be unnecessary
            else:
                target = tovar(T.zeros(*(cls_g.size())))            
            target_w = tovar(T.ones(*(cls_g_w.size())) * 1)
            weight = length_mask(cls_g.size(), nframes_g)
            nframes_max = (fake_len / args.framesize).data.max()
            weight_r = length_mask((batch_size, nframes_max), fake_len / args.framesize)
            loss = binary_cross_entropy_with_logits_per_sample(cls_g, target, weight=weight) / nframes_g.float()
            loss_w = binary_cross_entropy_with_logits_per_sample(cls_g_w, target_w, weight=weight) / nframes_g.float()
            loss = loss + loss_w

            reward = -loss.data
            baseline = reward.mean() if baseline is None else (baseline * 0.5 + reward.mean() * 0.5)
            d_train_writer.add_summary(
                    TF.Summary(
                        value=[
                            TF.Summary.Value(tag='reward_baseline', simple_value=baseline),
                            TF.Summary.Value(tag='reward/mean', simple_value=reward.cpu().numpy().mean()),
                            TF.Summary.Value(tag='reward/std', simple_value=reward.cpu().numpy().std()),
                            ]
                        ),
                    gen_iter
                    )
            reward = (reward - baseline).unsqueeze(1) * weight_r.data

            '''
            fp_raw = tonumpy(feature_penalty)
            if fp_raw  * lambda_fp > 100:
                lambda_fp *= .2
            if fp_raw  * lambda_fp > 10:
                lambda_fp *= .9
            if fp_raw  * lambda_fp < 1:
                lambda_fp *= 1.1
            '''

            loss = loss.mean()
            #loss = _loss + feature_penalty * lambda_fp
            #loss = _loss
            for i, fake_stop in enumerate(fake_stop_list):
                fake_stop.reinforce(args.lambda_pg * reward[:, i:i+1])
            # Debug the gradient norms
            opt_g.zero_grad()
            loss.backward(retain_graph=True)
            loss_grad_dict = {p: p.grad.data.clone() for p in param_g if p.grad is not None}
            loss_grad_norm = sum(T.norm(p.grad.data) for p in param_g if p.grad is not None)
            #opt_g.zero_grad()
            #feature_penalty.backward(T.Tensor([lambda_fp]).cuda(), retain_graph=True)
            #fp_grad_dict = {p: p.grad.data.clone() for p in param_g if p.grad is not None}
            #fp_grad_norm = sum(T.norm(p.grad.data) for p in param_g if p.grad is not None)
            for p in param_g:
                p.requires_grad = False
            for p in g.stopper.parameters():
                p.requires_grad = True
            #opt_g.zero_grad()
            T.autograd.backward(fake_stop_list, [None for _ in fake_stop_list])
            pg_grad_norm = sum(T.norm(p.grad.data) for p in param_g if p.grad is not None)
            # Do the real thing
            for p in param_g:
                if p.grad is not None:
                    if p in loss_grad_dict:
                        p.grad.data += loss_grad_dict[p]
                    #if p in fp_grad_dict:
                    #    p.grad.data += fp_grad_dict[p]

            check_grad(param_g)
            g_grad_norm = clip_grad(param_g, args.ggradclip)
            d_train_writer.add_summary(
                    TF.Summary(
                        value=[
                            TF.Summary.Value(tag='g_grad_norm', simple_value=g_grad_norm),
                            #TF.Summary.Value(tag='feature_penalty', simple_value=tonumpy(feature_penalty)[0]),
                            #TF.Summary.Value(tag='lambda_fp', simple_value=lambda_fp),
                            TF.Summary.Value(tag='g_loss_grad_norm', simple_value=loss_grad_norm),
                            #TF.Summary.Value(tag='g_fp_grad_norm', simple_value=fp_grad_norm),
                            TF.Summary.Value(tag='g_pg_grad_norm', simple_value=pg_grad_norm),
                            ]
                        ),
                    gen_iter
                    )
            opt_g.step()

        if gen_iter % 20 == 0:
            embed_g = e_g(cseq_fixed, clen_fixed)
            fake_data, _, _, fake_len = g(z=z_fixed, c=embed_g)
            fake_data, fake_len = tonumpy(fake_data, fake_len)

            for batch in range(batch_size):
                fake_sample = fake_data[batch, :fake_len[batch]]
                add_waveform_summary(d_train_writer, cseq[batch], fake_sample, gen_iter)

            if gen_iter % 500 == 0:
                for batch in range(batch_size):
                    fake_sample = fake_data[batch, :fake_len[batch]]
                    add_audio_summary(d_train_writer, cseq[batch], fake_sample, fake_len[batch], gen_iter)
                T.save(d, '%s-dis-%05d' % (modelnamesave, gen_iter + args.loaditerations))
                T.save(g, '%s-gen-%05d' % (modelnamesave, gen_iter + args.loaditerations))
                T.save(e_g, '%s-eg-%05d' % (modelnamesave, gen_iter + args.loaditerations))
                T.save(e_d, '%s-ed-%05d' % (modelnamesave, gen_iter + args.loaditerations))
        print 'G', gen_iter, tonumpy(loss), Timer.get('train_g')
