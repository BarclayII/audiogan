
from torch.nn import Parameter
from functools import wraps

import torch as T
import torch.nn as NN
import torch.nn.functional as F
import torch.nn.init as INIT
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import weight_norm as torch_weight_norm

import numpy as NP
import numpy.random as RNG
import tensorflow as TF     # for Tensorboard
from numpy import rate

NP.set_printoptions(suppress=True)

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
    return -x - F.softplus(-x)

def binary_cross_entropy_with_logits_per_sample(input, target, weight=None):
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    return loss.sum(1)
def binary_cross_entropy_with_logits(input, target):
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    return loss


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

class Conv1dResidualBottleneck(NN.Module):
    def __init__(self,kernel,stride,infilters,hidden_filters,outfilters, relu = True):
        NN.Module.__init__(self)
        self.infilters = infilters
        self.outfilters = outfilters
        self.conv = NN.Conv1d(infilters, hidden_filters, kernel_size = kernel, stride=2, padding=(kernel - 1) // 2)
        self.convh = NN.Conv1d(hidden_filters, hidden_filters, kernel_size = kernel, stride=1, padding=(kernel - 1) // 2)
        self.deconv = NN.ConvTranspose1d(hidden_filters, outfilters, 4, stride=2, padding=1)
        if relu:
            self.relu = NN.LeakyReLU()
        else:
            self.relu = 0
    def forward(self, x):
        act = self.relu(self.conv(x))
        act = self.convh(act)
        act = self.relu(act)
        act = self.deconv(act) + x
        if self.relu != 0:
            act = self.relu(act)
        return act
class Conv1dResidualBottleKernels(NN.Module):
    def __init__(self,kernel,stride,infilters,hidden_filters,outfilters, relu = True):
        NN.Module.__init__(self)
        self.infilters = infilters
        self.outfilters = outfilters
        self.conv = Conv1dKernels(infilters, hidden_filters//4, kernel_sizes=[1,1,3,3], stride=2)
        self.convh = Conv1dKernels(hidden_filters, hidden_filters//4, kernel_sizes=[1,1,1,3], stride=1)
        self.deconv = NN.ConvTranspose1d(hidden_filters, outfilters, 4, stride=2, padding=1)
        if relu:
            self.relu = NN.LeakyReLU()
        else:
            self.relu = 0
    def forward(self, x):
        act = self.relu(self.conv(x))
        act = self.convh(act)
        act = self.relu(act)
        act = self.deconv(act) + x
        if self.relu != 0:
            act = self.relu(act)
        return act

def check_grad(params):
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.data
        anynan = (g != g).long().sum()
        anybig = (g.abs() > 1e+5).long().sum()
        if anynan or anybig:
            return False
    return True


def clip_grad(params, clip_norm):
    norm = NP.sqrt(
            sum(p.grad.data.norm() ** 2
                for p in params if p.grad is not None
                )
            )
    if norm > clip_norm:
        for p in params:
            if p.grad is not None:
                p.grad /= norm / clip_norm
    return norm

def init_lstm(lstm):
    for name, param in lstm.named_parameters():
        if name.startswith('weight_ih'):
            INIT.xavier_uniform(param.data)
        elif name.startswith('weight_hh'):
            INIT.orthogonal(param.data)
        elif name.startswith('bias'):
            INIT.constant(param.data, 0)

def init_weights(module):
    for name, param in module.named_parameters():
        if name.find('weight') != -1:
            INIT.xavier_uniform(param.data)
        elif name.find('bias') != -1:
            INIT.constant(param.data, 0)

class Conv1dKernels(NN.Module):
    def __init__(self,infilters, outfilters, kernel_sizes, stride):
        NN.Module.__init__(self)
      
        self.convs = NN.ModuleList([NN.Conv1d(infilters, outfilters, kernel_size=kernel, 
                                stride=1, padding=(kernel-1)/2)
                        for kernel in kernel_sizes])
    
    def forward(self, x):
        conv_outs = [c(x) for c in self.convs]
        return T.cat(conv_outs,1)/10


class Residual(NN.Module):
    def __init__(self,size, relu = True):
        NN.Module.__init__(self)
        self.size = size
        self.linear = NN.Linear(size, size)
        if relu:
            self.relu = NN.LeakyReLU()
        else:
            self.relu = False

    def forward(self, x):
        if self.relu:
            return self.relu(self.linear(x) + x)
        else:
            return self.linear(x) + x
class Highway(NN.Module):
    def __init__(self, size):
        NN.Module.__init__(self)
        self.gater = NN.Linear(size, size)
        self.gater.bias.data.fill_(-1)
        self.transformer = NN.Linear(size, size)

    def forward(self, x):
        g = F.sigmoid(self.gater(x))
        return g * F.leaky_relu(self.transformer(x)) + (1 - g) * x


class ResidualConv(NN.Module):
    def __init__(self,size, kernel_size, relu = True):
        NN.Module.__init__(self)
        self.conv = NN.Conv1d(size, size, kernel_size=kernel_size, 
                                stride=1, padding=(kernel_size-1)/2)
        if relu:
            self.relu = NN.LeakyReLU()
        else:
            self.relu = False

    def forward(self, x):
        if self.relu:
            return self.relu(self.conv(x) + x)
        else:
            return self.conv(x) + x

class ConvMask(NN.Module):
    def __init__(self):
        NN.Module.__init__(self)

    def forward(self, x):
        global convlengths
        mask = length_mask((x.size()[0], x.size()[2]),convlengths).unsqueeze(1)
        x = x * mask
        return x


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

def moment(x, exp,lengths):
    x_size = tonumpy(x.size()[1])
    mask = length_mask((x.size()[0], x.size()[2]),lengths).unsqueeze(1)
    x = x * mask
    m = x.sum()/lengths.float().sum()/x_size
    if exp == 1:
        return m
    total_diff = (x - m.expand_as(x)) * mask
    exp_diff = total_diff ** exp
    mean_exp = exp_diff.sum() / lengths.float().sum()/x_size
    return mean_exp ** (1./exp)

def moment_by_index(x, exp,lengths):
    x_size = tonumpy(x.size()[1])
    mask = length_mask((x.size()[0], x.size()[2]),lengths).unsqueeze(1)
    x = x * mask
    m = x.sum(0).sum(1)/lengths.sum().float()
    if exp == 1:
        return m
    total_diff = (x - m.unsqueeze(0).unsqueeze(2)) * mask
    exp_diff = total_diff ** exp
    mean_exp = exp_diff.sum(0).sum(1) / lengths.sum().float()
    return mean_exp ** (1./exp)

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
                 noise_size=32,
                 state_size=1025,
                 num_layers=1,
                 maxlen=1
                 ):
        NN.Module.__init__(self)
        self._frame_size = frame_size
        self._noise_size = noise_size
        self._state_size = state_size
        self._embed_size = embed_size
        self._num_layers = num_layers
        self._maxlen = maxlen
        input_size = noise_size + embed_size
        self._last_hidden_size = last_hidden_size = state_size * maxlen
        self.stopper_conv = NN.DataParallel(NN.Sequential(
                Conv1dKernels(1024, 256, kernel_sizes=[1,3,3,5], stride=1),
                NN.LeakyReLU(),
                Conv1dKernels(1024, 256, kernel_sizes=[1,3,3,5], stride=1),
                NN.LeakyReLU(),
                NN.Conv1d(1024,1,kernel_size=3,stride=1,padding=1)
                ))
        
        ''' Conv1dBottleneck(NN.Module):
    def __init__(self,kernel,stride,infilters,hidden_filters,outfilters, relu = True):
        '''
        self.deconv1 = NN.DataParallel(NN.Sequential(
                Conv1dKernels(input_size, 128, kernel_sizes=[1,1,1,3], stride=1),
                NN.LeakyReLU(),
                Conv1dKernels(512, 256, kernel_sizes=[1,1,1,3], stride=1),
                NN.LeakyReLU(),
                Conv1dResidualBottleneck(kernel=3, stride=2, infilters = 1024, hidden_filters = 2048, outfilters = 1024),
                Conv1dResidualBottleneck(kernel=3, stride=2, infilters = 1024, hidden_filters = 2048, outfilters = 1024),
                Conv1dResidualBottleneck(kernel=3, stride=2, infilters = 1024, hidden_filters = 2048, outfilters = 1024),
                NN.ConvTranspose1d(1024, 1024, 4, stride=2, padding=1),
                NN.LeakyReLU(),
                ))
        self.deconv2 = NN.DataParallel(NN.Sequential(
                Conv1dKernels(1024, 256, kernel_sizes=[1,1,3,3], stride=1),
                NN.LeakyReLU(),
                Conv1dKernels(1024, 256, kernel_sizes=[1,1,3,3], stride=1),
                NN.LeakyReLU(),
                Conv1dResidualBottleneck(kernel=3, stride=2, infilters = 1024, hidden_filters = 1024, outfilters = 1024),
                Conv1dResidualBottleneck(kernel=3, stride=2, infilters = 1024, hidden_filters = 1024, outfilters = 1024),
                Conv1dResidualBottleneck(kernel=3, stride=2, infilters = 1024, hidden_filters = 1024, outfilters = 1024),
                NN.ConvTranspose1d(1024, 1024, 4, stride=2, padding=1),
                NN.LeakyReLU(),
                ))
        
        self.conv1 = NN.DataParallel(NN.Sequential(
                Conv1dKernels(1024, 512, kernel_sizes=[1,3,3,5], stride=1),
                NN.LeakyReLU(),
                NN.Conv1d(2048,1024,kernel_size=3,stride=1,padding=1)
                ))
        self.conv2 = NN.DataParallel(NN.Sequential(
                NN.Conv1d(1024,1025,kernel_size=3,stride=1,padding=1)
                ))
        self.conv3 = NN.DataParallel(NN.Sequential(
                Conv1dKernels(1025, 512, kernel_sizes=[1,3,3,5], stride=1),
                NN.LeakyReLU(),
                NN.Conv1d(2048,1025,kernel_size=3,stride=1,padding=1),
                ))
        init_weights(self.deconv1)
        init_weights(self.deconv2)
        init_weights(self.conv2)
        init_weights(self.conv3)
        init_weights(self.stopper_conv)
        self.sigmoid = NN.Sigmoid()
        self.Softplus = NN.Softplus()
        self.relu = NN.LeakyReLU()
        self.ConvMask = ConvMask()
        #self.tanh_scale = NN.Parameter(T.ones(1))
        #self.tanh_bias = NN.Parameter(T.zeros(1))
    
    def forward(self, batch_size=None, length=None, z=None, c=None):
        global convlengths
        frame_size = self._frame_size
        noise_size = self._noise_size
        state_size = self._state_size
        embed_size = self._embed_size
        num_layers = self._num_layers
        maxlen = self._maxlen

        if z is None:
            nframes = length#div_roundup(length, frame_size)
            z = tovar(T.randn(batch_size, maxlen//4, noise_size))
        else:
            batch_size, _, _ = z.size()

        c = c.unsqueeze(1).expand(batch_size, maxlen//4, embed_size)
        z = T.cat([z, c], 2)
        
        z = z.permute(0,2,1)
        x = self.deconv1(z)
        x = self.deconv2(x)
        x = self.conv1(x) + x
        x = self.relu(x)
        x = self.ConvMask(x)
        '''
        x = self.conv0(z)
        x = self.conv1(x) + x
        x = self.relu(x)
        x = self.conv1(x) + x
        x = self.relu(x)
        x = self.conv1(x) + x
        x = self.relu(x)
        x = self.conv1(x) + x
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x) + x
        x = self.conv3(x) + x
        x = self.conv3(x) + x
        '''
        stop = self.stopper_conv(x)
        stop = stop.view(batch_size,-1)
        stop = stop / stop.abs().mean()
        stop = self.Softplus(stop)
        stop_raw = stop.multinomial()
        stop = stop_raw + 1
        x = self.conv2(x)
        x = self.conv3(x) + x
        x = self.conv3(x) + x
        convlengths = stop.squeeze()
        #x = x.permute(0,2,1)
        #x = self.ConvMask(x)
        #x = x.permute(0,2,1)
        stop = stop.squeeze()
        stop_raw = stop_raw
        #s = T.stack(s_list, 1)
        #p = T.stack(p_list, 1)
  
        return x, stop, stop_raw


class Discriminator(NN.Module):
    def __init__(self,
                 state_size=1024,
                 embed_size=200,
                 num_layers=1,
                 nfreq = 1025,
                 maxlen=0):
        NN.Module.__init__(self)
        self._state_size = state_size
        self._embed_size = embed_size
        self._num_layers = num_layers
        self._frame_size = nfreq
        
        frame_size = args.nfreq
        self.classifier = NN.DataParallel(NN.Sequential(
                NN.Linear(maxlen, maxlen),
                NN.LeakyReLU(),
                NN.Linear(maxlen, 1),
                ))
        self.encoder = NN.DataParallel(NN.Sequential(
                NN.Linear(maxlen, maxlen),
                NN.LeakyReLU(),
                NN.Linear(maxlen, 1),
                ))
        self.conv1 = NN.DataParallel(NN.Sequential(
                ConvMask(),
                Conv1dKernels(1025, 512, kernel_sizes=[1,1,3,3], stride=1),
                NN.LeakyReLU()
                ))
        self.conv2 = NN.DataParallel(NN.Sequential(
                Conv1dKernels(2048, 512, kernel_sizes=[1,1,3,3], stride=1),
                NN.LeakyReLU()
                ))
        self.conv3 = NN.DataParallel(NN.Sequential(
                Conv1dKernels(2048, 512, kernel_sizes=[1,1,3,3], stride=1),
                NN.LeakyReLU()
                ))
        self.conv4 = NN.DataParallel(NN.Sequential(
                NN.Conv1d(2048,1024,kernel_size=3,stride=1,padding=1),
                NN.LeakyReLU(),
                ConvMask(),
                ))
        self.highway = NN.DataParallel(NN.Sequential(*[Highway(1024) for _ in range(4)]))
        self.conv5 = NN.DataParallel(NN.Sequential(
                NN.Conv1d(1024,1,kernel_size=3,stride=1,padding=1),
                ConvMask(),
                ))
        self.ConvMask = ConvMask()
        init_weights(self.highway)
        init_weights(self.conv1)
        init_weights(self.conv2)
        init_weights(self.conv3)
        init_weights(self.conv4)
        init_weights(self.conv5)
        init_weights(self.classifier)
        init_weights(self.encoder)

    def forward(self, x, length, c, percent_used = 0.1):
        global convlengths
        frame_size = self._frame_size
        state_size = self._state_size
        num_layers = self._num_layers
        embed_size = self._embed_size
        batch_size, nfreq, maxlen = x.size()
        
        max_nframes = x.size()[2]
        convlengths = length
        x = self.ConvMask(x)
        h1 = self.ConvMask(self.conv1(x))
        h2 = self.ConvMask(self.conv2(h1))
        h3 = self.ConvMask(self.conv3(h2))
        h4 = self.ConvMask(self.conv4(h3))
        x = h4
        conv_acts = [h1,h2,h3,h4]
        x = x.permute(0,2,1)
        x = self.highway(x.contiguous().view(batch_size * max_nframes, -1))
        x = x.view(batch_size, max_nframes,-1)
        x = x.permute(0,2,1)
        x = self.conv5(x)
        x = x.view(batch_size,-1)
        x = self.ConvMask(x.unsqueeze(2)).squeeze()

        ranking = self.encoder(x).squeeze()
        classifier_out = self.classifier(x).squeeze()

        #code_unitnorm = code / (code.norm(2, 1, keepdim=True) + 1e-4)
        #c_unitnorm = c / (c.norm(2, 1, keepdim=True) + 1e-4)
        #ranking = T.bmm(code_unitnorm.unsqueeze(1), c_unitnorm.unsqueeze(2)).squeeze()

        return classifier_out, ranking, conv_acts

parser = argparse.ArgumentParser()
parser.add_argument('--critic_iter', default=1000, type=int)
parser.add_argument('--rnng_layers', type=int, default=2)
parser.add_argument('--rnnd_layers', type=int, default=2)
parser.add_argument('--framesize', type=int, default=200, help='# of amplitudes to generate at a time for RNN')
parser.add_argument('--noisesize', type=int, default=64, help='noise vector size')
parser.add_argument('--gstatesize', type=int, default=1025, help='RNN state size')
parser.add_argument('--dstatesize', type=int, default=512, help='RNN state size')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--dgradclip', type=float, default=1)
parser.add_argument('--ggradclip', type=float, default=1)
parser.add_argument('--dlr', type=float, default=1e-4)
parser.add_argument('--glr', type=float, default=1e-4)
parser.add_argument('--modelname', type=str, default = '')
parser.add_argument('--modelnamesave', type=str, default='')
parser.add_argument('--modelnameload', type=str, default='')
parser.add_argument('--just_run', type=str, default='')
parser.add_argument('--loaditerations', type=int, default=0)
parser.add_argument('--logdir', type=str, default='.', help='log directory')
parser.add_argument('--dataset', type=str, default='data-spect.h5')
parser.add_argument('--embedsize', type=int, default=64)
parser.add_argument('--minwordlen', type=int, default=1)
parser.add_argument('--maxlen', type=int, default=24, help='maximum sample length (0 for unlimited)')
parser.add_argument('--noisescale', type=float, default=10.)
parser.add_argument('--g_optim', default = 'boundary_seeking')
parser.add_argument('--require_acc', type=float, default=0.7)
parser.add_argument('--lambda_pg', type=float, default=.1)
parser.add_argument('--lambda_rank', type=float, default=.1)
parser.add_argument('--lambda_loss', type=float, default=1)
parser.add_argument('--lambda_fp', type=float, default=.1)
parser.add_argument('--lambda_fp_conv', type=float, default=.1)
parser.add_argument('--pretrain_d', type=int, default=0)
parser.add_argument('--nfreq', type=int, default=1025)
parser.add_argument('--gencatchup', type=int, default=1)


args = parser.parse_args()
args.conditional = True
if args.just_run not in ['', 'gen', 'dis']:
    print('just run should be empty string, gen, or dis. Other values not accepted')
    sys.exit(0)
if len(args.modelname) > 0:
    modelnamesave = args.modelname
    modelnameload = None
else:
    modelnamesave = args.modelnamesave
    modelnameload = args.modelnameload
lambda_fp_g = args.lambda_fp/1000.
lambda_fp_conv = args.lambda_fp_conv/1000.
lambda_pg_g = args.lambda_pg/100.
lambda_rank_g = args.lambda_rank/10.
lambda_loss_g = args.lambda_loss/10.

args.framesize = args.nfreq
print modelnamesave
print args
reward_scatter = []
length_scatter = []
batch_size = args.batchsize
batch_size_massive = batch_size * 10
dataset_h5, maxlen, dataloader, dataloader_val, keys_train, keys_val = \
        dataset.dataloader(batch_size_massive, args, maxlen=args.maxlen, frame_size=args.framesize)
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
maxlen = args.maxlen

g = Generator(
        frame_size=args.framesize,
        noise_size=args.noisesize,
        state_size=args.gstatesize,
        embed_size=args.embedsize,
        num_layers=args.rnng_layers,
        maxlen=args.maxlen
        ).cuda()
nframes = maxlen#div_roundup(maxlen, args.framesize)
z_fixed = tovar(RNG.randn(batch_size, maxlen//4, args.noisesize))

e_g = Embedder(args.embedsize).cuda()
e_d = Embedder(args.embedsize).cuda()

d = Discriminator(
        state_size=args.dstatesize,
        embed_size=args.embedsize,
        num_layers=args.rnnd_layers,
        nfreq = args.nfreq,
        maxlen = args.maxlen).cuda()

def spect_to_audio(spect):
    spect = NP.clip(spect, a_min = 0, a_max=None)
    spect = spect * 20
    audio_sample = librosa.istft(spect)
    for i in range(100):
        w_ = audio_sample
        y = librosa.stft(audio_sample)
        z = spect * NP.exp(1j * NP.angle(y))
        audio_sample = librosa.istft(z)
    return audio_sample


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

def add_scatterplot(writer, reward_scatter, length_scatter, gen_iter, tag = 'scatterplot'):
    reward_scatter = NP.stack(reward_scatter)
    s = reward_scatter.std()
    reward_scatter = reward_scatter + \
        NP.random.rand(reward_scatter.shape[0], reward_scatter.shape[1]) * s / 100.
    PL.scatter(length_scatter, reward_scatter)
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
    summary = TF.Summary.Value(tag='%s' % (tag), image=summary)
    writer.add_summary(TF.Summary(value=[summary]), gen_iter)



def add_heatmap_summary(writer, word, sample, gen_iter, tag='plot'):
    n_repeat = 4
    sample = NP.clip(sample,0,None)
    PL.imshow(NP.repeat(sample, n_repeat, 1))
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
    sample_max = sample.max()
    sample_min = sample.min()
    rng = NP.max((sample_max - sample_min,1e-5))
    sample = ((sample - sample_min) * 2. / rng) - 1
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
samples = samples[:batch_size]
lengths = lengths[:batch_size]
cseq = cseq[:batch_size]
cseq_fixed = cseq_fixed[:batch_size]
clen_fixed = clen_fixed[:batch_size]
samples = dataset.invtransform(samples)
for i in range(batch_size):
    real_len = lengths[i]
    real_spect = samples[i, :,:real_len]
    add_heatmap_summary(d_train_writer, cseq[i], real_spect, 0, 'real_spect')
    real_sample = spect_to_audio(real_spect)
    add_waveform_summary(d_train_writer, cseq[i], real_sample, 0, 'real_waveform')
    add_audio_summary(d_train_writer, cseq[i], real_sample, real_len, 0, 'real_audio')
cseq_fixed = NP.array(cseq_fixed)
clen_fixed = NP.array(clen_fixed)
cseq_fixed, clen_fixed = tovar(cseq_fixed, clen_fixed)
cseq_fixed = cseq_fixed.long()
clen_fixed = clen_fixed.long()

gen_iter = 0
dis_iter = 0
epoch = 1
l = 10
baseline = None

def discriminate(d, data, length, embed, target, real):
    cls, rank, _ = d(data, length, embed)
    target = tovar(T.ones(*(cls.size())) * target)
    loss_c = binary_cross_entropy_with_logits(cls, target)
    loss_c = loss_c.mean()
    correct = ((cls.data > 0) if real else (cls.data < 0)).float()
    correct = correct.sum()
    acc = correct / data.size()[0]
    return cls, length, target, loss_c, rank, acc
init_data_loader = 1
if __name__ == '__main__':
    if modelnameload:
        if len(modelnameload) > 0:
            d = T.load('%s-dis-%05d' % (modelnameload, args.loaditerations))
            g = T.load('%s-gen-%05d' % (modelnameload, args.loaditerations))
            e_g = T.load('%s-eg-%05d' % (modelnameload, args.loaditerations))
            e_d = T.load('%s-ed-%05d' % (modelnameload, args.loaditerations))

    param_g = list(g.parameters()) + list(e_g.parameters())
    param_d = list(d.parameters()) + list(e_d.parameters())
    for p in param_g:
        p.requires_grad = True
    for p in param_d:
        p.requires_grad = True
    opt_g = T.optim.RMSprop(param_g, lr=args.glr)
    opt_d = T.optim.RMSprop(param_d, lr=args.dlr,weight_decay=1e-4)
    grad_nan = 0
    g_grad_nan = 0
    while True:
        _epoch = epoch

        for p in param_g:
            p.requires_grad = False
        for p in param_d:
            p.requires_grad = True
        for j in range(args.critic_iter):
            dis_iter += 1
            if dis_iter % 500 == 0:
                args.noisescale = args.noisescale * .9
            with Timer.new('load', print_=False):
                if init_data_loader or dis_iter % 100 == 0:
                    init_data_loader = 0
                    epoch, batch_id, _real_data_massive, \
                        _real_len_massive, _, _cs_massive, \
                        _cl_massive = dataloader.next()
                    num_samples_loaded = _real_data_massive.shape[0]
                random_indexes = NP.random.choice(
                        num_samples_loaded, batch_size, replace = True)
                _real_data = _real_data_massive[random_indexes]
                _real_len = _real_len_massive[random_indexes]
                _cs = _cs_massive[random_indexes]
                _cl = _cl_massive[random_indexes]
                
                
                
            with Timer.new('train_d', print_=False):
                noise = tovar(RNG.randn(*_real_data.shape) * args.noisescale)
                real_data = tovar(_real_data) + noise
                real_len = tovar(_real_len).long()
                cs = tovar(_cs).long()
                cl = tovar(_cl).long()

                embed_d = e_d(cs, cl)
                embed_g = e_g(cs, cl)

                cls_d, _, _, loss_d, rank_d, acc_d = \
                        discriminate(d, real_data, real_len, embed_d, 0.9, True)
                cls_d_x, _, _, _, rank_d_x, acc_d_x = discriminate(
                        d, real_data, real_len, T.cat([embed_d[-1:,:], embed_d[:-1,:]],0), 0.9, True)

                fake_data, fake_len, stop_raw = g(batch_size=batch_size, length=maxlen, c=embed_g)
                noise = tovar(T.randn(*fake_data.size()) * args.noisescale)
                fake_data = tovar((fake_data + noise).data)
                cls_g, _, _, loss_g, rank_g, acc_g = \
                        discriminate(d, fake_data, fake_len, embed_d, 0, False)

                loss_rank = ((1 - rank_d + rank_d_x).clamp(min=0)).mean()
                loss = loss_d + loss_g + loss_rank/10
                opt_d.zero_grad()
                loss.backward()
                if not check_grad(param_d):
                    grad_nan += 1
                    print 'Gradient exploded %d times', grad_nan
                    assert grad_nan <= 0
                    continue
                grad_nan = 0
                d_grad_norm = clip_grad(param_d, args.dgradclip)
                opt_d.step()

            loss_d, loss_g, loss_rank, loss, cls_d, cls_g, rank_d, rank_d_x = \
                    tonumpy(loss_d, loss_g, loss_rank, loss, cls_d, cls_g, rank_d, rank_d_x)
            d_train_writer.add_summary(
                    TF.Summary(
                        value=[
                            TF.Summary.Value(tag='loss_d', simple_value=loss_d),
                            TF.Summary.Value(tag='loss_g', simple_value=loss_g),
                            TF.Summary.Value(tag='loss_rank', simple_value=loss_rank),
                            TF.Summary.Value(tag='loss', simple_value=loss),
                            TF.Summary.Value(tag='cls_d/mean', simple_value=cls_d.mean()),
                            TF.Summary.Value(tag='cls_d/std', simple_value=cls_d.std()),
                            TF.Summary.Value(tag='cls_g/mean', simple_value=cls_g.mean()),
                            TF.Summary.Value(tag='cls_g/std', simple_value=cls_g.std()),
                            TF.Summary.Value(tag='rank_d/mean', simple_value=rank_d.mean()),
                            TF.Summary.Value(tag='rank_d/std', simple_value=rank_d.std()),
                            TF.Summary.Value(tag='rank_d_x/mean', simple_value=rank_d_x.mean()),
                            TF.Summary.Value(tag='rank_d_x/std', simple_value=rank_d_x.std()),
                            TF.Summary.Value(tag='acc_d', simple_value=acc_d),
                            TF.Summary.Value(tag='acc_g', simple_value=acc_g),
                            TF.Summary.Value(tag='d_grad_norm', simple_value=d_grad_norm),
                            ]
                        ),
                    dis_iter
                    )

            accs = [acc_d, acc_g]
            if dis_iter % 10 == 0:
                print 'D', epoch, dis_iter, loss, ';'.join('%.03f' % a for a in accs), Timer.get('load'), Timer.get('train_d')
                print 'lengths'
                print 'fake', list(fake_len.data)
                print 'real', list(real_len.data)
                print 'fake', tonumpy(fake_len).mean(), tonumpy(fake_len).std(), \
                    tonumpy(moment(fake_data.float(),1, fake_len)), tonumpy(moment(fake_data.float(),2, fake_len))
                print 'real', tonumpy(real_len).mean(), tonumpy(real_len).std(), \
                    tonumpy(moment(real_data.float(),1, real_len)), tonumpy(moment(real_data.float(),2, real_len))

            if acc_d > args.require_acc and acc_g > args.require_acc:
                break

        for p in param_g:
            p.requires_grad = True
        for p in param_d:
            p.requires_grad = False
        for _ in range(args.gencatchup):
            gen_iter += 1
            
            if args.gencatchup > 1:
                _, cs, cl, _, _ = dataset.pick_words(
                        batch_size, maxlen, dataset_h5, keys_train, maxcharlen_train, args, skip_samples=True)
                cs = tovar(cs).long()
                cl = tovar(cl).long()
            with Timer.new('train_g', print_=False):
                embed_g = e_g(cs, cl)
                embed_d = e_d(cs, cl)
                fake_data, fake_len, stop_raw = g(batch_size=batch_size, length=maxlen, c=embed_g)
                noise = tovar(T.randn(*fake_data.size()) * args.noisescale)
                fake_data += noise
                
                cls_g, rank_g, conv_acts_g = d(fake_data, fake_len, embed_d)
                _, rank_d, conv_acts_d = d(real_data, real_len, embed_d)
                if args.g_optim == 'boundary_seeking':
                    target = tovar(T.ones(*(cls_g.size())) * 0.5)   # TODO: add logZ estimate, may be unnecessary
                else:
                    target = tovar(T.zeros(*(cls_g.size())))            
                nframes_max = fake_len.data.max()
                _loss = binary_cross_entropy_with_logits(cls_g, target)
                _loss *= lambda_loss_g
                loss_fp_data = 0
                loss_fp_conv = 0
                for fake_act, real_act in zip(conv_acts_g, conv_acts_d):
                    for exp in [1,2,4]:
                        loss_fp_conv += ((moment_by_index(fake_act.float(),exp, fake_len) - 
                                      moment_by_index(real_act.float(),exp,real_len))**2).mean()
                for exp in [1,2,4,6]:
                    #loss_fp_data += T.abs(moment(fake_data.float(),exp, fake_len) - moment(real_data.float(),exp,real_len)) **1.5
                    #loss_fp_data += (T.abs(moment_by_index(fake_data.float(),exp, fake_len) - 
                    #                  moment_by_index(real_data.float(),exp,real_len))**1.5).mean()
                    loss_fp_data += (moment(fake_data.float(),exp, fake_len) - moment(real_data.float(),exp,real_len))**2
                    loss_fp_data += ((moment_by_index(fake_data.float(),exp, fake_len) - 
                                      moment_by_index(real_data.float(),exp,real_len))**2).mean()

                rank_g *= lambda_rank_g
                
                loss = _loss - rank_g
                
                reward = -loss.data# - loss_fp_len.data
                baseline = reward.mean() if baseline is None else baseline * 0.5 + reward.mean() * 0.5
                
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
                reward = (reward - baseline)
                average_reward = reward.abs().mean()
                reward = reward/average_reward
                
                reward_scatter.append(reward.cpu().numpy())
                length_scatter.append(fake_len.cpu().data.numpy())
                
                _loss = _loss.mean()
                _rank_g = -(rank_g).mean()
                stop_raw.reinforce(lambda_pg_g * reward.unsqueeze(1))
                # Debug the gradient norms
                opt_g.zero_grad()
                _loss.backward(retain_graph=True)
                loss_grad_dict = {p: p.grad.data.clone() for p in param_g if p.grad is not None}
                loss_grad_norm = sum(T.norm(p.grad.data) for p in param_g if p.grad is not None)
                opt_g.zero_grad()
                _rank_g.backward(T.Tensor([1]).cuda(), retain_graph=True)
                rank_grad_dict = {p: p.grad.data.clone() for p in param_g if p.grad is not None}
                rank_grad_norm = sum(T.norm(p.grad.data) for p in param_g if p.grad is not None)
                opt_g.zero_grad()
                loss_fp_data.backward(T.Tensor([lambda_fp_g]).cuda(), retain_graph=True)
                fp_grad_dict = {p: p.grad.data.clone() for p in param_g if p.grad is not None}
                fp_grad_norm = sum(T.norm(p.grad.data) for p in param_g if p.grad is not None)
                opt_g.zero_grad()
                loss_fp_conv.backward(T.Tensor([lambda_fp_conv]).cuda(), retain_graph=True)
                conv_fp_grad_dict = {p: p.grad.data.clone() for p in param_g if p.grad is not None}
                conv_fp_grad_norm = sum(T.norm(p.grad.data) for p in param_g if p.grad is not None)
                
                opt_g.zero_grad()
                T.autograd.backward(fake_len, [None for _ in fake_len])
                pg_grad_norm = sum(T.norm(p.grad.data) for p in param_g if p.grad is not None)
                # Do the real thing
                for p in param_g:
                    if p.grad is not None:
                        if p in loss_grad_dict:
                            p.grad.data += loss_grad_dict[p]
                        if p in rank_grad_dict:
                            p.grad.data += rank_grad_dict[p]
                        if p in fp_grad_dict:
                            p.grad.data += fp_grad_dict[p]
                        if p in conv_fp_grad_dict:
                            p.grad.data += conv_fp_grad_dict[p]
    
                
                if not check_grad(param_d):
                    grad_nan += 1
                    print 'Gradient exploded %d times', grad_nan
                    assert grad_nan <= 0
                    continue
                grad_nan = 0
                g_grad_norm = clip_grad(param_g, args.ggradclip)
                
                #Rank loss is smallest contributor because it is most unstable?
                if rank_grad_norm < .5:
                    lambda_rank_g *= 1.1
                if rank_grad_norm > .5:
                    lambda_rank_g /= 1.3
                if rank_grad_norm > 5:
                    lambda_rank_g /= 2.
                    
                if fp_grad_norm < 2:
                    lambda_fp_g *= 1.1
                if fp_grad_norm > 2:
                    lambda_fp_g /=1.3
                if fp_grad_norm > 20:
                    lambda_fp_g /=2.
                    
                if conv_fp_grad_norm < 2:
                    lambda_fp_conv *= 1.1
                if conv_fp_grad_norm > 2:
                    lambda_fp_conv /=1.3
                if conv_fp_grad_norm > 20:
                    lambda_fp_conv /=2.
                    
                if pg_grad_norm < 1:
                    lambda_pg_g *= 1.2
                if pg_grad_norm > 1:
                    lambda_pg_g /= 1.4
                if pg_grad_norm > 10:
                    lambda_pg_g /= 2.
                    
                if loss_grad_norm < 2:
                    lambda_loss_g *= 1.2
                if loss_grad_norm > 2:
                    lambda_loss_g /= 1.4
                if loss_grad_norm > 20:
                    lambda_loss_g /= 2.
                    
                if lambda_rank_g > args.lambda_rank:
                    lambda_rank_g = args.lambda_rank
                if lambda_fp_g > args.lambda_fp:
                    lambda_fp_g = args.lambda_fp
                if lambda_fp_conv > args.lambda_fp_conv:
                    lambda_fp_conv = args.lambda_fp_conv
                if lambda_pg_g > args.lambda_pg:
                    lambda_pg_g = args.lambda_pg
                if lambda_loss_g > args.lambda_loss:
                    lambda_loss_g = args.lambda_loss
                    
                d_train_writer.add_summary(
                        TF.Summary(
                            value=[
                                TF.Summary.Value(tag='g_grad_norm', simple_value=g_grad_norm),
                                TF.Summary.Value(tag='g_len_mean', simple_value=fake_len.cpu().data.numpy().mean()),
                                TF.Summary.Value(tag='g_loss_grad_norm', simple_value=loss_grad_norm),
                                TF.Summary.Value(tag='g_rank_grad_norm', simple_value=rank_grad_norm),
                                TF.Summary.Value(tag='g_fp_data_grad_norm', simple_value=fp_grad_norm),
                                TF.Summary.Value(tag='g_fp_conv_data_grad_norm', simple_value=conv_fp_grad_norm),
                                TF.Summary.Value(tag='g_pg_grad_norm', simple_value=pg_grad_norm),
                                ]
                            ),
                        gen_iter
                        )
                opt_g.step()
            
            if gen_iter % 10 == 0:
                add_scatterplot(d_train_writer,reward_scatter, length_scatter, gen_iter, 'scatterplot')
                reward_scatter = []
                length_scatter = []
                embed_g = e_g(cseq_fixed, clen_fixed)
                fake_data, fake_len, _ = g(z=z_fixed, c=embed_g)
                fake_data, fake_len = tonumpy(fake_data, fake_len)
                fake_data = dataset.invtransform(fake_data)
    
                for batch in range(batch_size):
                    fake_sample = fake_data[batch, :,:fake_len[batch]]
                    add_heatmap_summary(d_train_writer, cseq[batch], fake_sample, gen_iter, 'fake_spectogram')
                if gen_iter % 1000 == 0:
                    for batch in range(batch_size):
                        if fake_len[batch] > 3:
                            fake_spect = fake_data[batch, :,:fake_len[batch]]
                            fake_sample = spect_to_audio(fake_spect)
                            add_waveform_summary(d_train_writer, cseq[batch], fake_sample, gen_iter, 'fake_waveform')
                            add_audio_summary(d_train_writer, cseq[batch], fake_sample, fake_len[batch], gen_iter, 'fake_audio')
                    T.save(d, '%s-dis-%05d' % (modelnamesave, gen_iter + args.loaditerations))
                    T.save(g, '%s-gen-%05d' % (modelnamesave, gen_iter + args.loaditerations))
                    T.save(e_g, '%s-eg-%05d' % (modelnamesave, gen_iter + args.loaditerations))
                    T.save(e_d, '%s-ed-%05d' % (modelnamesave, gen_iter + args.loaditerations))
            print 'G', gen_iter, loss_grad_norm, rank_grad_norm, pg_grad_norm, tonumpy(_loss), Timer.get('train_g')
