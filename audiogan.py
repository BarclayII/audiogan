
from torch.nn import Parameter
from functools import wraps

import torch as T
import gc
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

def adversarially_sample_z(batch_size, nframes, _noise_size, maxlen, embed_g, noisescale, embed_d, real_data, real_len, 
                                       g_optim, framesize, scale=1e-2):
    z = tovar(T.randn(batch_size, nframes, _noise_size))
    z.requires_grad = True
    fake_data, fake_s, fake_stop_list, fake_len = g(batch_size=batch_size, length=maxlen, c=embed_g, z = z)
    noise = tovar(T.randn(*fake_data.size()) * noisescale)
    fake_data += noise
    
    cls_g, hidden_states_g, hidden_states_length_g, nframes_g = d(fake_data, fake_len, embed_d)
    
    _, hidden_states_d, hidden_states_length_d, nframes_d = d(real_data, real_len, embed_d)
    dists_d = calc_dists(hidden_states_d, hidden_states_length_d)
    dists_g = calc_dists(hidden_states_g, hidden_states_length_g)
    dists_d += calc_dists([real_data.unsqueeze(1)], hidden_states_length_d)
    dists_g += calc_dists([fake_data.unsqueeze(1)], hidden_states_length_g)
    feature_penalty = 0
    #dists are (object, std) pairs.
    #penalizing z-scores of gen from real distribution
    #Note that the model could not do anything to r[1] by optimizing G.
    for r, f in zip(dists_d, dists_g):
        feature_penalty += T.pow((r[0] - f[0]), 2).mean() / batch_size

    if g_optim == 'boundary_seeking':
        target = tovar(T.ones(*(cls_g.size())) * 0.5)   # TODO: add logZ estimate, may be unnecessary
    else:
        target = tovar(T.zeros(*(cls_g.size())))            
    weight = length_mask(cls_g.size(), nframes_g)
    nframes_max = (fake_len / framesize).data.max()
    weight_r = length_mask((batch_size, nframes_max), fake_len / framesize)
    loss = binary_cross_entropy_with_logits_per_sample(cls_g, target, weight=weight) / nframes_g.float()


    
    # Check gradient w.r.t. generated output occasionally
    grad = T.autograd.grad(loss, z, grad_outputs=T.ones(loss.size()).cuda(), 
                           create_graph=True, retain_graph=True, only_inputs=True)[0]
    advers = (grad > 1e-9).type(T.FloatTensor) * scale - (grad < -1e-9).type(T.FloatTensor) * scale
    advers = advers.data

    z = tovar((z + tovar(advers)).data)
    return z
def adversarial_movement_d(data, data_len, embed_d, target, weight, d, scale = 1e-3):
    cls, _, _, nframes = d(data, data_len, embed_d)

    #feature_penalty = [T.pow(r - f,2).mean() for r, f in zip(dists_d, dists_g)]
    loss = binary_cross_entropy_with_logits_per_sample(cls, target, weight=weight) / nframes.float()
    # Check gradient w.r.t. generated output occasionally
    grad = T.autograd.grad(loss, data, grad_outputs=T.ones(loss.size()).cuda(), 
                           create_graph=True, retain_graph=True, only_inputs=True)[0]
                           
    advers = (grad > 0).type(T.FloatTensor) * scale - (grad < 0).type(T.FloatTensor) * scale
    advers = advers.data
    return advers


def adversarial_movement_g(data, embed_d, target, weight, d, scale = 1e-3):
    cls, _, _, nframes = d(data, data_len, embed_d)

    #feature_penalty = [T.pow(r - f,2).mean() for r, f in zip(dists_d, dists_g)]
    loss = binary_cross_entropy_with_logits_per_sample(cls, target, weight=weight) / nframes.float()
    # Check gradient w.r.t. generated output occasionally
    grad = T.autograd.grad(loss, data, grad_outputs=T.ones(loss.size()).cuda(), 
                           create_graph=True, retain_graph=True, only_inputs=True)[0]
                           
    advers = (grad > 0).type(T.FloatTensor) * scale - (grad < 0).type(T.FloatTensor) * scale
    advers = advers.data
    return advers


def tonumpy(*vars_):
    arrs = [v.data.cpu().numpy() for v in vars_]
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


class upsample(NN.Module):
    def __init__(self,scale_factor):
        NN.Module.__init__(self)
        self.upsample = NN.Upsample(scale_factor=scale_factor, mode='bilinear')

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.upsample(x)[:,:,:,0]
        return x
#


class Residual(NN.Module):
    def __init__(self,size):
        NN.Module.__init__(self)
        self.size = size
        self.linear = weight_norm(NN.Linear(size, size), ['weight','bias'])
        self.relu = NN.LeakyReLU()

    def forward(self, x):
        return self.relu(self.linear(x) + x)
#
class conv_res_bottleneck(NN.Module):
    def __init__(self,kernel,stride,infilters,hidden_filters):
        NN.Module.__init__(self)
        self.infilters = infilters
        self.conv = weight_norm(
            NN.Conv1d(infilters, hidden_filters, kernel_size = kernel, stride=stride, padding=(kernel - 1) // 2),
            ['weight','bias'])
        self.conv_h = weight_norm(
            NN.Conv1d(hidden_filters, hidden_filters, kernel_size = kernel, stride=1, padding=(kernel - 1) // 2),
            ['weight','bias'])
        self.deconv = weight_norm(
            NN.ConvTranspose1d(hidden_filters, infilters, kernel-1, stride, padding=stride//2),
            ['weight','bias'])
        self.relu = NN.LeakyReLU()
    def forward(self, x):
        res = x
        act = self.relu(self.conv(x))
        act = self.relu(self.conv_h(act))
        d = self.deconv(act)
        diff = d.size()[-1] - res.size()[-1]
        diff_half = int(diff/2.)
        remain = int(diff % 2)
        length = d.size()[-1]
        d = d[:,:,diff_half:-diff_half-remain + length]
        d += x
        return self.relu(d)
#
class conv_res_bottleneck_no_relu(NN.Module):
    def __init__(self,kernel,stride,infilters,hidden_filters):
        NN.Module.__init__(self)
        self.infilters = infilters
        self.conv = weight_norm(
            NN.Conv1d(infilters, hidden_filters, kernel_size = kernel, stride=stride, padding=(kernel - 1) // 2),
            ['weight','bias'])
        self.conv_h = weight_norm(
            NN.Conv1d(hidden_filters, hidden_filters, kernel_size = kernel, stride=1, padding=(kernel - 1) // 2),
            ['weight','bias'])
        self.deconv = weight_norm(
            NN.ConvTranspose1d(hidden_filters, infilters, kernel-1, stride, padding=stride-1),
            ['weight','bias'])
        self.relu = NN.LeakyReLU()
    def forward(self, x):
        res = x
        act = self.relu(self.conv(x))
        act = self.relu(self.conv_h(act))
        d = self.deconv(act)
        diff = d.size()[-1] - res.size()[-1]
        diff_half = int(diff/2.)
        remain = int(diff % 2)
        length = d.size()[-1]
        d = d[:,:,diff_half:-diff_half-remain + length]
        d += x
        return d
class deconv_layer(NN.Module):
    def __init__(self,infilters, outfilters, kernel, stride, relu=True):
        NN.Module.__init__(self)
        self.deconv = weight_norm(NN.ConvTranspose1d(infilters, outfilters, kernel-1, stride=stride, padding=stride-1),['weight','bias'])
        if relu:
            self.relu = NN.LeakyReLU()
        else:
            self.relu = False
    def forward(self, x):
        d = self.deconv(x)
        if self.relu:
            d = self.relu(d)
        return d
class deconv_unpool_residual(NN.Module):
    def __init__(self,infilters, outfilters, kernel, stride):
        NN.Module.__init__(self)
        self.deconv = weight_norm(NN.ConvTranspose1d(infilters, outfilters, kernel-1, stride=stride, padding=stride-1),['weight','bias'])
        self.unpool = NN.DataParallel(upsample(stride))
        self.relu = NN.LeakyReLU()
    def forward(self, x):
        d = self.deconv(x)
        res = self.unpool(x)
        diff = res.size()[-1] - d.size()[-1]
        diff_half = int(diff/2.)
        remain = int(diff % 2)
        length = res.size()[-1]
        res = res[:,:,diff_half:-diff_half-remain + length]
        return d + res
class dense_res_bottleneck(NN.Module):
    def __init__(self,kernel,stride,infilters,hidden_filters,outfilters):
        NN.Module.__init__(self)
        self.infilters = infilters
        self.outfilters = outfilters
        self.conv = weight_norm(
            NN.Conv1d(infilters, hidden_filters, kernel_size = kernel, stride=stride, padding=(kernel - 1) // 2),
            ['weight','bias'])
        self.deconv = weight_norm(
            NN.ConvTranspose1d(hidden_filters, outfilters, kernel-1, stride, padding=stride//2),
            ['weight','bias'])
        self.relu = NN.LeakyReLU()
    def forward(self, x):
        act = self.relu(self.conv(x))
        act = self.deconv(act)
        if self.infilters >= self.outfilters:
            act += x[:,-self.outfilters:,:]
        return self.relu(act)


class dense_res(NN.Module):
    def __init__(self,kernel,infilters,outfilters):
        NN.Module.__init__(self)
        self.infilters = infilters
        self.outfilters = outfilters
        self.conv = NN.Conv1d(infilters, outfilters, kernel_size = kernel, stride=1, padding=(kernel - 1) // 2)
        self.relu = NN.LeakyReLU()

    def forward(self, x):
        if self.infilters < self.outfilters:
            act = self.conv(x)
        else:
            act = self.conv(x) + x[:,-self.outfilters:,:]
        return act


class Embedder(NN.Module):
    def __init__(self,
                 output_size=100,
                 char_embed_size=50,
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
        return h[:, -2:].view(batch_size, output_size)

def fourth_moment(v):
    v_mean = v.mean(0)
    fourth_var = (((v -v_mean.unsqueeze(0))**4).sum(0))**(1/4)
    return fourth_var

def calc_dists(hidden_states, hidden_state_lengths):
    means_d = []
    stds_d = []
    fourth_d = []
    for h, l in zip(hidden_states, hidden_state_lengths):
        mask = length_mask((h.size()[0], h.size()[2]), l)
        m = (h*mask.unsqueeze(1).float()).sum(2) / l.unsqueeze(1).float()
        s = (((h - m.unsqueeze(2) * mask.unsqueeze(1).float()) ** 2).sum(2) ** (1./2.) ) / l.unsqueeze(1).float()
        f = (((h - m.unsqueeze(2) * mask.unsqueeze(1).float()) ** 4).sum(2) ** (1./4.) ) / l.unsqueeze(1).float()
        means_d.append((m.mean(0),m.std(0)))
        means_d.append((s.mean(0),s.std(0)))
        means_d.append((f.mean(0),f.std(0)))
        stds_d.append((m.std(0),m.std(0)))
        stds_d.append((s.std(0),s.std(0)))
        stds_d.append((f.std(0),f.std(0)))
        fourth_d.append((fourth_moment(m),m.std(0)))
        fourth_d.append((fourth_moment(s),s.std(0)))
        fourth_d.append((fourth_moment(f),f.std(0)))
    return means_d + stds_d + fourth_d

class Generator(NN.Module):
    def __init__(self,
                 frame_size=200,
                 embed_size=200,
                 noise_size=100,
                 state_size=1024,
                 num_layers=1,
                 cnn_struct = [[11, 5, 600],[3, 1, 600],[11, 5, 400],[3, 1, 400],
                               [5, 2, 300],[3, 1, 200],[5, 2, 200],[3, 1, 200],[5, 2, 100],[3, 1, 1]]#,['unpool', 5, 80]
                 ):
        NN.Module.__init__(self)
        self._frame_size = frame_size
        self._noise_size = noise_size
        self._state_size = state_size
        self._embed_size = embed_size
        self._num_layers = num_layers

        self.rnn = NN.ModuleList()
        self.rnn.append(
                NN.DataParallel(weight_norm(
                    NN.LSTMCell(frame_size + embed_size + noise_size, state_size), 
                    ['weight_ih', 'weight_hh', 'bias_hh', 'bias_ih'])))
        for _ in range(1, num_layers):
            self.rnn.append(
                    NN.DataParallel(weight_norm(
                        NN.LSTMCell(state_size, state_size),
                        ['weight_ih', 'weight_hh', 'bias_hh', 'bias_ih'])))
        self.deconv = NN.ModuleList()
        infilters = 200
        for idx, layer in enumerate(cnn_struct):
            kernel, stride, outfilters = layer[0],layer[1],layer[2]
            if idx < len(cnn_struct)-1:
                outfilters = outfilters * 10
                deconv = deconv_layer(infilters, outfilters, kernel, stride)
            else:
                deconv = deconv_layer(infilters, outfilters, kernel, stride, relu=False)
            self.deconv.append(NN.DataParallel(deconv))
            infilters = outfilters
            
        '''
        for idx, layer in enumerate(cnn_struct[:1]):
            kernel, stride, outfilters = layer[0],layer[1],layer[2]
            if kernel == 'unpool':
                pool = NN.DataParallel(upsample(stride))
                self.deconv.append(pool)
            else:
                deconv = deconv_unpool_residual(infilters, outfilters, kernel, stride)
                self.deconv.append(NN.DataParallel(deconv))
                res = conv_res_bottleneck(kernel=11,stride=4,infilters = outfilters,hidden_filters = outfilters * 4)
                self.deconv.append(NN.DataParallel(res))
                infilters = outfilters
        for idx, layer in enumerate(cnn_struct[1:]):
            kernel, stride, outfilters = layer[0],layer[1],layer[2]
            if kernel == 'unpool':
                pool = NN.DataParallel(upsample(stride))
                self.deconv.append(pool)
            else:
                res = conv_res_bottleneck(kernel=11,stride=4,infilters = infilters,hidden_filters = infilters * 4)
                self.deconv.append(NN.DataParallel(res))
                deconv = deconv_unpool_residual(infilters, outfilters, kernel, stride)
                self.deconv.append(NN.DataParallel(deconv))
                res = conv_res_bottleneck(kernel=11,stride=4,infilters = outfilters,hidden_filters = outfilters * 4)
                self.deconv.append(NN.DataParallel(res))
                infilters = outfilters
        
        kernel=11
        stride = 5
        last_deconv = weight_norm(
                NN.ConvTranspose1d(infilters, 1, kernel, stride=stride, padding=(kernel - 1) // 2),
                ['weight','bias'])
        self.deconv.append(NN.DataParallel(last_deconv))
        last_res = conv_res_bottleneck_no_relu(kernel=11,stride=4,infilters = 1,hidden_filters = 20)
        self.deconv.append(NN.DataParallel(last_res))
        '''
        self.proj = NN.DataParallel(weight_norm(NN.Linear(state_size, frame_size), ['weight', 'bias']))
        self.stopper = NN.DataParallel(weight_norm(NN.Linear(state_size, 1), ['weight', 'bias']))
        self.tanh = NN.Sigmoid()
        self.relu = NN.LeakyReLU()
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
            x_t = self.relu(self.proj(lstm_h[-1]))
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
        if 1:
            x = x.unsqueeze(1)
            #print('start', x.size())
            x = x.view(batch_size, -1, 200)
            x = x.permute(0,2,1)
            #x = x.permute(0,2,1)
            #print(x.size())
            for layer in self.deconv:
                x = layer(x)
                #print(x.size())
            #x = x.permute(0,2,1)
            x = x.squeeze(1)
        #print('end', x.size())
        return (self.tanh(x) * 2.4) -1.2, s, stop_list, tovar(length * frame_size)


class Discriminator(NN.Module):
    def __init__(self,
                 state_size=1024,
                 embed_size=200,
                 num_layers=1,
                 cnn_struct = [[11,5,64],[5,2,128],[5,2,256]]):
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

            conv = weight_norm(
                NN.Conv1d(infilters, outfilters, kernel, stride=stride, padding=(kernel - 1) // 2),
                ['weight','bias'])
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
        self.residual_net = NN.DataParallel(NN.Sequential(
                Residual(state_size),
                Residual(state_size),
            ))
        self.classifier = NN.DataParallel(NN.Sequential(
                weight_norm(NN.Linear(state_size, state_size // 2),['weight','bias']),
                NN.LeakyReLU(),
                weight_norm(NN.Linear(state_size // 2, 1),['weight','bias'])
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
            #cnn_output = cnn_output * length_mask((batch_size, cnn_output.size()[2]), nframes).unsqueeze(1)
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
        
        return classifier_out, cnn_outputs, cnn_output_lengths, nframes

parser = argparse.ArgumentParser()
parser.add_argument('--critic_iter', default=3, type=int)
parser.add_argument('--rnng_layers', type=int, default=1)
parser.add_argument('--rnnd_layers', type=int, default=1)
parser.add_argument('--framesize', type=int, default=200, help='# of amplitudes to generate at a time for RNN')
parser.add_argument('--noisesize', type=int, default=100, help='noise vector size')
parser.add_argument('--gstatesize', type=int, default=1024, help='RNN state size')
parser.add_argument('--dstatesize', type=int, default=1024, help='RNN state size')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--dgradclip', type=float, default=1)
parser.add_argument('--ggradclip', type=float, default=1)
parser.add_argument('--dlr', type=float, default=1e-4)
parser.add_argument('--glr', type=float, default=1e-3)
parser.add_argument('--modelname', type=str, default = '')
parser.add_argument('--modelnamesave', type=str, default='')
parser.add_argument('--modelnameload', type=str, default='')
parser.add_argument('--just_run', type=str, default='')
parser.add_argument('--loaditerations', type=int, default=0)
parser.add_argument('--gencatchup', type=int, default=1)
parser.add_argument('--logdir', type=str, default='.', help='log directory')
parser.add_argument('--dataset', type=str, default='dataset.h5')
parser.add_argument('--embedsize', type=int, default=100)
parser.add_argument('--minwordlen', type=int, default=1)
parser.add_argument('--maxlen', type=int, default=40000, help='maximum sample length (0 for unlimited)')
parser.add_argument('--noisescale', type=float, default=0.01)
parser.add_argument('--g_optim', default = 'boundary_seeking')
parser.add_argument('--require_acc', type=float, default=0.5)

args = parser.parse_args()
args.conditional = True
if args.just_run not in ['', 'gen', 'dis']:
    print('just run should be empty string, gen, or dis. Other values not accepted')
    sys.exit(0)
lambda_fp = .1
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
gencatchup = args.gencatchup
param_g = list(g.parameters()) + list(e_g.parameters())
param_d = list(d.parameters()) + list(e_d.parameters())

opt_g = T.optim.RMSprop(param_g, lr=args.glr)
opt_d = T.optim.RMSprop(param_d, lr=args.dlr)
if __name__ == '__main__':
    if modelnameload:
        if len(modelnameload) > 0:
            d = T.load('%s-dis-%05d' % (modelnameload, args.loaditerations))
            g = T.load('%s-gen-%05d' % (modelnameload, args.loaditerations))
            e_g = T.load('%s-eg-%05d' % (modelnameload, args.loaditerations))
            e_d = T.load('%s-ed-%05d' % (modelnameload, args.loaditerations))

    while True:
        _epoch = epoch

        for p in param_g:
            p.requires_grad = False
        for p in param_d:
            p.requires_grad = True
        for j in range(args.critic_iter):
            dis_iter += 1
            with Timer.new('load', print_=False):
                gc.collect()
                epoch, batch_id, real_data, real_len, _, cs, cl = dataloader.next()
                _, cs2, cl2, _, _ = dataset.pick_words(
                        batch_size, maxlen, dataset_h5, keys_train, maxcharlen_train, args, skip_samples=True)
                #last_real_raw = [real_data, real_len]
            with Timer.new('train_d', print_=False):
                cs = tovar(cs).long()
                cl = tovar(cl).long()
                embed_d = e_d(cs, cl)
                real_len = tovar(real_len).long()
                if dis_iter % 2 == 0:
                    noise = tovar(RNG.randn(*real_data.shape) * args.noisescale)
                    real_data = tovar(real_data) + noise
                    cls_d, hidden_states_d, hidden_states_length_d, nframes_d = d(real_data, real_len, embed_d)
                    target = tovar(T.ones(*(cls_d.size())) * 0.9)
                    weight = length_mask(cls_d.size(), nframes_d)
                else:
                    real_data = tovar(real_data)
                    real_data.requires_grad = True
                    cls_d, hidden_states_d, hidden_states_length_d, nframes_d = d(real_data, real_len, embed_d)
                    target = tovar(T.ones(*(cls_d.size())) * 0.9)
                    weight = length_mask(cls_d.size(), nframes_d)
                    advers = adversarial_movement_d(real_data, real_len, embed_d, target, weight, d)
                    real_data = tovar((real_data + tovar(advers)).data)

                #real_data.requires_grad = True
                loss_d = binary_cross_entropy_with_logits_per_sample(cls_d, target, weight=weight) / nframes_d.float()
                loss_d = loss_d.mean()
                correct_d = ((cls_d.data > 0).float() * weight.data).sum()
                num_d = weight.data.sum()

                cs2 = tovar(cs2).long()
                cl2 = tovar(cl2).long()
                embed_g = e_g(cs2, cl2)
                embed_d = e_d(cs2, cl2)
                fake_data, _, _, fake_len = g(batch_size=batch_size, length=maxlen, c=embed_g)
                if dis_iter % 2 == 0:
                    noise = tovar(T.randn(*fake_data.size()) * args.noisescale)
                    fake_data = tovar((fake_data + noise).data)
                else:
                    fake_data = tovar(fake_data.data)
                    fake_data.requires_grad = True
                    cls_g, _, _, nframes_g = d(fake_data, fake_len, embed_d)
                    target = tovar(T.zeros(*(cls_g.size())))
                    weight = length_mask(cls_g.size(), nframes_g)
                    advers = adversarial_movement_d(fake_data, fake_len, embed_d, target, weight, d)
                    fake_data = tovar((fake_data + tovar(advers)).data)
                fake_data.requires_grad = True
                cls_g, _, _, nframes_g = d(fake_data, fake_len, embed_d)
                target = tovar(T.zeros(*(cls_g.size())))
                weight = length_mask(cls_g.size(), nframes_g)

                #feature_penalty = [T.pow(r - f,2).mean() for r, f in zip(dists_d, dists_g)]
                loss_g = binary_cross_entropy_with_logits_per_sample(cls_g, target, weight=weight) / nframes_g.float()

                # Check gradient w.r.t. generated output occasionally
                grad = T.autograd.grad(loss_g, fake_data, grad_outputs=T.ones(loss_g.size()).cuda(), 
                                       create_graph=True, retain_graph=True, only_inputs=True)[0]
                                       
                #advers = (grad > 0).type(T.FloatTensor) *.001 - (grad < 0).type(T.FloatTensor) * .001
                norm = grad.norm(2, 1) ** 2
                norm = (norm / nframes_g.float()).data
                x_grad_norm = norm.mean()
                d_train_writer.add_summary(
                        TF.Summary(value=[TF.Summary.Value(tag='x_grad_norm', simple_value=x_grad_norm)]),
                        dis_iter)

                loss_g = loss_g.mean()
                correct_g = ((cls_g.data < 0).float() * weight.data).sum()
                num_g = weight.data.sum()
                loss = loss_d + loss_g
                opt_d.zero_grad()
                loss.backward()
                check_grad(param_d)
                d_grad_norm = clip_grad(param_d, args.dgradclip)
                opt_d.step()

            loss_d, loss_g, loss, cls_d, cls_g = tonumpy(loss_d, loss_g, loss, cls_d, cls_g)
            acc_d = correct_d / num_d
            acc_g = correct_g / num_g
            d_train_writer.add_summary(
                    TF.Summary(
                        value=[
                            TF.Summary.Value(tag='loss_d', simple_value=loss_d),
                            TF.Summary.Value(tag='loss_g', simple_value=loss_g),
                            TF.Summary.Value(tag='loss', simple_value=loss),
                            TF.Summary.Value(tag='cls_d/mean', simple_value=cls_d.mean()),
                            TF.Summary.Value(tag='cls_d/std', simple_value=cls_d.std()),
                            TF.Summary.Value(tag='cls_g/mean', simple_value=cls_g.mean()),
                            TF.Summary.Value(tag='cls_g/std', simple_value=cls_g.std()),
                            TF.Summary.Value(tag='acc_d', simple_value=acc_d),
                            TF.Summary.Value(tag='acc_g', simple_value=acc_g),
                            TF.Summary.Value(tag='d_grad_norm', simple_value=d_grad_norm),
                            ]
                        ),
                    dis_iter
                    )

            print 'D', epoch, dis_iter, loss, acc_d, acc_g, Timer.get('load'), Timer.get('train_d')
            if acc_d > .97 and acc_g > .97:
                gencatchup = 10#Right now, running generator twice crashes system.
            if acc_d > args.require_acc and acc_g > args.require_acc:
                break

        #real_data, real_len = last_real_raw[0], last_real_raw[1]
        for _ in range(gencatchup):
            for p in param_g:
                p.requires_grad = True
            for p in param_d:
                p.requires_grad = False
            gen_iter += 1
            _, _, real_data, real_len, _, _, _ = dataloader.next()
            noise = tovar(RNG.randn(*real_data.shape) * args.noisescale)
            real_data = tovar(real_data) + noise
            real_len = tovar(real_len).long()
            _, cs, cl, _, _ = dataset.pick_words(
                    batch_size, maxlen, dataset_h5, keys_train, maxcharlen_train, args, skip_samples=True)
            with Timer.new('train_g', print_=False):
                cs = tovar(cs).long()
                cl = tovar(cl).long()
                embed_g = e_g(cs, cl)
                embed_d = e_d(cs, cl)
                nframes = div_roundup(maxlen, g._frame_size)
                
                z = adversarially_sample_z(batch_size, nframes, g._noise_size, maxlen, embed_g, args.noisescale, embed_d, real_data, real_len, 
                                           args.g_optim, args.framesize, scale=1e-2)
    
    
    
                fake_data, fake_s, fake_stop_list, fake_len = g(batch_size=batch_size, length=maxlen, c=embed_g, z = z)
                noise = tovar(T.randn(*fake_data.size()) * args.noisescale)
                fake_data += noise
                
                cls_g, hidden_states_g, hidden_states_length_g, nframes_g = d(fake_data, fake_len, embed_d)
                
                _, hidden_states_d, hidden_states_length_d, nframes_d = d(real_data, real_len, embed_d)
                dists_d = calc_dists(hidden_states_d, hidden_states_length_d)
                dists_g = calc_dists(hidden_states_g, hidden_states_length_g)
                dists_d += calc_dists([real_data.unsqueeze(1)], hidden_states_length_d)
                dists_g += calc_dists([fake_data.unsqueeze(1)], hidden_states_length_g)
                feature_penalty = 0
                #dists are (object, std) pairs.
                #penalizing z-scores of gen from real distribution
                #Note that the model could not do anything to r[1] by optimizing G.
                for r, f in zip(dists_d, dists_g):
                    feature_penalty += T.pow((r[0] - f[0]), 2).mean() / batch_size
                
                if args.g_optim == 'boundary_seeking':
                    target = tovar(T.ones(*(cls_g.size())) * 0.5)   # TODO: add logZ estimate, may be unnecessary
                else:
                    target = tovar(T.zeros(*(cls_g.size())))            
                weight = length_mask(cls_g.size(), nframes_g)
                nframes_max = (fake_len / args.framesize).data.max()
                weight_r = length_mask((batch_size, nframes_max), fake_len / args.framesize)
                loss = binary_cross_entropy_with_logits_per_sample(cls_g, target, weight=weight) / nframes_g.float()
    
    
    
    
    
    
    
                
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
    
                
                _loss = loss.mean()
                opt_g.zero_grad()
                _loss.backward(retain_graph=True)
                loss_grads = {p: p.grad.data.clone() for p in param_g if p.grad is not None}
                loss_norm = sum(T.norm(loss_grads[p]) for p in loss_grads)
                opt_g.zero_grad()
                feature_penalty.backward(T.Tensor([lambda_fp]).cuda(), retain_graph=True)
                fp_grads = {p: p.grad.data.clone() for p in param_g if p.grad is not None}
                fp_norm = sum(T.norm(fp_grads[p]) for p in fp_grads)
                #loss = _loss + feature_penalty * lambda_fp
                #loss = _loss
                opt_g.zero_grad()
                for i, fake_stop in enumerate(fake_stop_list):
                    fake_stop.reinforce(0.1 * reward[:, i:i+1])
                #opt_g.zero_grad()
                #loss.backward(retain_graph=True)
                for p in param_g:
                    p.requires_grad = False
                for p in g.stopper.parameters():
                    p.requires_grad = True
                T.autograd.backward(fake_stop_list, [None for _ in fake_stop_list])
                pg_grads = {p: p.grad.data.clone() for p in param_g if p.grad is not None}
                pg_norm = sum(T.norm(pg_grads[p]) for p in pg_grads)
                for p in param_g:
                    if p in loss_grads:
                        p.grad.data += loss_grads[p]
                    if p in fp_grads:
                        p.grad.data += fp_grads[p]
                check_grad(param_g)
                g_grad_norm = clip_grad(param_g, args.ggradclip)
                d_train_writer.add_summary(
                        TF.Summary(
                            value=[
                                TF.Summary.Value(tag='g_grad_norm', simple_value=g_grad_norm),
                                TF.Summary.Value(tag='feature_penalty', simple_value=tonumpy(feature_penalty)[0]),
                                TF.Summary.Value(tag='lambda_fp', simple_value=lambda_fp),
                                TF.Summary.Value(tag='g_loss_grad_norm', simple_value=loss_norm),
                                TF.Summary.Value(tag='g_fp_grad_norm', simple_value=fp_norm),
                                TF.Summary.Value(tag='g_pg_grad_norm', simple_value=pg_norm),
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
            print 'G', gen_iter, tonumpy(_loss), tonumpy(feature_penalty), lambda_fp, Timer.get('train_g')
        gencatchup = 1