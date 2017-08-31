
from __future__ import division

import sys
import tensorflow as TF
import itertools
import keras.backend as K
import keras.layers as KL
import keras.models as KM
import keras.activations as KA
import utiltf as util

from cells import Conv2DLSTMCell, ProjectedLSTMCell, FeedbackMultiLSTMCell
session = None

def start():
    global session
    session = TF.Session()
    return session

def run(*args, **kwargs):
    global session
    return session.run(*args, **kwargs)

class Model(object):
    count = 0
    def __init__(self, name=None, **kwargs):
        super(Model, self).__init__()

        self.name = name if name is not None else 'model-' + str(Model.count)
        Model.count += 1
        self.built = False

    @property
    def saver(self):
        if not hasattr(self, '_saver'):
            self._saver = TF.train.Saver(var_list=self.get_weights())
        return self._saver

    def save(self, path):
        global session
        self.saver.save(session, path)

    def load(self, path):
        global session
        self.saver.restore(session, path)

    def get_weights(self, **kwargs):
        return TF.get_collection(
                TF.GraphKeys.GLOBAL_VARIABLES,
                scope=self.name
                )

    def get_trainable_weights(self, **kwargs):
        return TF.get_collection(
                TF.GraphKeys.TRAINABLE_VARIABLES,
                scope=self.name
                )

    def __call__(self, *args, **kwargs):
        with TF.variable_scope(self.name, reuse=self.built):
            result = self.call(*args, **kwargs)
            self.built = True
        return result

    def call(self, *args, **kwargs):
        raise NotImplementedError

# Embedding models

class Embedder(Model):
    def embed(self, *args, **kwargs):
        return self(*args, **kwargs)

class CharRNNEmbedder(Embedder):
    def __init__(self,
                 embed_size=100,
                 n_possible_chars=256,
                 char_embed_size=None,
                 cell=TF.nn.rnn_cell.LSTMCell,
                 **kwargs):
        super(CharRNNEmbedder, self).__init__(**kwargs)
        self._embed_size = embed_size
        self._nchars = n_possible_chars
        self._cell = cell
        self._char_embed_size = char_embed_size or embed_size

    def call(self, chars, length):
        '''
        @chars: Tensor of int (batch_size, max_n_chars)
        @length: Tensor of int (batch_size,)
        '''
        embed_seq = TF.contrib.layers.embed_sequence(
                chars, self._nchars, self._char_embed_size)

        lstms_f = [self._cell(self._embed_size // 2)]
        lstms_b = [self._cell(self._embed_size // 2)]

        outputs, state_fw, state_bw = TF.contrib.rnn.stack_bidirectional_dynamic_rnn(
                lstms_f, lstms_b, embed_seq, dtype=TF.float32, sequence_length=length)
        h_fw = state_fw[0][1]
        h_bw = state_bw[0][1]

        h = TF.concat([h_fw, h_bw], axis=1)
        return h

# Generators

class Generator(Model):
    def generate(self, *args, **kwargs):
        return self(*args, **kwargs)


class RNNConvGenerator(Generator):
    # Sadly Keras' LSTM does not have output projection & feedback so I have
    # to use raw Tensorflow here.
    def __init__(self,
                 frame_size=200,        # # of amplitudes to generate at a time
                 noise_size=100,        # noise dimension at a time
                 state_size=200,
                 num_layers=1,
                 cell=FeedbackMultiLSTMCell,
                 **kwargs):
        super(RNNConvGenerator, self).__init__(**kwargs)

        self._frame_size = frame_size
        self._noise_size = noise_size
        self._state_size = state_size
        self._cell = cell
        self._num_layers = num_layers
        self.config = [[100, 3, 1], [100, 3, 1]]

    def call(self, batch_size=None, length=None, z=None, c=None, initial_h=None):
        frame_size = self._frame_size
        noise_size = self._noise_size
        state_size = self._state_size
        cell = self._cell

        lstm_f = cell(
                state_size, num_proj=frame_size, activation=TF.tanh,
                num_layers=self._num_layers)

        if z is None:
            nframes = util.div_roundup(length, frame_size)
            z = TF.random_normal((batch_size, nframes, noise_size))
        else:
            batch_size = TF.shape(z)[0]
            nframes = TF.shape(z)[1]

        if c is not None:
            c = TF.tile(TF.expand_dims(c, 1), (1, nframes, 1))
            z = TF.concat([z, c], axis=2)

        if initial_h is None:
            initial_h = lstm_f.random_state(batch_size, TF.float32)

        z_unstack = TF.unstack(TF.transpose(z, (1, 0, 2)))

        x = TF.nn.static_rnn(
                lstm_f, z_unstack, dtype=TF.float32,
                initial_state=initial_h,
                )[0]
        x = TF.concat(x, axis=1)
        x = TF.expand_dims(x, 1)
        x = TF.expand_dims(x, -1)
        x = TF.tanh(x)
        x = self.build_dense_conv(x)
        x = KL.Conv2DTranspose(
                filters=1, kernel_size=(1, 1), strides=(1, 1),
                padding='same')(x)
        x = TF.tanh(x)
        x = x[:, 0, :, 0]
        return x

    def build_dense_conv(self, x):
        for num_filters, filter_size, filter_stride in self.config:
            x_next = KL.Conv2DTranspose(
                    filters=num_filters, kernel_size=(1, filter_size),
                    strides=(1, filter_stride), padding='same')(x)
            x_next = TF.tanh(x_next)
            x = TF.concat([x, x_next], -1)
        return x


class RNNGenerator(Generator):
    # Sadly Keras' LSTM does not have output projection & feedback so I have
    # to use raw Tensorflow here.
    def __init__(self,
                 frame_size=200,        # # of amplitudes to generate at a time
                 noise_size=100,        # noise dimension at a time
                 state_size=200,
                 num_layers=1,
                 cell=FeedbackMultiLSTMCell,
                 **kwargs):
        super(RNNGenerator, self).__init__(**kwargs)

        self._frame_size = frame_size
        self._noise_size = noise_size
        self._state_size = state_size
        self._cell = cell
        self._num_layers = num_layers

    def _rnn(self, batch_size=None, length=None, z=None, c=None, initial_h=None):
        frame_size = self._frame_size
        noise_size = self._noise_size
        state_size = self._state_size
        cell = self._cell

        lstm_f = cell(
                state_size, num_proj=frame_size, activation=TF.tanh,
                num_layers=self._num_layers)

        if z is None:
            nframes = util.div_roundup(length, frame_size)
            z = TF.random_normal((batch_size, nframes, noise_size))
        else:
            batch_size = TF.shape(z)[0]
            nframes = TF.shape(z)[1]

        if c is not None:
            c = TF.tile(TF.expand_dims(c, 1), (1, nframes, 1))
            z = TF.concat([z, c], axis=2)

        if initial_h is None:
            initial_h = lstm_f.random_state(batch_size, TF.float32)

        z_unstack = TF.unstack(TF.transpose(z, (1, 0, 2)))

        x, h = TF.nn.static_rnn(
                lstm_f, z_unstack, dtype=TF.float32,
                initial_state=initial_h,
                )
        x = TF.concat(x, axis=1)

        return x, h

    def call(self, batch_size=None, length=None, z=None, c=None, initial_h=None):
        x, h = self._rnn(batch_size, length, z, c, initial_h)
        return x


class Conv1DGenerator(Generator):
    def __init__(self, config):
        super(Conv1DGenerator, self).__init__()

        self.config = config
        self.multiplier = 1
        for num_filters, filter_size, filter_stride in config:
            self.multiplier *= filter_stride

    def call(self, batch_size=None, length=None, z=None):
        if z is None:
            z = TF.random_normal((batch_size, length // self.multiplier))

        x1 = K.expand_dims(K.expand_dims(z, 1), 3)
        x1 = self.build_conv(x1)
        pooled = KL.Conv2DTranspose(
                filters=1, kernel_size=(1, 1), strides=(1, 1),
                padding='same')(x1)
        pooled = TF.tanh(pooled)
        pooled = pooled[:, 0, :, 0]

        return pooled

    def build_conv(self, x):
        for num_filters, filter_size, filter_stride in self.config:
            x = KL.Conv2DTranspose(
                    filters=num_filters, kernel_size=(1, filter_size),
                    strides=(1, filter_stride), padding='same')(x)
            x = KL.LeakyReLU()(x)

        return x


class Conv2DRNNGenerator(Generator):
    def __init__(self,
                 noise_size,
                 kernel_size,
                 filters,
                 cell=Conv2DLSTMCell):
        super(Conv2DRNNGenerator, self).__init__()

        self._noise_size = noise_size
        self._kernel_size = kernel_size
        self._filters = filters
        self._cell = cell

    def call(self, batch_size=None, length=None, z=None, initial_h=None):
        noise_size = self._noise_size
        filters = self._filters
        kernel_size = self._kernel_size
        cell = self._cell

        # As you can see Conv2DLSTM currently only supports inputs and states
        # (and outputs) with the same size.  You can specify pre_rnn_callback
        # and post_rnn_callback if you want to add some more fascinating
        # transformation.
        frame_size = noise_size
        num_frames = util.div_roundup(length, frame_size)

        def post_rnn_callback(x):
            return TF.tanh(TF.reduce_mean(x, axis=-1, keep_dims=True))

        lstm_f = cell(
                (1, frame_size), filters, (1, kernel_size),
                post_rnn_callback=post_rnn_callback,
                output_shape=(1, self._noise_size, 1),
                )

        if z is None:
            z = TF.random_normal((batch_size, num_frames, 1, noise_size, 1))

        if initial_h is None:
            _initial_h = lstm_f.zero_state(batch_size, TF.float32)
            initial_h = tuple(TF.random_normal(TF.shape(h)) for h in _initial_h)

        z_unstack = TF.unstack(TF.transpose(z, (1, 0, 2, 3, 4)))

        x = TF.nn.static_rnn(
                lstm_f, z_unstack, dtype=TF.float32,
                initial_state=initial_h,
                )[0]
        x = TF.concat(x, axis=2)[:, 0, :, 0]

        return x


class ResNetGenerator(Conv1DGenerator):
    def __init__(self, config, cardinality=1, **kwargs):
        super(ResNetGenerator, self).__init__(config, **kwargs)
        self.cardinality = cardinality

    def build_conv(self, x):
        last_num_filters = 0

        for num_filters, filter_size, filter_stride in self.config:
            if filter_stride == 1 and num_filters == last_num_filters:
                x = self._residual_block(
                        x, filter_size, num_filters // 2, num_filters)
            else:
                x = TF.contrib.layers.conv2d_transpose(
                        x,
                        num_filters, kernel_size=(1, filter_size),
                        stride=(1, filter_stride), padding='SAME',
                        activation_fn=None)
                x = KL.LeakyReLU()(x)

            last_num_filters = num_filters

        return x

    def _add_common_layers(self, y):
        y = TF.contrib.layers.batch_norm(
                y, is_training=K.learning_phase(), scale=True)
        y = KL.LeakyReLU()(y)
        return y

    def _grouped_convolution(self, y, kernel_size, nchannels, strides):
        if self.cardinality == 1:
            return TF.contrib.layers.conv2d(
                    y,
                    nchannels, kernel_size=kernel_size, stride=strides,
                    padding='SAME')

        assert not nchannels % cardinality
        _d = nchannels // cardinality

        groups = []
        for j in range(cardinality):
            group = y[:, :, :, j*_d:j*_d+_d]
            groups.append(
                    TF.contrib.layers.conv2d(
                        group,
                        _d, kernel_size=kernel_size, stride=strides,
                        padding='SAME')
                    )
        y = TF.concat(groups, axis=-1)
        return y

    def _residual_block(self,
                        y,
                        filter_size,
                        nchannels_in,
                        nchannels_out,
                        strides=(1, 1),
                        project_shortcut=False):
        shortcut = y

        y = TF.contrib.layers.conv2d(
                y,
                nchannels_in, kernel_size=(1, 1), stride=(1, 1),
                padding='SAME')
        y = self._add_common_layers(y)

        y = self._grouped_convolution(
            y, filter_size, nchannels_in, strides)
        y = self._add_common_layers(y)

        y = TF.contrib.layers.conv2d(
                y,
                nchannels_out, kernel_size=(1, 1), stride=(1, 1),
                padding='SAME')
        y = TF.contrib.layers.batch_norm(
                y, is_training=K.learning_phase(), scale=True)

        if project_shortcut or strides != (1, 1):
            shortcut = TF.contrib.layers.conv2d(
                    shortcut,
                    nchannels_out, kernel_size=(1, 1), stride=(1, 1),
                    padding='SAME')
            shortcut = TF.contrib.layers.batch_norm(
                    shortcut, is_training=K.learning_phase(), scale=True)

        y = shortcut + y
        y = KL.LeakyReLU()(y)

        return y


class Discriminator(Model):
    def discriminate(self, *args, **kwargs):
        return self(*args, **kwargs)

    def __init__(self,
                 metric='wasserstein',
                 constraint=None,
                 mode='unconditional',
                 **kwargs):
        # constraint:
        # - None,
        # - 'gp': Improved WGAN Training (Gulrajani et al., 2017)
        # - 'wc': Wasserstein GAN (Arjovsky et al., 2017)
        # - 'noise:xxx': Towards Principled Analysis of GAN (Arjovsky et al., 2016)
        super(Discriminator, self).__init__(**kwargs)
        self.metric = metric
        self.constraint = constraint
        self.mode = mode
        if constraint.startswith('noise:'):
            self.noise_scale = float(constraint[6:])
            self.noise = True
            self.gp = False
        elif constraint == 'gp':
            self.noise = False
            self.gp = True

    def grad_penalty(self, x_real, x_fake, c=None, **kwargs):
        eps = K.random_uniform([K.shape(x_real)[0], 1])
        x_inter = eps * x_real + (1 - eps) * x_fake
        d_inter, _ = self.discriminate(x_inter, c=c, **kwargs)

        grads = K.gradients(d_inter, x_inter)[0]
        grad_norms = K.sqrt(K.sum(K.square(grads), axis=1))
        penalty = K.square(grad_norms - 1)

        return penalty

    def compare(self,
                x_real,
                x_fake,
                c=None,
                c_wrong=None,   # for conditional_input mode (Reed et al., 2016)
                **kwargs):
        # mode: 'unconditional', 'conditional_input', 'aux_classifier'
        cond_input = self.mode == 'conditional_input'

        if self.noise:
            x_real = x_real + TF.random_normal(TF.shape(x_real)) * self.noise_scale
            x_fake = x_fake + TF.random_normal(TF.shape(x_fake)) * self.noise_scale

        d_real, d_real_pred = self.discriminate(
                x_real, c=c if cond_input else None)
        d_fake, d_fake_pred = self.discriminate(
                x_fake, c=c if cond_input else None)
        
        if hasattr(util, self.metric):
            loss_fn = getattr(util, self.metric)
        else:
            print('not an eligible loss function. Use Wasserstein or l2_loss')
        loss = loss_fn(d_real, d_fake)
        if cond_input:
            d_wrong, d_wrong_pred = self.discriminate(x_real, c=c_wrong)
            loss += loss_fn(d_real, d_wrong)

        if self.gp:
            penalty = self.grad_penalty(
                    x_real, x_fake, c=c if cond_input else None, **kwargs)
        else:
            penalty = K.constant(0)
        # XXX: do we need gradient penalty for real-audio/wrong-word?

        return K.mean(loss), d_real, d_fake, penalty, d_real_pred, d_fake_pred


class TimeDistributedCritic(Discriminator):
    def grad_penalty(self, x_real, x_fake, c=None, **kwargs):
        eps = K.random_uniform([K.shape(x_real)[0], 1])
        x_inter = eps * x_real + (1 - eps) * x_fake

        d_inter, _ = self.discriminate(x_inter, c=c, sum_=False, **kwargs)
        penalty = 0

        for i in range(self._nframes):
            grads = K.gradients(d_inter[:, i], x_inter)[0]
            grad_norms = K.sqrt(K.sum(K.square(grads), axis=1))
            penalty += K.square(grad_norms - 1)

        return penalty


class RNNDiscriminator(Discriminator):
    def __init__(self,
                 frame_size=200,
                 state_size=100,
                 length=8000,
                 num_layers=1,
                 cell=TF.nn.rnn_cell.LSTMCell,
                 **kwargs):
        super(RNNDiscriminator, self).__init__(**kwargs)

        self._frame_size = frame_size
        self._state_size = state_size
        self._cell = cell
        self._length = length
        self._num_layers = num_layers
        self._nframes = util.div_roundup(length, frame_size)

    def _rnn(self, x, c=None, **kwargs):
        batch_size = TF.shape(x)[0]
        nframes = self._nframes
        _x = TF.reshape(x, (batch_size, nframes, self._frame_size))
        if c is not None:
            c = TF.tile(TF.expand_dims(c, 1), (1, nframes, 1))
            _x = TF.concat([_x, c], axis=2)

        lstms_f = []
        lstms_b = []
        for _ in range(self._num_layers):
            lstms_f.append(self._cell(self._state_size))
            lstms_b.append(self._cell(self._state_size))

        x_unstack = TF.unstack(TF.transpose(_x, (1, 0, 2)))

        outputs, state_fw, state_bw = TF.contrib.rnn.stack_bidirectional_rnn(
                lstms_f, lstms_b, x_unstack, dtype=TF.float32)
        h = TF.concat(outputs, axis=1)
        h_fw = state_fw[0][1]
        h_bw = state_bw[0][1]
        return h, h_fw, h_bw

    def call(self, x, c=None, **kwargs):
        batch_size = TF.shape(x)[0]
        _, h_fw, h_bw = self._rnn(x, c=c, **kwargs)
        h = TF.concat([h_fw, h_bw], axis=1)
        h_size = 2 * self._state_size

        w = TF.get_variable('w', shape=(h_size,), dtype=TF.float32)
        b = TF.get_variable('b', shape=(), dtype=TF.float32)
        d = TF.matmul(TF.reshape(h, (batch_size, h_size)),
                      TF.reshape(w, (h_size, 1))) + b

        return d[:, 0], None


class Conv1DDiscriminator(Discriminator):
    def __init__(self, config, **kwargs):
        super(Conv1DDiscriminator, self).__init__(**kwargs)

        self.config = config

    def call(self, x, c=None, **kwargs):
        x1 = K.expand_dims(x, 2)
        for num_filters, filter_size, filter_stride in self.config:
            x1 = KL.Conv1D(
                    filters=num_filters, kernel_size=filter_size,
                    strides=filter_stride, padding='same')(x1)
            x1 = KL.LeakyReLU()(x1)

        pooled = KL.GlobalAvgPool1D()(x1)
        d = KL.Dense(1)(pooled)

        return d[:, 0], None


class RNNTimeDistributedDiscriminator(TimeDistributedCritic):
    # At every output from discriminator time step (concatenated from forward and backward),
    # project to a single scalar output value
    def __init__(self,
                 frame_size=200,
                 state_size=100,
                 length=8000,
                 num_layers=1,
                 cell=TF.nn.rnn_cell.LSTMCell,
                 **kwargs):
        super(RNNTimeDistributedDiscriminator, self).__init__(**kwargs)

        self._frame_size = frame_size
        self._state_size = state_size
        self._cell = cell
        self._length = length
        self._num_layers = num_layers
        self._nframes = util.div_roundup(length, frame_size)

    def call(self, x, c=None, sum_=True, **kwargs):
        batch_size = TF.shape(x)[0]
        nframes = self._nframes
        _x = TF.reshape(x, (batch_size, nframes, self._frame_size))
        if c is not None:
            c = TF.tile(TF.expand_dims(c, 1), (1, nframes, 1))
            _x = TF.concat([_x, c], axis=2)

        lstms_f = []
        lstms_b = []
        for _ in range(self._num_layers):
            lstms_f.append(self._cell(self._state_size))
            lstms_b.append(self._cell(self._state_size))

        x_unstack = TF.unstack(TF.transpose(_x, (1, 0, 2)))

        h = TF.contrib.rnn.stack_bidirectional_rnn(lstms_f, lstms_b, x_unstack, dtype = TF.float32)[0]
        h = TF.concat(h, axis=1)
        w = TF.get_variable('w', shape=(2*self._state_size,), dtype=TF.float32)
        b = TF.get_variable('b', shape=(), dtype=TF.float32)
        d = TF.matmul(TF.reshape(h, (batch_size * nframes, 2*self._state_size)),
                      TF.reshape(w, (2 * self._state_size, 1))) + b
        d = TF.reshape(d, (batch_size, nframes))
        
        return d, None


class ManyDiscriminator(Discriminator):
    def __init__(self, d_list, **kwargs):
        super(ManyDiscriminator, self).__init__(**kwargs)
        self.d_list = d_list
        self.num_d = len(d_list)
        
    def grad_penalty(self, x_real, x_fake, c=None, **kwargs):
        return sum([d.grad_penalty(x_real, x_fake, c, **kwargs) for d in self.d_list])

    def get_trainable_weights(self, **kwargs):
        all_listed = [TF.get_collection(
                TF.GraphKeys.TRAINABLE_VARIABLES,
                scope=d.name
                ) for d in self.d_list]
        return list(itertools.chain.from_iterable(all_listed))

    def get_weights(self, **kwargs):
        all_listed = [TF.get_collection(
                TF.GraphKeys.GLOBAL_VARIABLES,
                scope=d.name
                ) for d in self.d_list]
        return list(itertools.chain.from_iterable(all_listed))

    def call(self, x, c=None, **kwargs):
        d_list, d_pred = zip(*[d.discriminate(x, c, **kwargs) for d in self.d_list])

        return sum(d_list), None        # Aggregate d_pred here
    
    def compare(self,
                x_real,
                x_fake,
                c=None,
                mode='unconditional',
                **kwargs):
        num_d = self.num_d
        cond_input = mode == 'conditional_input'
        l1, dr1, df1, p1, drp1, dfp1 = zip(
                *[d.compare(x_real, x_fake, c, mode, **kwargs)
                    for d in self.d_list])
        l = sum(l1) / num_d
        dr = sum(dr1) / num_d
        df = sum(df1) / num_d
        p = sum(p1) / num_d
        if cond_input:
            drp = sum(drp1) / num_d
            dfp = sum(dfp1) / num_d
        else:
            drp = None
            dfp = None
        return l, dr, df, p, drp, dfp

class LocalDiscriminatorWrapper(Discriminator):
    '''
    Discriminates a chunk of audio.
    Usage:
    If you want to use the same discriminator for both local chunk and global:
    >>> d = RNNDiscriminator()
    >>> d_local = LocalDiscriminatorWrapper(d)
    >>> d_loss, _, _, _, _, _ = d_local.compare(x_real, x_fake)

    If you want a separate one, just do something like
    >>> d = LocalDiscrimiatorWrapper(RNNDiscriminator())
    '''
    def __init__(self,
                 discriminator,
                 frame_size=200,
                 length=None,
                 **kwargs):
        super(LocalDiscriminatorWrapper, self).__init__(**kwargs)

        self._d = discriminator
        self._frame_size = frame_size
        self._length = length or frame_size

    def call(self, x, c=None, **kwargs):
        x_crop = TF.random_crop(x, size=[TF.shape(x)[0], length])
        return self._d(x_crop, c=c, **kwargs)

    def compare(self, *args, **kwargs):
        return self._d.compare(*args, **kwargs)

    def grad_penalty(self, *args, **kwargs):
        return self._d.grad_penalty(*args, **kwargs)

# Policy Gradient models
def tanh_except_last(x):
    x_pre = x[..., :-1]
    x_last = x[..., -1:]
    return TF.concat([TF.tanh(x_pre), x_last], axis=-1)

class RNNDynamicGenerator(RNNGenerator):
    def _rnn(self, batch_size=None, length=None, z=None, c=None, initial_h=None):
        frame_size = self._frame_size
        noise_size = self._noise_size
        state_size = self._state_size
        cell = self._cell

        lstm_f = cell(
                state_size, num_proj=frame_size + 1, activation=tanh_except_last,
                num_layers=self._num_layers)

        if z is None:
            nframes = util.div_roundup(length, frame_size)
            z = TF.random_normal((batch_size, nframes, noise_size))
        else:
            batch_size = TF.shape(z)[0]
            nframes = TF.shape(z)[1]

        if c is not None:
            c = TF.tile(TF.expand_dims(c, 1), (1, nframes, 1))
            z = TF.concat([z, c], axis=2)

        if initial_h is None:
            initial_h = lstm_f.random_state(batch_size, TF.float32)

        z_unstack = TF.unstack(TF.transpose(z, (1, 0, 2)))

        x, h = TF.nn.static_rnn(
                lstm_f, z_unstack, dtype=TF.float32,
                initial_state=initial_h,
                )
        x = TF.stack(x, axis=1)
        s = x[..., -1]
        x = TF.reshape(x[..., :-1], (batch_size, -1))

        return x, s, h

    def call(self, batch_size=None, length=None, z=None, c=None, initial_h=None):
        frame_size = self._frame_size
        state_size = self._state_size
        if z is None:
            nframes = util.div_roundup(length, frame_size)
        else:
            batch_size = TF.shape(z)[0]
            nframes = TF.shape(z)[1]

        x, s, h = self._rnn(batch_size, length, z, c, initial_h)

        logp = TF.log_sigmoid(s)
        log_one_minus_p = util.log_one_minus_sigmoid(s)
        one_zero = TF.concat([TF.ones((batch_size, nframes - 1)), TF.zeros((batch_size, 1))], axis=1)
        logit = TF.cumsum(log_one_minus_p, axis=1, exclusive=True) + logp * one_zero

        a = TF.multinomial(logit, 1)[:, 0]
        gen_len = (a + 1) * frame_size

        return x, logit, a, gen_len


class RNNTimeDistributedDynamicDiscriminator(TimeDistributedCritic):
    def __init__(self,
                 frame_size=200,
                 state_size=100,
                 num_layers=1,
                 cell=TF.nn.rnn_cell.LSTMCell,
                 **kwargs):
        super(RNNTimeDistributedDynamicDiscriminator, self).__init__(**kwargs)

        self._frame_size = frame_size
        self._state_size = state_size
        self._cell = cell
        self._num_layers = num_layers

    def call(self, x, length, c=None, sum_=True, **kwargs):
        '''
        x: (batch_size, max_amplitude_samples)
        length: (batch_size,), # of effective samples
                the amplitudes after this number are discarded.
        '''
        nframes = util.div_roundup(length, self._frame_size)
        batch_size = TF.shape(x)[0]
        _x = TF.reshape(x, (batch_size, -1, self._frame_size))
        if c is not None:
            c = TF.tile(TF.expand_dims(c, 1), (1, nframes, 1))
            _x = TF.concat([_x, c], axis=2)

        lstms_f = []
        lstms_b = []
        for _ in range(self._num_layers):
            lstms_f.append(self._cell(self._state_size))
            lstms_b.append(self._cell(self._state_size))

        h, _, _ = TF.contrib.rnn.stack_bidirectional_dynamic_rnn(
                lstm_f,
                lstm_b,
                _x,
                sequence_length=nframes,
                )

        w = TF.get_variable('w', shape=(2*self._state_size,), dtype=TF.float32)
        b = TF.get_variable('b', shape=(), dtype=TF.float32)
        d = TF.matmul(TF.reshape(h, (batch_size * nframes, 2*self._state_size)),
                      TF.reshape(w, (2 * self._state_size, 1))) + b
        d = TF.reshape(d, (batch_size, nframes))
        
        return d, None

    def compare(self,
                x_real,
                x_fake,
                len_real,
                len_fake,
                c=None,
                c_wrong=None,
                **kwargs):
        # mode: 'unconditional', 'conditional_input', 'aux_classifier'
        cond_input = self.mode == 'conditional_input'

        if self.noise:
            x_real = x_real + TF.random_normal(TF.shape(x_real)) * self.noise_scale
            x_fake = x_fake + TF.random_normal(TF.shape(x_fake)) * self.noise_scale

        d_real, d_real_pred = self.discriminate(
                x_real, len_real, c=c if cond_input else None)
        d_fake, d_fake_pred = self.discriminate(
                x_fake, len_fake, c=c if cond_input else None)
        
        if hasattr(util, self.metric):
            loss_fn = getattr(util, self.metric)
        else:
            print('not an eligible loss function. Use Wasserstein or l2_loss')
        loss = loss_fn(d_real, d_fake)
        if cond_input:
            d_wrong, d_wrong_pred = self.discriminate(x_real, len_real, c=c_wrong)
            loss += loss_fn(d_real, d_wrong)

        # For now second-order gradient is not supported for dynamic RNN in Tensorflow
        penalty = K.constant(0)

        return K.mean(loss), d_real, d_fake, penalty, d_real_pred, d_fake_pred
