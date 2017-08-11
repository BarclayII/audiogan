
from __future__ import division

import tensorflow as TF

import keras.backend as K
import keras.layers as KL
import keras.models as KM
import keras.activations as KA

from cells import Conv2DLSTMCell, ProjectedLSTMCell

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
    def __init__(self, name=None):
        super(Model, self).__init__()

        self.name = name if name is not None else 'model-' + str(Model.count)
        Model.count += 1
        self.built = False

    @property
    def saver(self):
        if not hasattr(self, _saver):
            self._saver = TF.train.Saver()
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

    def __call__(self, **kwargs):
        with TF.variable_scope(self.name, reuse=self.built):
            result = self.call(**kwargs)
            self.built = True
        return result

    def call(self, **kwargs):
        raise NotImplementedError

# Joint Embedding Mixers

class Mixer(Model):
    def mix(self, **kwargs):
        return self(**kwargs)


class MultiplyMixer(Mixer):
    # Reference:
    # https://github.com/shaform/DeepNetworks/blob/master/deep_networks/models/iwacgan.py
    def call(self, z, c):
        return z * c


class ConcatMixer(Mixer):
    # Reference:
    # https://github.com/fairytale0011/Conditional-WassersteinGAN/blob/master/WGAN_AC.py
    # Generative Adversarial Text to Image Synthesis (Reed et al, 2016)
    def call(self, z, c):
        return TF.concat([z, c], axis=1)

# Generators

class Generator(Model):
    def generate(self, **kwargs):
        return self(**kwargs)


class RNNGenerator(Generator):
    # Sadly Keras' LSTM does not have output projection & feedback so I have
    # to use raw Tensorflow here.
    def __init__(self,
                 frame_size=200,        # # of amplitudes to generate at a time
                 noise_size=100,        # noise dimension at a time
                 state_size=100,
                 bidirectional=False,
                 cell=ProjectedLSTMCell,
                 **kwargs):
        super(RNNGenerator, self).__init__(**kwargs)

        self._frame_size = frame_size
        self._noise_size = noise_size
        self._state_size = state_size
        self._cell = cell
        self._bidirectional = bidirectional

    def call(self, batch_size=None, length=None, z=None, hf=None, hb=None):
        frame_size = self._frame_size
        noise_size = self._noise_size
        state_size = self._state_size
        cell = self._cell

        lstm_f = cell(state_size, num_proj=frame_size, activation=TF.tanh)
        lstm_b = cell(state_size, num_proj=frame_size, activation=TF.tanh)

        if z is None and hf is None and hb is None:
            nframes = length // frame_size
            z = TF.random_normal((batch_size, nframes, noise_size))

            _hf = lstm_f.zero_state(batch_size, TF.float32)
            _hb = lstm_b.zero_state(batch_size, TF.float32)
            hf = tuple(TF.random_normal(TF.shape(h)) for h in _hf)
            hb = tuple(TF.random_normal(TF.shape(h)) for h in _hb)

        z_unstack = TF.unstack(TF.transpose(z, (1, 0, 2)))

        if self._bidirectional:
            x = TF.nn.static_bidirectional_rnn(
                    lstm_f, lstm_b, z_unstack, dtype=TF.float32,
                    initial_state_fw=hf, initial_state_bw=hb,
                    )
        else:
            x = TF.nn.static_rnn(
                    lstm_f, z_unstack, dtype=TF.float32,
                    initial_state=hf,
                    )

        return TF.concat(x[0], axis=1)


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
                 bidirectional=False,
                 cell=Conv2DLSTMCell):
        super(Conv2DRNNGenerator, self).__init__()

        self._noise_size = noise_size
        self._kernel_size = kernel_size
        self._filters = filters
        self._cell = cell
        self._bidirectional = bidirectional

    def call(self, batch_size=None, length=None, z=None, hf=None, hb=None):
        noise_size = self._noise_size
        filters = self._filters
        kernel_size = self._kernel_size
        cell = self._cell

        # As you can see Conv2DLSTM currently only supports inputs and states
        # (and outputs) with the same size.  You can specify pre_rnn_callback
        # and post_rnn_callback if you want to add some more fascinating
        # transformation.
        frame_size = noise_size
        num_frames = length // frame_size

        def post_rnn_callback(x):
            return TF.tanh(TF.reduce_mean(x, axis=-1, keep_dims=True))

        lstm_f = cell(
                (1, frame_size), filters, (1, kernel_size),
                post_rnn_callback=post_rnn_callback,
                output_shape=(1, self._noise_size, 1),
                )
        lstm_b = cell(
                (1, frame_size), filters, (1, kernel_size),
                post_rnn_callback=post_rnn_callback,
                output_shape=(1, self._noise_size, 1),
                )

        if z is None and hf is None and hb is None:
            z = TF.random_normal((batch_size, num_frames, 1, noise_size, 1))

            _hf = lstm_f.zero_state(batch_size, TF.float32)
            _hb = lstm_b.zero_state(batch_size, TF.float32)
            hf = tuple(TF.random_normal(TF.shape(h)) for h in _hf)
            hb = tuple(TF.random_normal(TF.shape(h)) for h in _hb)

        z_unstack = TF.unstack(TF.transpose(z, (1, 0, 2, 3, 4)))

        if self._bidirectional:
            x = TF.nn.static_bidirectional_rnn(
                    lstm_f, lstm_b, z_unstack, dtype=TF.float32,
                    initial_state_fw=hf, initial_state_bw=hb,
                    )
        else:
            x = TF.nn.static_rnn(
                    lstm_f, z_unstack, dtype=TF.float32,
                    initial_state=hf,
                    )

        x = TF.concat(x[0], axis=2)[:, 0, :, 0]
        return x


class ResNetGenerator(Conv1DGenerator):
    def __init__(self, config, cardinality=1, **kwargs):
        super(ResNetGenerator, self).__init__(config, **kwargs)
        self.cardinality = cardinality

    def build_conv(self, x):
        last_num_filters = 0

        for num_filters, filter_size, filter_stride in config:
            if filter_stride == 1 and num_filters == last_num_filters:
                x = self._residual_block(
                        x, filter_size, num_filters // 2, num_filters)
            else:
                x = KL.Conv2DTranspose(
                        filters=num_filters, kernel_size=(1, filter_size),
                        strides=(1, filter_stride), padding='same')(x)
                x = KL.LeakyReLU()(x)

            last_num_filters = num_filters

        return x

    def _add_common_layers(self, y):
        y = AutoUpdate(KL.BatchNormalization())(y)
        y = KL.LeakyReLU()(y)
        return y

    def _grouped_convolution(self, y, kernel_size, nchannels, strides):
        if self.cardinality == 1:
            return KL.Conv2D(
                    nchannels, kernel_size=kernel_size, strides=strides,
                    padding='same')(y)

        assert not nchannels % cardinality
        _d = nchannels // cardinality

        groups = []
        for j in range(cardinality):
            group = KL.Lambda(lambda z: z[:, :, :, j*_d:j*_d+_d])(y)
            groups.append(
                    KL.Conv2D(
                        _d, kernel_size=kernel_size, strides=strides,
                        padding='same')(group)
                    )
        y = KL.concatenate(groups)
        return y

    def _residual_block(self,
                        y,
                        filter_size,
                        nchannels_in,
                        nchannels_out,
                        strides=(1, 1),
                        project_shortcut=False):
        shortcut = y

        y = KL.Conv2D(
                nchannels_in, kernel_size=(1, 1), strides=(1, 1),
                padding='same')(y)
        y = self._add_common_layers(y)

        y = self._grouped_convolution(
            y, filter_size, nchannels_in, strides)
        y = self._add_common_layers(y)

        y = KL.Conv2D(
                nchannels_out, kernel_size=(1, 1), strides=(1, 1),
                padding='same')(y)
        y = AutoUpdate(KL.BatchNormalization())(y)

        if project_shortcut or strides != (1, 1):
            shortcut = KL.Conv2D(
                    nchannels_out, kernel_size=(1, 1), strides=(1, 1),
                    padding='same')(shortcut)
            shortcut = AutoUpdate(KL.BatchNormalization())(shortcut)

        y = KL.add([shortcut, y])
        y = KL.LeakyReLU()(y)

        return y


class Discriminator(Model):
    def discriminate(self, x, c=None, **kwargs):
        return self(x, c=c, **kwargs)


class WGANCritic(Discriminator):
    def __init__(self, **kwargs):
        super(WGANCritic, self).__init__(**kwargs)

    def grad_penalty(self, x_real, x_fake, c=None, **kwargs):
        eps = K.random_uniform([K.shape(x_real)[0], 1])
        x_inter = eps * x_real + (1 - eps) * x_fake
        d_inter, _ = self.discriminate(x_inter, **kwargs)

        grads = K.gradients(d_inter, x_inter)[0]
        grad_norms = K.sqrt(K.sum(K.square(grads), axis=1))
        penalty = K.square(grad_norms - 1)

        return penalty

    def compare(self,
                x_real,
                x_fake,
                grad_penalty=True,
                lambda_=10,
                c=None,
                mode='unconditional'):
        # mode: 'unconditional', 'conditional_input', 'auxiliary_prediction'
        uncond = mode == 'unconditional'
        cond_input = mode == 'conditional_input'
        aux_pred = mode == 'auxiliary_prediction'

        d_real, d_real_pred = self.discriminate(
                x_real, c=c if cond_input else None)
        d_fake, d_fake_pred = self.discriminate(
                x_fake, c=c if cond_input else None)
        loss = d_fake - d_real
        penalty = (
                self.grad_penalty(
                    x_real, x_fake, c=c if cond_input else None, **kwargs)
                if grad_penalty else K.constant(0)
                )
        loss += lambda_ * penalty

        return K.mean(loss), d_real, d_fake, penalty, d_real_pred, d_fake_pred


class RNNDiscriminator(WGANCritic):
    def __init__(self,
                 frame_size=200,
                 state_size=100,
                 length=8000,
                 **kwargs):
        super(RNNDiscriminator, self).__init__(**kwargs)

        self._frame_size = frame_size
        self._state_size = state_size
        self._num_frames = num_frames = length // frame_size

    def call(self, x, c=None, **kwargs):
        _x = TF.reshape(x, (-1, self._num_frames, self._frame_size))
        if c is not None:
            c = TF.tile(TF.expand_dims(c, 1), (1, num_frames, 1))
            _x = TF.concatenate([_x, c], axis=2)

        _lstm = KL.Bidirectional(
                KL.LSTM(
                    units=state_size, activation='tanh',
                    return_sequences=False, unroll=True)
                )(_x)
        _d = KL.Dense(1)(_lstm)

        return _d[:, 0], None


class Conv1DDiscriminator(WGANCritic):
    def __init__(self, config, **kwargs):
        super(Conv1DDiscriminator, self).__init__(**kwargs)

        self.config = config

    def call(self, x, c=None, **kwargs):
        x1 = K.expand_dims(x, 2)
        for num_filters, filter_size, filter_stride in config:
            x1 = KL.Conv1D(
                    filters=num_filters, kernel_size=filter_size,
                    strides=flter_stride, padding='same')(x1)
            x1 = KL.LeakyReLU()(_x1)

        pooled = KL.GlobalAvgPool1D()(x1)
        d = KL.Dense(1)(pooled)

        return d[:, 0]
