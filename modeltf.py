
from __future__ import division

import tensorflow as TF

import keras.backend as K
import keras.layers as KL
import keras.models as KM
import keras.activations as KA

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

# Joint Embedding Mixers

class Mixer(Model):
    def mix(self, *args, **kwargs):
        return self(*args, **kwargs)


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
    def generate(self, *args, **kwargs):
        return self(*args, **kwargs)


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

    def call(self, batch_size=None, length=None, z=None, initial_h=None):
        frame_size = self._frame_size
        noise_size = self._noise_size
        state_size = self._state_size
        cell = self._cell

        lstm_f = cell(
                state_size, num_proj=frame_size, activation=TF.tanh,
                num_layers=self._num_layers)

        if z is None:
            nframes = length // frame_size
            z = TF.random_normal((batch_size, nframes, noise_size))
        else:
            batch_size = TF.shape(z)[0]

        if initial_h is None:
            initial_h = lstm_f.random_state(batch_size, TF.float32)

        z_unstack = TF.unstack(TF.transpose(z, (1, 0, 2)))

        x = TF.nn.static_rnn(
                lstm_f, z_unstack, dtype=TF.float32,
                initial_state=initial_h,
                )[0]
        x = TF.concat(x, axis=1)

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
        num_frames = length // frame_size

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


class WGANCritic(Discriminator):
    def __init__(self, **kwargs):
        super(WGANCritic, self).__init__(**kwargs)

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
                grad_penalty=True,
                lambda_=10,
                c=None,
                mode='unconditional',
                **kwargs):
        # mode: 'unconditional', 'conditional_input'
        cond_input = mode == 'conditional_input'

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
                 num_layers=1,
                 cell=TF.nn.rnn_cell.LSTMCell,
                 **kwargs):
        super(RNNDiscriminator, self).__init__(**kwargs)

        self._frame_size = frame_size
        self._state_size = state_size
        self._cell = cell
        self._length = length
        self._num_layers = num_layers
        self._nframes = length // frame_size

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


class Conv1DDiscriminator(WGANCritic):
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


class RNNTimeDistributedDiscriminator(RNNDiscriminator):
    def __init__(self,
                 frame_size=200,
                 state_size=100,
                 length=8000,
                 num_layers=1,
                 cell=TF.nn.rnn_cell.LSTMCell,
                 approx=True,
                 **kwargs):
        super(RNNTimeDistributedDiscriminator, self).__init__(
                frame_size=frame_size, state_size=state_size,
                length=length, num_layers=num_layers, cell=cell,
                **kwargs)
        self._approx = approx

    def call(self, x, c=None, sum_=True, **kwargs):
        batch_size = TF.shape(x)[0]
        nframes = self._nframes
        h, _, _ = self._rnn(x, c=c, **kwargs)
        h_size = 2 * self._state_size

        w = TF.get_variable('w', shape=(h_size,), dtype=TF.float32)
        b = TF.get_variable('b', shape=(), dtype=TF.float32)
        d = TF.matmul(TF.reshape(h, (batch_size * nframes, h_size)),
                      TF.reshape(w, (h_size, 1))) + b
        d = TF.reshape(d, (batch_size, nframes))

        return d if not sum_ else TF.reduce_sum(d, axis=1), None

    def grad_penalty(self, x_real, x_fake, c=None, **kwargs):
        eps = K.random_uniform([K.shape(x_real)[0], 1])
        x_inter = eps * x_real + (1 - eps) * x_fake

        if self._approx:
            d_inter, _ = self.discriminate(x_inter, c=c, **kwargs)

            grads = K.gradients(d_inter, x_inter)[0]
            grad_norms = K.sqrt(K.sum(K.square(grads), axis=1))
            penalty = K.square(grad_norms - self._nframes)
        else:
            d_inter, _ = self.discriminate(x_inter, c=c, sum_=False, **kwargs)
            penalty = 0

            for i in range(self._nframes):
                grads = K.gradients(d_inter[:, i], x_inter)[0]
                grad_norms = K.sqrt(K.sum(K.square(grads), axis=1))
                penalty += K.square(grad_norms - 1)

        return penalty


class DualDiscriminator(object):
    
    def __init__(self, rnn, cnn, **kwargs):
        super(DualDiscriminator, self).__init__(**kwargs)
        self.rnn = rnn
        self.cnn = cnn
        
    def grad_penalty(self, x_real, x_fake, c=None, **kwargs):
        p1 = self.rnn.grad_penalty(x_real, x_fake, c, **kwargs)
        p2 = self.cnn.grad_penalty(x_real, x_fake, c, **kwargs)
        penalty = p1 + p2
        return penalty

    def get_trainable_weights(self, **kwargs):
        return TF.get_collection(
                TF.GraphKeys.TRAINABLE_VARIABLES,
                scope=self.rnn.name
                ) + TF.get_collection(
                TF.GraphKeys.TRAINABLE_VARIABLES,
                scope=self.cnn.name
                )
        
    def compare(self,
                x_real,
                x_fake,
                grad_penalty=True,
                lambda_=10,
                c=None,
                mode='unconditional',
                **kwargs):
        
        cond_input = mode == 'conditional_input'
        l1, dr1, df1, p1, drp1, dfp1 = self.rnn.compare(x_real, x_fake, grad_penalty, lambda_, c, mode, **kwargs)
        l2, dr2, df2, p2, drp2, dfp2 = self.cnn.compare(x_real, x_fake, grad_penalty, lambda_, c, mode, **kwargs)
        
        return (l1+l2)/2, (dr1+dr2)/2, (df1+df2)/2, (p1+p2)/2, \
            (drp1+drp2)/2 if cond_input else None, (dfp1+dfp2)/2 if cond_input else None

class LocalDiscriminatorWrapper(WGANCritic):
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
