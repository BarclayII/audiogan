
from __future__ import division

import tensorflow as TF

import keras.backend as K
import keras.layers as KL
import keras.models as KM
import keras.activations as KA

import utiltf as util

class RNNGenerator(util.Component):
    def __init__(self, frame_size=200, noise_size=100, state_size=100):
        super(RNNGenerator, self).__init__()

        self._frame_size = frame_size
        self._noise_size = noise_size

        _z = KL.Input(shape=(None, noise_size))
        _lstm = KL.Bidirectional(KL.LSTM(units=state_size, activation='tanh', return_sequences=True))(_z)
        _x = KL.TimeDistributed(KL.Dense(frame_size, activation='tanh'))(_lstm)
        _x = KL.Lambda(lambda x: K.reshape(x, (K.shape(x)[0], K.shape(x)[1] * K.shape(x)[2])))(_x)

        self.model = KM.Model(inputs=_z, outputs=_x)

    def generate(self, batch_size=None, length=None, z=None):
        if z is None:
            _len = length // self._frame_size
            _z0 = K.random_normal((batch_size, 1, self._noise_size))
            _z1 = K.random_normal((batch_size, 1, self._noise_size))
            z = K.concatenate([_z0, K.zeros((batch_size, _len - 2, self._noise_size)), _z1], axis=1)
        else:
            batch_size = K.shape(z)[0]
            length = K.shape(z)[1] * self._frame_size
        return self.model(z)

class Conv1DGenerator(util.Component):
    def __init__(self, config, noise_size=100, old=False):
        super(Conv1DGenerator, self).__init__()

        _x = KL.Input(shape=(None,))
        _x1 = KL.Lambda(lambda x: K.expand_dims(K.expand_dims(x, 1), 3))(_x)
        self._multiplier = 1
        self._noise_size = noise_size
        for num_filters, filter_size, filter_stride in config:
            _x1 = KL.Conv2DTranspose(filters=num_filters, kernel_size=(1, filter_size), strides=(1, filter_stride), padding='same', kernel_initializer='random_uniform')(_x1)
            _x1 = KL.LeakyReLU()(_x1)
            self._multiplier *= filter_stride
        _x1 = KL.Conv2DTranspose(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(_x1)
        _x1 = KL.Activation('tanh')(_x1)
        _pooled = KL.Lambda(lambda x: x[:, 0, :, 0])(_x1)

        self.model = KM.Model(inputs=_x, outputs=_pooled)

    def generate(self, batch_size=None, length=None, z=None):
        if z is None:
            z = K.random_normal((batch_size, self._noise_size))
        return self.model(z)

class Discriminator(util.Component):
    def __init__(self):
        super(Discriminator, self).__init__()

    def discriminate(self, x):
        raise NotImplementedError

    def compare(self, x_real, x_fake, grad_penalty=True, lambda_=10):
        d_real = self.discriminate(x_real)
        d_fake = self.discriminate(x_fake)
        loss = d_fake - d_real

        if grad_penalty:
            eps = K.random_uniform([K.shape(x_real)[0], 1])
            x_inter = eps * x_real + (1 - eps) * x_fake
            d_inter = self.discriminate(x_inter)
            grads = K.gradients(d_inter, x_inter)[0]
            grad_norms = K.sqrt(K.sum(K.square(grads), axis=1))
            penalty = K.square(grad_norms - 1)

            loss += lambda_ * penalty
        else:
            penalty = K.constant(0)
        return K.mean(loss), d_real, d_fake, penalty

class RNNDiscriminator(Discriminator):
    def __init__(self, frame_size=200, state_size=100, num_frames=40):
        super(RNNDiscriminator, self).__init__()

        self._frame_size = frame_size
        self._state_size = state_size
        self._num_frames = num_frames

        _x = KL.Input(shape=(num_frames, frame_size))
        _lstm = KL.Bidirectional(KL.LSTM(units=state_size, activation='tanh', return_sequences=False, unroll=True))(_x)
        _d = KL.Dense(1)(_lstm)

        self.model = KM.Model(inputs=_x, outputs=_d)

    def discriminate(self, x):
        x = K.reshape(x, (K.shape(x)[0], self._num_frames, self._frame_size))
        return self.model(x)[:, 0]

class Conv1DDiscriminator(Discriminator):
    def __init__(self, config):
        super(Conv1DDiscriminator, self).__init__()

        _x = KL.Input(shape=(None,))
        _x1 = KL.Lambda(lambda x: K.expand_dims(x, 2))(_x)
        for num_filters, filter_size, filter_stride in config:
            _x1 = KL.Conv1D(filters=num_filters, kernel_size=filter_size, strides=filter_stride, padding='valid')(_x1)
            _x1 = KL.LeakyReLU()(_x1)
        _pooled = KL.GlobalAvgPool1D()(_x1)
        _d = KL.Dense(1)(_pooled)

        self.model = KM.Model(inputs=_x, outputs=_d)

    def discriminate(self, x):
        return self.model(x)[:, 0]
