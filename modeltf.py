
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
def add_common_layers(y):
    y = KL.BatchNormalization()(y)
    y = KL.LeakyReLU()(y)

    return y
cardinality=1
def grouped_convolution(y, nb_channels, _strides):
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return KL.Conv2D(nb_channels, kernel_size=(1, 3), strides=_strides, padding='same')(y)
    
    assert not nb_channels % cardinality
    _d = nb_channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = KL.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
        groups.append(KL.Conv2D(_d, kernel_size=(1, 3), strides=_strides, padding='same')(group))
        
    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = KL.concatenate(groups)

    return y
def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:
    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """
    shortcut = y

    # we modify the residual building block as a bottleneck design to make the network more economical
    y = KL.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    y = add_common_layers(y)

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    y = grouped_convolution(y, nb_channels_in, _strides=_strides)
    y = add_common_layers(y)

    y = KL.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    # batch normalization is employed after aggregating the transformations and before adding to the shortcut
    y = KL.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = KL.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = KL.BatchNormalization()(shortcut)

    y = KL.add([shortcut, y])

    # relu is performed right after each batch normalization,
    # expect for the output of the block where relu is performed after the adding to the shortcut
    y = KL.LeakyReLU()(y)

    return y

class Conv1DGenerator(util.Component):
    def __init__(self, config, noise_size=100, old=False, resnet = True):
        super(Conv1DGenerator, self).__init__()

        _x = KL.Input(shape=(None,))
        _x1 = KL.Lambda(lambda x: K.expand_dims(K.expand_dims(x, 1), 3))(_x)
        self._multiplier = 1
        self._noise_size = noise_size
        last_num_filters = 0
        for num_filters, filter_size, filter_stride in config:
            if resnet and filter_stride ==1 and num_filters == last_num_filters:
                #nb_channels_out = _x1.shape[-1]
                #nb_channels_in = nb_channels_out//2
                _x1 = residual_block(_x1, num_filters//2, num_filters)
            else:
                _x1 = KL.Conv2DTranspose(filters=num_filters, kernel_size=(1, filter_size), strides=(1, filter_stride), padding='same', kernel_initializer='random_uniform')(_x1)
                _x1 = KL.LeakyReLU()(_x1)
            self._multiplier *= filter_stride
            last_num_filters = num_filters
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
