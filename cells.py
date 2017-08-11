
import tensorflow as TF

class Conv2DLSTMCell(TF.nn.rnn_cell.RNNCell):
    """
    A LSTM cell with convolutions instead of multiplications.
    Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
    """

    def __init__(self,
                 shape,                     # ConvLSTM state shape
                 filters,
                 kernel,
                 forget_bias=1.0,
                 activation=TF.tanh,
                 normalize=True,
                 peephole=True,
                 data_format='channels_last',
                 pre_rnn_callback=None,     # Callback function before RNN
                 post_rnn_callback=None,    # Callback function after RNN
                 output_shape=None,         # Supply this if you have @post_rnn
                 reuse=None):
        super(Conv2DLSTMCell, self).__init__(_reuse=reuse)
        shape = list(shape)
        self._kernel = list(kernel)
        self._filters = filters
        self._forget_bias = forget_bias
        self._activation = activation
        self._normalize = normalize
        self._peephole = peephole
        if data_format == 'channels_last':
            self._size = TF.TensorShape(shape + [self._filters])
            self._feature_axis = self._size.ndims
            self._data_format = None
        elif data_format == 'channels_first':
            self._size = TF.TensorShape([self._filters] + shape)
            self._feature_axis = 0
            self._data_format = 'NC'
        else:
            raise ValueError('Unknown data_format')
        self._pre_rnn_callback = pre_rnn_callback
        self._post_rnn_callback = post_rnn_callback
        self._output_shape = TF.TensorShape(output_shape)

    @property
    def state_size(self):
        return TF.nn.rnn_cell.LSTMStateTuple(self._size, self.output_size)

    @property
    def output_size(self):
        return self._output_shape or self._size

    def call(self, x, state):
        c, h = state

        if self._pre_rnn_callback:
            x, h = self._pre_rnn_callback(x, h)

        x = TF.concat([x, h], axis=self._feature_axis)
        n = x.shape[-1].value
        m = 4 * self._filters if self._filters > 1 else 4
        W = TF.get_variable('kernel', self._kernel + [n, m])
        y = TF.nn.convolution(x, W, 'SAME', data_format=self._data_format)
        if not self._normalize:
            y += TF.get_variable('bias', [m], initializer=TF.zeros_initializer())
        j, i, f, o = TF.split(y, 4, axis=self._feature_axis)

        if self._peephole:
            i += TF.get_variable('W_ci', c.shape[1:]) * c
            f += TF.get_variable('W_cf', c.shape[1:]) * c

        if self._normalize:
            j = TF.contrib.layers.layer_norm(j)
            i = TF.contrib.layers.layer_norm(i)
            f = TF.contrib.layers.layer_norm(f)

        f = TF.sigmoid(f + self._forget_bias)
        i = TF.sigmoid(i)
        c = c * f + i * self._activation(j)

        if self._peephole:
            o += TF.get_variable('W_co', c.shape[1:]) * c

        if self._normalize:
            o = TF.contrib.layers.layer_norm(o)
            c = TF.contrib.layers.layer_norm(c)

        o = TF.sigmoid(o)
        h = o * self._activation(c)

        # TODO
        #TF.summary.histogram('forget_gate', f)
        #TF.summary.histogram('input_gate', i)
        #TF.summary.histogram('output_gate', o)
        #TF.summary.histogram('cell_state', c)

        if self._post_rnn_callback:
            h = self._post_rnn_callback(h)

        state = TF.nn.rnn_cell.LSTMStateTuple(c, h)

        return h, state

class ProjectedLSTMCell(TF.nn.rnn_cell.LSTMCell):
    '''
    A monkey patch for LSTMCell since
    (1) LSTMCell already has projection, and
    (2) It feeds back the projection output to input, but
    (3) The projection is linear
    '''
    def __init__(self,
                 num_units,
                 num_proj,
                 projection_activation=TF.tanh,
                 **kwargs):
        self._projection_activation = projection_activation
        super(ProjectedLSTMCell, self).__init__(
                num_units, num_proj=num_proj, **kwargs)

    def call(self, inputs, state):
        h, state = super(ProjectedLSTMCell, self).call(inputs, state)
        c, h = state
        h = self._projection_activation(h)
        return h, TF.nn.rnn_cell.LSTMStateTuple(c, h)
