
import keras.backend as K
from keras.engine import Layer
import keras.initializers as KI

class Scale(Layer):
    def __init__(self, initial_value=1, **kwargs):
        self.initial_value = initial_value
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale',
                                     shape=(),
                                     initializer=KI.constant(self.initial_value),
                                     trainable=True)
        super(Scale, self).build(input_shape)

    def call(self, inputs):
        return self.scale * inputs

    def compute_output_shape(self, input_shape):
        return input_shape
