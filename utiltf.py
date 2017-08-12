
import tensorflow as TF
import keras.layers as KL

def summarize_var(var, name, mean=False, std=False, max_=False, min_=False):
    summaries = []
    m = TF.reduce_mean(var)
    with TF.name_scope(name):
        if mean:
            summaries.append(TF.summary.scalar('mean', m))
        if std:
            with TF.name_scope('stddev'):       # ?????
                stddev = TF.sqrt(TF.reduce_mean(TF.square(var - m)))
            summaries.append(TF.summary.scalar('std', stddev))
        if max_:
            summaries.append(TF.summary.scalar('max', TF.reduce_max(var)))
        if min_:
            summaries.append(TF.summary.scalar('min', TF.reduce_min(var)))

    return TF.summary.merge(summaries)


class AutoUpdate(KL.Wrapper):
    # Keras update ops would only be created/run if a Function object is
    # created, which is pretty unfriendly to Functional API.  I wrote a
    # wrapper for automatically including the update ops into
    # TF.GraphKeys.UPDATE_OPS collection.
    layers = []
    def __init__(self, layer, **kwargs):
        super(AutoUpdate, self).__init__(layer, **kwargs)
        AutoUpdate.layers.append(self.layer)

    def build(self, input_shape):
        return self.layer.build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, inputs, **kwargs):
        return self.layer.call(inputs, **kwargs)

    @staticmethod
    def get_update_op():
        ops = []
        for l in AutoUpdate.layers:
            for u in l.updates:
                if isinstance(u, tuple):
                    p, new_p = u
                    ops.append(TF.assign(p, new_p))
                else:
                    ops.append(u)
        return TF.group(*ops)
