
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
