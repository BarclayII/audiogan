
import tensorflow as TF
import keras.layers as KL
import matplotlib.pyplot as plt
import numpy as np

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



def plot_waves(x, row, col, fig_name):
    f, a = plt.subplots(row, col, figsize=(col*5, row*1.8))
    for j in range(row):
        for i in range(col):
            idx = i + j*col
            a[j][i].plot(x[idx,:])
            a[j,i].set_xticks([])
    f.savefig(fig_name)
    plt.close()