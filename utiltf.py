
import tensorflow as TF
import keras.layers as KL
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as SPM
import datetime
import os

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



def plot_waves(session, gan, x_gen, g_writer, step):
    buf = []
    for x in x_gen:
        plt.plot(x)
        plt.savefig('temp.png')
        plt.close()
        buf.append(SPM.imread('temp.png', mode='RGB'))

    plot = session.run(gan.buf_plot_op, feed_dict={gan.buf_ph: np.array(buf)})
    g_writer.add_summary(plot, step)


def wasserstein(d_real, d_fake):
    return d_fake - d_real


def l2_loss(d_real, d_fake):
    d_fake = TF.nn.sigmoid(d_fake)
    d_real = TF.nn.sigmoid(d_real)
    return TF.square(d_fake - 1) + TF.square(d_real - 0)


def cross_entropy(d_real, d_fake):
    y_fake = TF.ones_like(d_fake)
    y_real = TF.zeros_like(d_real)
    return TF.nn.sigmoid_cross_entropy_with_logits(labels=y_fake, logits=d_fake) + \
           TF.nn.sigmoid_cross_entropy_with_logits(labels=y_real, logits=d_real)


def wasserstein_g(d_fake):
    return -d_fake


def l2_loss_g(d_fake):
    return TF.square(d_fake - 0)


def cross_entropy_g(d_fake):
    y_fake = TF.zeros_like(d_fake)
    return TF.losses.sigmoid_cross_entropy(y_fake, d_fake)


def logdirs(logdir, modelnamesave):
    logdir = (
            logdir + '/%s-%s' % 
            (modelnamesave, datetime.datetime.strftime(
                datetime.datetime.now(), '%Y%m%d%H%M%S')
                )
            )
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    elif not os.path.isdir(logdir):
        raise IOError('%s is not a directory' % logdir)
    log_train_d = '%s/train-d' % logdir
    log_valid_d = '%s/valid-d' % logdir
    log_train_g = '%s/train-g' % logdir
    for subdir in [log_train_d, log_valid_d, log_train_g]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        elif not os.path.isdir(subdir):
            raise IOError('%s exists and is not a directory' % subdir)

    return log_train_d, log_valid_d, log_train_g


def log_one_minus_sigmoid(x):
    y_neg = TF.log(1 - TF.sigmoid(x))
    y_pos = -x - TF.log(1 + TF.exp(-x))
    return TF.where(x > 0, y_pos, y_neg)


def div_roundup(x, d):
    return (x + d - 1) // d

def roundup(x, d):
    return div_roundup(x, d) * d
