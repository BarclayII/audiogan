import librosa
import librosa.display
import numpy as np
import time
import h5py
import sys, os
import matplotlib.pyplot as PL
fold = h5py.File('dataset.h5','r')
f = h5py.File("data-spect-full.h5", "w")

def boolean_indexing(v):
    lens = np.array([item.shape[1] for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.zeros(mask.shape,dtype=int)
    out[mask] = np.concatenate(v)
    return out


total_idx = 0
n_fft = 2048
keys = fold.keys()
for key in keys:
    data = fold[key].value
    spect_data = []
    first = True
    pos = 0
    print key
    for idx in range(data.shape[0]):
        x = data[idx,:]
        #x = librosa.load('/tmp/a.wav', sr=8000)[0]  # The input
        x = librosa.util.fix_length(x, x.shape[0] + n_fft // 2) # See documentation of librosa.istft
        y = librosa.stft(x)
        y_abs = np.abs(y)
        spect_data.append(y_abs)

        if first:
            f.create_dataset(key, shape=[data.shape[0]] + list(y_abs.shape), compression='gzip')
            first = False

        if len(spect_data) >= 100 or idx == data.shape[0] - 1:
            f[key][pos:pos+len(spect_data)] = np.stack(spect_data)
            pos += len(spect_data)
            spect_data = []
            print '\t', pos
