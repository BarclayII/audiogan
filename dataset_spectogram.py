import librosa
import librosa.display
import numpy as np
import time
import h5py
import sys, os
import matplotlib.pyplot as PL
fold = h5py.File('dataset-small.h5','r')
f = h5py.File("data-spect-sm.h5", "w")

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
    for idx in range(data.shape[0]):
        x = data[idx,:]
        #x = librosa.load('/tmp/a.wav', sr=8000)[0]  # The input
        x = librosa.util.fix_length(x, x.shape[0] + n_fft // 2) # See documentation of librosa.istft
        y = librosa.stft(x)
        y_abs = np.abs(y)
        spect_data.append(y_abs)
    f[key] = np.stack(spect_data)
    #spect_data = boolean_indexing(spect_data)
        #f[str(total_idx)] = y_abs


