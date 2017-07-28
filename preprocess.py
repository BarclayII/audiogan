
# Usage:
# python2 preprocess.py THRESHOLD DATASET-NAME FILELIST-NAME [SAMPLE-RATE]
import sys
import librosa
import numpy as NP
import h5py

data = []
thres = float(sys.argv[1])
if len(sys.argv) > 4:
    sr = int(sys.argv[4])
else:
    sr = 8000

datafile = h5py.File(sys.argv[2], 'w')
dataset = datafile.create_dataset('data', shape=(0, sr),
                                  maxshape=(None, sr), dtype='float32',
                                  compression='gzip')

with open(sys.argv[3]) as filelist:
    for f in filelist:
        print f.strip()
        x, _ = librosa.core.load(f.strip(), sr=sr)
        for i in range(0, x.shape[0] - sr + 1):
            if (NP.abs(x[i:i+sr]) > thres).sum() > sr / 2:
                data.append(x[i:i+sr])
            if len(data) == sr:
                old_shape = dataset.shape[0]
                dataset.resize(old_shape + len(data), axis=0)
                dataset[old_shape:] = data
                data = []

if len(data) > 0:
    old_shape = dataset.shape[0]
    dataset.resize(old_shape + len(data), axis=0)
    dataset[old_shape:] = data
datafile.close()
