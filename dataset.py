
import os
import h5py
import numpy.random as RNG
import numpy as NP
from torch.utils.data import DataLoader, Dataset
import torch as T
from collections import OrderedDict

n_fetches = 0
n_zero_fetches = 0

def transform(x):
    x[x==0] = x[x==0] + 1e-6
    return NP.log(x)

def invtransform(x):
    return NP.exp(x)

def _valid_keys(keys):
    keys = [k for k in keys if not (k[-1] == '-' or k[0] == '(')]
    keys = [k for k in keys if len(k) >= 1]
    return keys

def word_to_seq(word, maxcharlen):
    char_seq = NP.zeros(maxcharlen, dtype=NP.int32)
    char_seq[:len(word)] = [ord(c) for c in word]
    return char_seq


class AudioDataset(Dataset):
    def __init__(self, directory, keys, maxlen):
        self.directory = directory
        self.keys = keys
        self.maxcharlen = max(len(k) for k in keys)

        datadir = [os.path.join(directory, d)
                   for d in os.listdir(directory)
                   if d.endswith('-data')]
        datadir_with_keys = OrderedDict((k, os.path.join(directory, k + '-data')) for k in keys)
        filename = os.path.join(datadir[0], '0000000.npy')
        sample = NP.load(filename)
        self.nfreq = sample.shape[0]

        self._size = [len([f for f in os.listdir(v) if f.endswith('.npy')])
                      for k, v in datadir_with_keys.items()]
        self.maxlen = maxlen or max(
                NP.load(self.get_filename(k, 0)).shape[1] for k in keys)
        self._total_size = sum(self._size)
        self._index_map = NP.array([0] + list(NP.cumsum(self._size)[:-1]))

    def __len__(self):
        return self._total_size

    def get_filename(self, key, index):
        return os.path.join(self.directory, key + '-data', '%07d.npy' % index)

    def pick_word(self):
        key = NP.asscalar(RNG.choice(self.keys))
        return key, word_to_seq(key, self.maxcharlen), len(key)

    def __getitem__(self, index):
        global n_fetches, n_zero_fetches
        n_fetches += 1

        key_idx = NP.searchsorted(self._index_map, index, 'right') - 1
        chunk_offset = index - self._index_map[key_idx]

        key = NP.asscalar(self.keys[key_idx])
        key_cseq = word_to_seq(key, self.maxcharlen)
        key_len = len(key)
        sample_out = NP.zeros((self.nfreq, self.maxlen))
        sample_in = NP.load(self.get_filename(key, chunk_offset))
        # Truncating rather than discarding
        sample_len = min(sum(1 - NP.cumprod((sample_in.sum(0) == 0)[::-1])), self.maxlen)
        sample_out[:, :sample_len] = sample_in[:, :sample_len]
        if sample_len == 0:
            n_zero_fetches += 1
            print 'Fetches %d all zero samples out of %d samples' % n_zero_fetches, n_fetches
        sample_out = transform(sample_out)

        return key, key_cseq, key_len, sample_out, sample_len


class AudioDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, num_workers=2):
        DataLoader.__init__(self,
                            dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            drop_last=True)

    def pick_words(self):
        samples = []
        for _ in range(self.batch_size):
            samples.append(self.dataset.pick_word())
        return self.collate_fn(samples)


def prepare(batch_size, directory, maxlen=None):
    keys = [d[:-5] for d in os.listdir(directory) if d.endswith('-data')]
    keys = _valid_keys(keys)
    keys = list(RNG.permutation(keys))
    n_train_keys = len(keys) // 10 * 9

    train_dataset = AudioDataset(directory, keys[:n_train_keys], maxlen)
    valid_dataset = AudioDataset(directory, keys[n_train_keys:], maxlen)

    train_dataloader = AudioDataLoader(train_dataset, batch_size, 4)
    valid_dataloader = AudioDataLoader(valid_dataset, batch_size, 1)

    return train_dataloader, valid_dataloader
