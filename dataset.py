
import h5py
import numpy.random as RNG
import numpy as NP
import utiltf as util

def _unconditional_dataloader(batch_size, data, lower, upper, args):
    epoch = 1
    batch = 0
    idx = RNG.permutation(range(lower, upper))
    cur = 0

    while True:
        indices = []
        for i in range(batch_size):
            if cur == len(idx):
                cur = 0
                idx_set = list(set(range(lower, upper)) - set(indices))
                idx = RNG.permutation(idx_set)
                epoch += 1
                batch = 0
            indices.append(idx[cur])
            cur += 1
        sample = data[sorted(indices)]
        yield [epoch, batch, NP.array(sample)[:, :args.amplitudes]] + [None] * 6
        batch += 1

def unconditional_dataloader(batch_size, args):
    dataset = h5py.File(args.dataset)
    data = dataset['data']
    nsamples = data.shape[0]
    if args.subset:
        nsample_indices = RNG.permutation(range(nsamples))[:args.subset]
        data = data[sorted(nsample_indices)]
        nsamples = args.subset
    n_train_samples = nsamples // 10 * 9

    dataloader = _unconditional_dataloader(batch_size, data, 0, n_train_samples, args)
    dataloader_val = _unconditional_dataloader(batch_size, data, n_train_samples, nsamples, args)

    return None, dataloader, dataloader_val

def word_to_seq(word, maxcharlen):
    char_seq = NP.zeros(maxcharlen, dtype=NP.int32)
    char_seq[:len(word)] = [ord(c) for c in word]
    return char_seq

def _pick_sample_from_word(key, maxlen, dataset, frame_size=None, skip_samples=False, nfreq=None):
    sample_idx = RNG.choice(dataset[key].shape[0])
    sample_out = NP.zeros((nfreq,maxlen))
    length = 0
    if not skip_samples:
        sample_in = dataset[key][sample_idx]
        
        sample_len = sum(1 - NP.cumprod((sample_in.sum(0) == 0)[::-1]))
        if sample_len > maxlen:
            return None, None
        length = sample_len

        sample_out[:,:sample_len] = sample_in[:,:sample_len]
    return sample_out, length

def transform(x):
    return x - 0.5

def invtransform(x):
    return x + 0.5

def pick_word(maxlen, dataset, keys, maxcharlen, args, frame_size=None, skip_samples=False):
    while True:
        key = RNG.choice(keys)
        sample_out, length = _pick_sample_from_word(key, maxlen, dataset, frame_size, skip_samples, args.nfreq)
        if sample_out is not None:
            if not skip_samples:
                maxabs = NP.abs(sample_out).max()
                if maxabs == 0:
                    continue
                sample_out = transform(sample_out / maxabs)
            break

    return key, word_to_seq(key, maxcharlen), len(key), sample_out, length

def pick_words(batch_size, maxlen, dataset, keys, maxcharlen, args, frame_size=None, skip_samples=False):
    return [NP.array(a) for a in zip(*(pick_word(maxlen, dataset, keys, maxcharlen, args, frame_size, skip_samples) for _ in range(batch_size)))]

def _conditional_dataloader(batch_size, dataset, maxlen, keys, args, frame_size=None):
    epoch = 0
    batch = 0
    maxcharlen = max(len(k) for k in keys)

    if frame_size is not None:
        pass#maxlen = util.roundup(maxlen, frame_size)
    while True:
        samples = []
        batch += 1
        i = 0
        picked_keys, cseq, clen, samples, lengths = pick_words(batch_size, maxlen, dataset, keys, maxcharlen, args, frame_size)
        yield [epoch, batch, samples, lengths, picked_keys, cseq, clen]

def _valid_keys(keys, args):
    keys = [k for k in keys if not (k[-1] == '-' or k[0] == '(')]
    keys = [k for k in keys if len(k) >= args.minwordlen]
    return keys

def conditional_dataloader(batch_size, args, maxlen=None, frame_size=None):
    dataset = h5py.File(args.dataset)
    keys = _valid_keys(dataset.keys(), args)
    keys = list(RNG.permutation(keys))
    n_train_keys = len(keys) // 10 * 9
    maxlen = maxlen or max(dataset[k].shape[2] for k in keys)

    dataloader = _conditional_dataloader(
            batch_size, dataset, maxlen, keys[:n_train_keys], args, frame_size)
    dataloader_val = _conditional_dataloader(
            batch_size, dataset, maxlen, keys[n_train_keys:], args, frame_size)

    return dataset, maxlen, dataloader, dataloader_val, keys[:n_train_keys], keys[n_train_keys:]

def dataloader(batch_size, args, maxlen=None, frame_size=None):
    # Returns a generator which returns
    # (epoch, batch, audio, word, char_seq, char_seq_len, word_wrong, char_seq_wrong, char_seq_wrong_len)
    if not args.conditional:
        return unconditional_dataloader(batch_size, args)
    else:
        return conditional_dataloader(batch_size, args, maxlen=maxlen, frame_size=frame_size)
