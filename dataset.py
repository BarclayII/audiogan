
import h5py
import numpy.random as RNG
import numpy as NP

def _dataloader(batch_size, data, lower, upper, args):
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
        yield epoch, batch, NP.array(sample)[:, :args.amplitudes]
        batch += 1

def dataloader(batch_size, args):
    dataset = h5py.File(args.dataset)
    data = dataset['data']
    nsamples = data.shape[0]
    if args.subset:
        nsample_indices = RNG.permutation(range(nsamples))[:args.subset]
        data = data[sorted(nsample_indices)]
        nsamples = args.subset
    n_train_samples = nsamples // 10 * 9

    dataloader = _dataloader(batch_size, data, 0, n_train_samples, args)
    dataloader_val = _dataloader(batch_size, data, n_train_samples, nsamples, args)

    return dataloader, dataloader_val
