#! /usr/bin/env python
# Usage:
# preprocess.py transcript_file_list_file sphere_file_list_file dataset
# where dataset can be either an HDF5 file (suffixed by .h5) or a directory.
#
# Put this program under the same directory as FAAValign.py
# Also you need sph2pipe to uncompress the sphere files
import sys
import os
import subprocess
import praat
import h5py
import librosa
import numpy as NP
from collections import Counter
import multiprocessing as MP
import tempfile
import logging
import sh
import shutil

NUM_WORKERS = 10
sph2pipe = 'sph2pipe_v2.5/sph2pipe'
faav_align = 'FAAValign.py'

def setup_workdir(workdir):
    shutil.copy('FAAValign.py', workdir)
    shutil.copy('get_duration.praat', workdir)
    shutil.copytree('model', os.path.join(workdir, 'model'))
    shutil.copy('praat.py', workdir)

class HDF5Writer(object):
    def __init__(self, h5, buffer_slots=1000, buffer_size=1000):
        self.h5 = h5
        self.buffer = {}
        self.buffer_size = buffer_size
        self.buffer_slots = buffer_slots
        self.hist = []

    def append(self, dataset, value):
        if dataset not in self.h5:
            shape = [0] * (value.ndim + 1)
            maxshape = [None] * (value.ndim + 1)
            self.h5.create_dataset(
                    dataset, shape=shape, dtype=NP.float32, maxshape=maxshape,
                    compression='gzip')
        if dataset not in self.buffer:
            # Check if the buffer slots are full, and flush one buffer slot
            if len(self.buffer) == self.buffer_slots:
                self._flush_one()
            self.buffer[dataset] = []
            self.hist.append(dataset)

        if len(self.buffer[dataset]) == self.buffer_size:
            self._flush(dataset)
        else:
            # Update access history
            self.hist.remove(dataset)
        self.buffer[dataset].append(value)
        self.hist.append(dataset)
        logging.debug('New value added to dataset %s' % dataset)

    def close(self):
        self._flush_all()
        self.h5.close()

    def _flush(self, victim):
        shape = self.h5[victim].shape
        recs = shape[0]

        # convert buffer list to numpy array
        maxshape = [max(s) for s in zip(*([shape[1:]] + [a.shape for a in self.buffer[victim]]))]
        buffer_recs = len(self.buffer[victim])
        buffer_ = NP.zeros([buffer_recs] + maxshape)
        for i, a in enumerate(self.buffer[victim]):
            pad_shape = [(0, maxshape[_] - a.shape[_]) for _ in range(len(maxshape))]
            a_pad = NP.pad(a, pad_shape, 'constant')
            buffer_[i] = a_pad

        target_shape = [buffer_recs + recs] + maxshape

        self.h5[victim].resize(target_shape)
        self.h5[victim][recs:recs+buffer_recs] = buffer_

        self.buffer[victim] = []

        self.hist.remove(victim)
        logging.debug('Flushed dataset %s with %d buffered records' % (victim, buffer_recs))

    def _flush_one(self):
        victim = self.hist[0]
        self._flush(victim)
        del self.buffer[victim]

    def _flush_all(self):
        for k in self.buffer:
            self._flush(k)
        self.buffer.clear()

def worker(q, p, console, workdir):
    setup_workdir(workdir)

    while True:
        tran, wav = q.get()
        if tran is None:
            break
        output = wav[:-4] + '.TextGrid'
        faavlog = wav[:-4] + '.FAAVlog'
        errorlog = wav[:-4] + '.errorlog'
        with console:
            print tran, '+', wav, '->', output

        work_tran = os.path.join(workdir, os.path.basename(tran))
        work_wav = os.path.join(workdir, os.path.basename(wav))
        work_output = os.path.join(workdir, os.path.basename(output))
        work_faavlog = os.path.join(workdir, os.path.basename(faavlog))
        work_errorlog = os.path.join(workdir, os.path.basename(errorlog))
        sh.cp(tran, work_tran)
        sh.cp(wav, work_wav)
        subprocess.check_output(
                ['python', faav_align, '-vn', work_wav, work_tran, work_output],
                cwd=workdir,
                )

        grid = praat.TextGrid()
        grid.read(work_output)

        for tier in grid:
            assert isinstance(tier, praat.IntervalTier)
            if tier.name().find('word') == -1:
                continue
            nbi = [i for i in range(len(tier)) if tier[i].mark() != 'sp']
            with console:
                print '%s: Non-blank intervals: %d' % (output, len(nbi))
            for i in range(len(nbi) - 1):
                prev = nbi[i]
                next_ = nbi[i+1]
                if tier[prev].xmax() > tier[next_].xmin():
                    print '\t', output, str(tier[prev]), str(tier[next_].mark())
            starts = []
            ends = []
            words = []
            for i in nbi:
                interval = tier[i]
                starts.append(str(int(interval.xmin() * 8000)))
                ends.append(str(int(interval.xmax() * 8000)))
                words.append(interval.mark())

            p.put((wav, ' '.join(starts), ' '.join(ends), ' '.join(words)))

        sh.rm('-f', work_tran, work_wav, work_output, work_faavlog, work_errorlog)

    p.put((None, None, None, None))

if __name__ == '__main__':
    trans = {}
    wavs = {}
    logging.basicConfig(level=logging.DEBUG, filename='debug.log')

    # Convert transcripts to ELAN format
    with open(sys.argv[1]) as trans_file_list:
        for filename in trans_file_list:
            fname = filename.strip()
            gname = fname[:-4] + '-trans.txt'
            trans[os.path.basename(fname)[:-4]] = gname
            print fname, '->', gname
            #'''
            f = open(fname)
            g = open(gname, 'w')
            for l in f:
                try:
                    start, end, spk, words = l.strip().split(' ', 3)
                    start = float(start)
                    end = float(end)
                    spk = spk[0]
                    g.write('\t'.join([spk, spk, '%.2f' % start, '%.2f' % end, words]) + '\n')
                except:
                    continue
            g.close()
            f.close()
            #'''

    # Uncompress sphere files
    with open(sys.argv[2]) as sphere_file_list:
        for filename in sphere_file_list:
            fname = filename.strip()
            output = fname[:-4] + '.wav'
            print fname, '->', output
            #'''
            subprocess.check_output([sph2pipe, '-f', 'wav', fname, output])
            #'''
            wavs[os.path.basename(fname)[:-4]] = output

    # Prepare output HDF5/directory
    if sys.argv[3].endswith('.h5'):
        h5 = h5py.File(sys.argv[3], 'w')
        write_h5 = True
        h5writer = HDF5Writer(h5)
    else:
        h5 = None
        write_h5 = False
        if not os.path.exists(sys.argv[3]):
            os.mkdir(sys.argv[3])

    # Prepare for parallelizing multiple FAAValign.py jobs
    wordfreq = Counter()
    q = MP.Queue()
    p = MP.Queue()
    console = MP.Lock()

    for k in set(trans.keys()) & set(wavs.keys()):
        q.put((trans[k], wavs[k]))
    for _ in range(NUM_WORKERS):
        q.put((None, None))

    workdirs = []
    for i in range(NUM_WORKERS):
        workdir = tempfile.mkdtemp()
        workdirs.append(workdir)
        proc = MP.Process(target=worker, args=(q, p, console, workdir))
        proc.start()

    # Take from output queue and write them to dataset
    done = 0
    while True:
        wav, starts, ends, words = p.get()
        if wav is None:
            done += 1
            if done == NUM_WORKERS:
                break
        else:
            amps = librosa.load(wav, sr=8000)[0]
            starts = [int(s) for s in starts.split()]
            ends = [int(s) for s in ends.split()]
            words = words.split()
            assert len(starts) == len(ends) == len(words)
            for start, end, word in zip(starts, ends, words):
                wordfreq.update([word])

                amp_output = NP.zeros(max(10, end - start))
                amp_output[:end-start] = amps[start:end]

                if not write_h5:
                    target_dir = os.path.join(sys.argv[3], word)
                    if not os.path.exists(target_dir):
                        os.mkdir(target_dir)
                    filename = os.path.join(target_dir, str(wordfreq[word]))
                    librosa.output.write_wav(filename, amp_output, sr=8000)
                else:
                    h5writer.append(word, amp_output)

            with console:
                print 'Wrote word segments:', ' '.join(words)

    print 'Total word segments', wordfreq
    print 'Number of different words', len(wordfreq)

    if write_h5:
        h5writer.close()
    for workdir in workdirs:
        sh.rm('-rf', workdir)
