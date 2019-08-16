from __future__ import print_function

import numpy as np
import pandas as pd
import torch

import util as ut


class KeannTextClassDataLoader(object):

    def __init__(self, path_file, word_to_index, batch_size=32, max_length=80):
        """

        Args:
            path_file:
            word_to_index:
            batch_size:
        """

        self.batch_size = batch_size
        self.word_to_index = word_to_index
        self.max_length = max_length

        # read file
        df = pd.read_csv(path_file, delimiter='\t')
        df['arg1'] = df['arg1'].apply(ut._tokenize)
        df['arg1'] = df['arg1'].apply(self.generate_indexifyer())
        df['arg2'] = df['arg2'].apply(ut._tokenize)
        df['arg2'] = df['arg2'].apply(self.generate_indexifyer())

        self.samples = df.values.tolist()

        # for batch
        self.n_samples = len(self.samples)
        self.n_batches = int(self.n_samples / self.batch_size)
        self._shuffle_indices()

        self.report()

    def _shuffle_indices(self):
        self.indices = np.random.permutation(self.n_samples)
        self.index = 0
        self.batch_index = 0

    def generate_indexifyer(self):

        def indexify(lst_text):
            indices = []
            for word in lst_text:
                if word in self.word_to_index:
                    indices.append(self.word_to_index[word])
                else:
                    indices.append(self.word_to_index['__UNK__'])
            return indices

        return indexify

    @staticmethod
    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] = batch_x[i] + [0 for _ in range(missing)]
        return batch_x

    def _create_batch(self):
        batch = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        label, string1, string2 = tuple(zip(*batch))

        # get the length of each seq in your batch fixed by self.max_length
        seq_lengths = torch.from_numpy(np.full((len(string1),), self.max_length))

        # dump padding everywhere, and place seqs on the left.
        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor_arg1 = torch.zeros((len(string1), seq_lengths.max())).long()
        seq_tensor_arg2 = torch.zeros((len(string2), seq_lengths.max())).long()

        for idx, (seq1, seq2, seqlen) in enumerate(zip(string1, string2, seq_lengths)):
            if len(seq1) > self.max_length:
                seq1 = seq1[:self.max_length]
            if len(seq2) > self.max_length:
                seq2 = seq2[:self.max_length]

            seq_tensor_arg1[idx, :len(seq1)] = torch.LongTensor(seq1)
            seq_tensor_arg2[idx, :len(seq2)] = torch.LongTensor(seq2)

        # SORT YOUR TENSORS BY LENGTH!
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        seq_tensor_arg1 = seq_tensor_arg1[perm_idx]
        seq_tensor_arg2 = seq_tensor_arg2[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        label = torch.LongTensor(label)
        label = label[perm_idx]

        # seq_tensor.shape is [batch_size x max_length], label.shape is [batch_size], seq_lengths.shape is [batch_size]
        return seq_tensor_arg1, seq_tensor_arg2, label, seq_lengths

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self._shuffle_indices()
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

    def show_samples(self, n=10):
        for sample in self.samples[:n]:
            print(sample)

    def report(self):
        print('# samples: {}'.format(len(self.samples)))
        print('# max len: {}'.format(self.max_length))
        print('# vocab: {}'.format(len(self.word_to_index)))
        print('# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size))
