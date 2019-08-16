from __future__ import print_function

import numpy as np
import pandas as pd
import torch

import util as ut


class KeannKGTextClassDataLoader(object):

    def __init__(self,
                 path_file,
                 word_to_index,
                 batch_size=32,
                 max_length=80,
                 transE_embedding_path='',
                 transE_model_path='',
                 transE_entity_path=''):
        """

        Args:
            path_file:
            word_to_index:
            batch_size:
        """

        self.batch_size = batch_size
        self.word_to_index = word_to_index
        self.max_length = max_length
        self.transE_embedding_path = transE_embedding_path
        self.transE_model_path = transE_model_path
        self.transE_entity_path = transE_entity_path
        self.entity2id = {}

        # read file
        df = pd.read_csv(path_file, delimiter='\t')
        df['arg1'] = df['arg1'].apply(ut._tokenize)
        df['arg2'] = df['arg2'].apply(ut._tokenize)
        self.samples_original = df.values.tolist()
        df['arg1'] = df['arg1'].apply(self.generate_indexifyer())
        df['arg2'] = df['arg2'].apply(self.generate_indexifyer())

        self.init_transE_model()
        print("===> dealing sentences transE ...")
        df_args = pd.read_csv(path_file, delimiter='\t')
        df_args['arg1'] = df_args['arg1'].apply(ut._tokenize)
        df_args['arg2'] = df_args['arg2'].apply(ut._tokenize)
        df_args['arg1'] = df_args['arg1'].apply(self.get_transE_embedding())
        df_args['arg2'] = df_args['arg2'].apply(self.get_transE_embedding())

        self.samples = df.values.tolist()
        self.samples_args = df_args.values.tolist()

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

    def init_transE_model(self):
        print("===> initing transE model ...")
        fb = open(self.transE_embedding_path, 'r')
        self.ent_embeddings = np.array(list(map(lambda x: x.strip().split(), fb.readlines())), dtype=np.float)
        fb.close()

        # model = torch.load(self.transE_model_path)
        # self.ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy()

        f = open(self.transE_entity_path, 'r')
        lines = f.readlines()
        entity2id_list = list(map(lambda x: x.strip(), lines))
        f.close()

        for i, item in enumerate(entity2id_list):
            if i > 2:
                item_split = item.split(' ')
                self.entity2id[item_split[0]] = item_split[1]

    def get_transE_embedding(self):

        def get_embedding(lst_text):
            indices = []
            for word in lst_text:
                if word in self.entity2id.keys():
                    indices.append(self.ent_embeddings[int(self.entity2id[word])])
                else:
                    indices.append(np.zeros(300, dtype='float32'))
            return indices

        return get_embedding

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
        args_batch = []
        batch_original = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            args_batch.append(self.samples_args[_index])
            batch_original.append(self.samples_original[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        label, string1, string2 = tuple(zip(*batch))
        _, arg1, arg2 = tuple(zip(*args_batch))

        # get the length of each seq in your batch fixed by self.max_length
        seq_lengths = torch.from_numpy(np.full((len(string1),), self.max_length))

        # dump padding everywhere, and place seqs on the left.
        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor_arg1 = torch.zeros((len(string1), seq_lengths.max())).long()  # [32, 80]
        seq_tensor_arg2 = torch.zeros((len(string2), seq_lengths.max())).long()  # [32, 80]

        transE_tensor_arg1 = torch.zeros((len(string1), seq_lengths.max(), 300)).float()  # [32, 80, 300]
        transE_tensor_arg2 = torch.zeros((len(string2), seq_lengths.max(), 300)).float()  # [32, 80, 300]

        for idx, (seq1, seq2, tr_arg1, tr_arg2, seqlen) in enumerate(zip(string1, string2, arg1, arg2, seq_lengths)):
            # seq1 [17] tr_arg1 [17,300] 每个循环是一句话 seq1 有17个词
            if len(seq1) > self.max_length:
                seq1 = seq1[:self.max_length]
                tr_arg1 = tr_arg1[:self.max_length]
            if len(seq2) > self.max_length:
                seq2 = seq2[:self.max_length]
                tr_arg2 = tr_arg2[:self.max_length]

            if len(seq1):
                seq_tensor_arg1[idx, :len(seq1)] = torch.LongTensor(seq1)  # [17]
            if len(seq2):
                seq_tensor_arg2[idx, :len(seq2)] = torch.LongTensor(seq2)

            if len(tr_arg1):
                tr_arg1_tensor = torch.from_numpy(np.array(tr_arg1))  # [17, 300]
                for i, item in enumerate(transE_tensor_arg1[idx]):
                    if i >= len(tr_arg1_tensor):
                        break
                    transE_tensor_arg1[idx][i] = tr_arg1_tensor[i]

            if len(tr_arg2):
                tr_arg2_tensor = torch.from_numpy(np.array(tr_arg2))
                for i, item in enumerate(transE_tensor_arg2[idx]):
                    if i >= len(tr_arg2_tensor):
                        break
                    transE_tensor_arg2[idx][i] = tr_arg2_tensor[i]

        # SORT YOUR TENSORS BY LENGTH!
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        seq_tensor_arg1 = seq_tensor_arg1[perm_idx]
        seq_tensor_arg2 = seq_tensor_arg2[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        label = torch.LongTensor(label)
        label = label[perm_idx]

        # seq_tensor.shape is [batch_size x max_length], label.shape is [batch_size], seq_lengths.shape is [batch_size]
        return seq_tensor_arg1, seq_tensor_arg2, label, seq_lengths, transE_tensor_arg1, transE_tensor_arg2, batch_original

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
