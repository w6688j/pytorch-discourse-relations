import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor

else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor


class KEANNKG(nn.Module):

    def __init__(self, vocab_size, embed_size, num_output, rnn_model='LSTM', use_last=True, embedding_tensor=None,
                 padding_index=0, hidden_size=64, num_layers=1, batch_first=True, max_length=80):
        """

        Args:
            vocab_size: vocab size
            embed_size: embedding size
            num_output: number of output (classes)
            rnn_model:  LSTM or GRU
            use_last:  bool
            embedding_tensor:
            padding_index:
            hidden_size: hidden size of rnn module
            num_layers:  number of layers in rnn module
            batch_first: batch first option
        """

        super(KEANNKG, self).__init__()
        self.use_last = use_last
        # embedding
        self.encoder = None
        self.embed_size = embed_size
        self.num_output = num_output
        self.max_length = max_length

        word_weight = floatTensor(vocab_size, self.embed_size)
        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform_(word_weight)

        if torch.is_tensor(embedding_tensor):
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index, _weight=embedding_tensor)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)

        self.encoder.weight = nn.Parameter(word_weight)
        self.drop_en = nn.Dropout(p=0)

        # rnn module
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                               batch_first=True, bidirectional=True)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True)
        else:
            raise LookupError(' only support LSTM and GRU')

        self.bn2 = nn.BatchNorm1d(max_length * hidden_size * 4)
        self.fc = nn.Linear(max_length * hidden_size * 4, num_output)

        self.rand_matrix = torch.nn.Parameter(torch.rand(embed_size * 2, embed_size * 2))

    def forward(self, x, seq_lengths, transE_args, cuda):
        '''
        Args:
            x: input[0] is arg1, input[1] is arg2
            input[0]: (batch, max_length)
            input[1]: (batch, max_length)

        Returns:
            num_output size
        '''
        arg1 = x[0]  # [N, arg1_max_length] [128, 80]
        arg2 = x[1]  # [N, arg2_max_length] [128, 80]

        # knowledge-enhance with transE
        self.kg_relation, self.kg_relation_list = self.deal_transE(transE_args, seq_lengths.size(0), seq_lengths[0],
                                                                   cuda)

        arg1_embed = self.encoder(arg1)
        arg1_embed = self.drop_en(arg1_embed)  # [N, arg1_max_length, embed_size] [128, 80, 300]

        arg2_embed = self.encoder(arg2)
        arg2_embed = self.drop_en(arg2_embed)  # [N, arg1_max_length, embed_size] [128, 80, 300]

        out_rnn1, ht = self.rnn(arg1_embed, None)  # [128, 80, 600]
        out_rnn2, ht = self.rnn(arg2_embed, None)  # [128, 80, 600]

        last_tensor1 = out_rnn1.contiguous().view(seq_lengths.size(0) * seq_lengths[0], -1)  # [128 * 80, 600]
        last_tensor2 = out_rnn2.contiguous().view(seq_lengths.size(0) * seq_lengths[0], -1)  # [128 * 80, 600]

        last_tensor = torch.mm(last_tensor1, self.rand_matrix)  # [128 * 80, 600]
        last_tensor = torch.mm(last_tensor, torch.t(last_tensor2))  # [128 * 80, 128 * 80]
        last_tensor = torch.tanh(last_tensor)  # [128 * 80, 128 * 80]

        last_tensor = last_tensor + self.kg_relation  # [128 * 80, 128 * 80] add knowledge
        self.last_tensor = last_tensor

        #  torch.softmax(last_tensor, dim=1) [128 * 80, 128 * 80]
        sf1 = torch.mean(F.softmax(last_tensor, dim=1), dim=0, keepdim=True).view(-1, 1).expand(
            seq_lengths.size(0) * seq_lengths[0],
            self.embed_size * 2)  # 每行相加为1 [1, 128 * 80] -> [128 * 80, 1] -> [128 * 80, 600]

        sf2 = torch.mean(F.softmax(last_tensor, dim=0), dim=1, keepdim=True).expand(
            seq_lengths.size(0) * seq_lengths[0],
            self.embed_size * 2)  # 每列相加为1 [128 * 80, 1] -> [128 * 80, 600]

        out1 = last_tensor1.mul(sf2).view(seq_lengths.size(0), -1,
                                          self.embed_size * 2)  # [128 * 80, 600] -> [128, 80, 600]
        out2 = last_tensor2.mul(sf1).view(seq_lengths.size(0), -1,
                                          self.embed_size * 2)  # [128 * 80, 600] -> [128, 80, 600]

        out = torch.cat((out1, out2), 1).view(seq_lengths.size(0), -1)  # [128, 160, 600] -> [128, 160 * 600]

        fc_input = self.bn2(out)  # [128, 160 * 600]
        out_last = F.log_softmax(self.fc(fc_input), dim=1)  # [128, 4]

        return out_last

    def deal_transE(self, data, batch_size, max_length, cuda):
        transE_arg1 = data[0]  # [N, arg1_max_length, 300] [128, 80, 300]
        transE_arg2 = data[1]  # [N, arg1_max_length] [128, 80, 300]
        kg_relation = Variable(
            torch.zeros((batch_size * max_length, batch_size * max_length)).float(),
            requires_grad=False)  # [128 * 80, 128 * 80]
        relations = []

        if cuda:
            kg_relation = kg_relation.cuda()

        for num in range(batch_size):
            relation = Variable(torch.zeros((max_length, max_length)).float(), requires_grad=False)  # [80, 80]
            if cuda:
                relation = relation.cuda()

            for i, arg1_word in enumerate(transE_arg1):
                for j, arg2_word in enumerate(transE_arg2):
                    if i >= max_length or j >= max_length:
                        break
                    relation[i, j] = torch.tanh(torch.mean(arg1_word - arg2_word))  # 标量
                    relations.append(relation)

            kg_relation[num * max_length:(num + 1) * max_length,
            num * max_length:(num + 1) * max_length] = relation  # [0:80, 0:80] [80:160, 80:160] [160:240, 160:240] ...

        return kg_relation, relations
