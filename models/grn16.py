import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN16(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_output=2,
                 rnn_model='LSTM',
                 use_last=True,
                 embedding_tensor=None,
                 padding_index=0,
                 hidden_size=300,
                 num_layers=1,
                 batch_first=True,
                 tensor_slice=2):
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

        super(GRN16, self).__init__()
        self.use_last = use_last
        self.r = tensor_slice

        # embedding
        self.encoder = None
        if torch.is_tensor(embedding_tensor):
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index, _weight=embedding_tensor)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)

        self.drop_en = nn.Dropout(p=0.6)

        # rnn module
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                               batch_first=True, bidirectional=True)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                              batch_first=True, bidirectional=True)
        else:
            raise LookupError(' only support LSTM and GRU')

        self.gate = nn.Linear(200,
                              self.r)

        self.H = nn.Bilinear(
            100,
            100,
            self.r,
            bias=False
        )

        self.V = nn.Linear(
            200,
            self.r,
            bias=False

        )
        self.b = nn.Parameter(torch.zeros(1, 2))

        self.v = nn.Linear(self.r, 1)

        self.maxpool2d = nn.MaxPool2d([3, 3])
        self.linear1 = nn.Linear(16 * 16, hidden_size)  # [50, 50] --> (3, 3) maxpool --> [16, 16]
        self.linear2 = nn.Linear(hidden_size, num_output)

    def forward(self, input, seq_lengths):
        """

        Args
        ----------
        input   : python list
            input[0] is arg1, input[1] is arg2

        Returns
        ----------
        logprob : N x 1

        """
        arg1, arg2 = input  # N = 32 arg1.shape is [N, 50], arg2.shape is [N, 50]

        arg1_emb = self.encoder(arg1)  # [N, 50, 50]
        arg1_emb = self.drop_en(arg1_emb)  # [N, 50, 50]

        arg2_emb = self.encoder(arg2)  # [N, 50, 50]
        arg2_emb = self.drop_en(arg2_emb)  # [N, 50, 50]

        arg1_hid, _ = self.rnn(arg1_emb)  # [N, 50, 50 * 2]
        arg2_hid, _ = self.rnn(arg2_emb)  # [N, 50, 50 * 2]

        cross_hid = self.construct_cross_hid(
            arg1_hid,
            arg2_hid
        )  # [N, 50, 50, 200]

        gate = torch.sigmoid(self.gate(cross_hid))  # [N, 50, 50, r]
        one_minus_gate = 1 - gate

        bilinear_out = self.H(
            cross_hid[:, :, :, :100].contiguous().view(-1, 100),
            cross_hid[:, :, :, 100:].contiguous().view(-1, 100)
        ).view(seq_lengths.size(0), 50, 50, self.r)  # [N, 50, 50, r]

        singler_layer_out = torch.tanh(self.V(cross_hid))  # [N, 50, 50, r]

        s = gate * bilinear_out + one_minus_gate * singler_layer_out + self.b  # [N, 50, 50, r]
        s = self.v(s)  # [N, 50, 50, 1]
        s = s.transpose(1, 3).contiguous()  # [N, 1, 50, 50]

        s_pooled = self.maxpool2d(s)  # [N, 1, 16, 16]
        s_linear = s_pooled.view(-1, 16 * 16)  # [N, 256]

        linear1_out = torch.tanh(self.linear1(s_linear))  # [N, 50]
        logprob = F.log_softmax(self.linear2(linear1_out), dim=1)  # [N, 4]

        return logprob

    def construct_cross_hid(self, arg1_hid, arg2_hid):
        """Construct cross concatenation of hiddens:
           that is torch.cat([arg1_hid[i], arg2_hid[j]], dim=2)

        Args
        ----------
        arg1_hid  : torch.FloatTensor
            [N, 50, 100]
        arg2_hid  : torch.FloatTensor
            [N, 50, 100]

        Returns
        ----------
        cross_hid : torch.FloatTensor
            [N, 50, 50, 200]

        """

        N = arg1_hid.size(0)

        ## RuntimeError: in-place operatioins can be only used on variables that
        ## don't share storage with any other variables, but detected that there
        ## are 2 objects sharing it
        # cross_hid = Variable(torch.zeros(N, 50, 50, 200))

        # for i in range(50):
        # 	for j in range(50):
        # 		hid_cat = torch.cat([arg1_hid[:, i, :], arg2_hid[:, j, :]], dim=1)
        # 		cross_hid[:, i, j, :].copy_(hid_cat)
        hid_cat_stack = []

        for i in range(50):
            arg1_hid_slice = arg1_hid[:, i, :]  # [N, 100]
            arg1_hid_slice_repeat = arg1_hid_slice.repeat(1, 50).contiguous().view(N, 50, 100)
            hid_cat = torch.cat([arg1_hid_slice_repeat, arg2_hid], dim=2)  # [N, 50, 200]
            hid_cat_stack.append(hid_cat)

        cross_hid = torch.stack(hid_cat_stack, dim=1)  # [N, 50, 50, 200]
        return cross_hid
