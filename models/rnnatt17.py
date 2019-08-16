import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNATT17(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_output,
                 rnn_model='LSTM',
                 use_last=True,
                 embedding_tensor=None,
                 padding_index=0,
                 hidden_size=300,
                 num_layers=1,
                 batch_first=True):
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

        super(RNNATT17, self).__init__()
        self.use_last = use_last

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

        # Attention layer
        self.att = nn.Linear(embed_size, 1)

        # 4 represents the number of categories
        self.proj2class = nn.Linear(embed_size, num_output)

    def forward(self, x, seq_lengths):
        '''
        Args:
            x: (batch, input_size)

        Returns:
            num_output size
        '''

        batch_size = len(seq_lengths)  # N 128
        x_embed = self.encoder(x)
        x_embed = self.drop_en(x_embed)
        hids, _ = self.rnn(x_embed)  # [N, 256, 300 x 2]

        l2r_hids = hids.view(
            batch_size,
            256,
            300,
            2
        )[:, :, :, 0]  # [N, 256, 300]

        r2l_hids = hids.view(
            batch_size,
            256,
            300,
            2
        )[:, :, :, 1]  # [N, 256, 300]

        hids_sum = l2r_hids + r2l_hids  # [N, 256, 300]
        hids_sum_activated = torch.tanh(hids_sum)  # [N, 256, 300]
        hids_proj = self.att(hids_sum_activated).squeeze(2)  # [N, 256]
        alpha = F.softmax(hids_proj, dim=1)  # [N, 256], dim = 0 Softmax for each column, dim = 1 Softmax for each line

        r = torch.bmm(
            hids_sum.transpose(1, 2),
            alpha.unsqueeze(2)
        ).squeeze(2)  # [N, 300]

        unnormalized_prob = self.proj2class(r)  # [N, C]
        logprob = F.log_softmax(unnormalized_prob, dim=1)

        return logprob
