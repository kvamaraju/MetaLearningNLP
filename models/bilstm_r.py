from typing import Iterable
import torch
import torch.nn as nn
from torch.autograd import Variable

from functions.gradientreversal import GradientReversalFunction

from copy import deepcopy

import numpy as np


# Based on "Shortcut-Stacked Sentence Encoders for Multi-Domain Inference"
# (https://arxiv.org/abs/1708.02312)
# (https://github.com/easonnie/ResEncoder/blob/master/model/res_encoder.py)
class ResBiLSTMR(nn.Module):
    def __init__(self,
                 embedding_matrix: np.ndarray = None,
                 num_embeddings: int = 196245,
                 embedding_dim: int = 300,
                 hidden_dim: Iterable = (600, 600, 600),
                 mlp_dim: int = 800,
                 dropout_prob: float = 0.1,
                 max_sequence_len: int = 60):
        super(ResBiLSTMR, self).__init__()

        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        if embedding_matrix is not None:
            self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding_layer.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim[0],
                            num_layers=1, bidirectional=True)

        self.lstm_1 = nn.LSTM(input_size=(embedding_dim + hidden_dim[0] * 2), hidden_size=hidden_dim[1],
                              num_layers=1, bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=(embedding_dim + hidden_dim[0] * 2), hidden_size=hidden_dim[2],
                              num_layers=1, bidirectional=True)

        self.max_sequence_len = max_sequence_len

        self.mlp_1 = nn.Linear(hidden_dim[2] * 2 * 4, mlp_dim)
        self.final_layer = nn.Linear(mlp_dim, 3)
        self.domain_layer = nn.Linear(mlp_dim, 2)

        self.classifier = nn.Sequential(*[self.mlp_1,
                                          nn.ReLU(),
                                          nn.Dropout(dropout_prob)])

    @staticmethod
    def pack_for_rnn_seq(inputs: torch.Tensor,
                         lengths: torch.Tensor) -> (torch.Tensor, list):
        _, sorted_indices = lengths.sort()
        r_index = reversed(list(sorted_indices))

        s_inputs_list = []
        lengths_list = []
        reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)

        for j, i in enumerate(r_index):
            s_inputs_list.append(inputs[:, i, :].unsqueeze(1))
            lengths_list.append(lengths[i])
            reverse_indices[i] = j

        reverse_indices = list(reverse_indices)

        s_inputs = torch.cat(s_inputs_list, 1)
        packed_seq = nn.utils.rnn.pack_padded_sequence(s_inputs, lengths_list)

        return packed_seq, reverse_indices

    @staticmethod
    def unpack_from_rnn_seq(packed_seq: torch.Tensor,
                            reverse_indices: list) -> torch.Tensor:
        unpacked_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_seq)
        s_inputs_list = []

        for i in reverse_indices:
            s_inputs_list.append(unpacked_seq[:, i, :].unsqueeze(1))
        return torch.cat(s_inputs_list, 1)

    @staticmethod
    def max_along_time(inputs: torch.Tensor,
                       lengths: torch.Tensor) -> torch.Tensor:
        ls = list(lengths)

        b_seq_max_list = []
        for i, l in enumerate(ls):
            seq_i = inputs[:l, i, :]
            seq_i_max, _ = seq_i.max(dim=0)
            seq_i_max = seq_i_max.squeeze()
            b_seq_max_list.append(seq_i_max)

        return torch.stack(b_seq_max_list)

    def auto_rnn_bilstm(self,
                        lstm: nn.LSTM,
                        seqs: torch.Tensor,
                        lengths: torch.Tensor):
        batch_size = seqs.size(1)
        state_shape = lstm.num_layers * 2, batch_size, lstm.hidden_size
        h0 = c0 = Variable(seqs.data.new(*state_shape).zero_())
        packed_pinputs, r_index = self.pack_for_rnn_seq(seqs, lengths)
        output, _= lstm(packed_pinputs, (h0, c0))
        return self.unpack_from_rnn_seq(output, r_index)

    def forward(self,
                s1: torch.Tensor,
                s2: torch.Tensor,
                l1: torch.Tensor,
                l2: torch.Tensor,
                alpha: np.float32 = -1.0) -> (torch.Tensor, torch.Tensor):
        if self.max_sequence_len:
            if isinstance(l1, int):
                l1 = torch.from_numpy(np.array([l1 for _ in range(s1.shape[0])], dtype=int))
                l2 = torch.from_numpy(np.array([l2 for _ in range(s2.shape[0])], dtype=int))
                if torch.cuda.is_available():
                    l1 = l1.cuda()
                    l2 = l2.cuda()
            l1 = l1.clamp(max=self.max_sequence_len)
            l2 = l2.clamp(max=self.max_sequence_len)
            if s1.size(0) > self.max_sequence_len:
                s1 = s1[:, :self.max_sequence_len]
            if s2.size(0) > self.max_sequence_len:
                s2 = s2[:, :self.max_sequence_len]

        s1_ = self.embedding_layer(s1).permute(1, 0, 2)
        s2_ = self.embedding_layer(s2).permute(1, 0, 2)

        s1_layer1_out = self.auto_rnn_bilstm(self.lstm, s1_, l1)
        s2_layer1_out = self.auto_rnn_bilstm(self.lstm, s2_, l2)

        s1_ = s1_[:s1_layer1_out.size(0), :, :]
        s2_ = s2_[:s2_layer1_out.size(0), :, :]

        s1_layer2_in = torch.cat([s1_, s1_layer1_out], dim=2)
        s2_layer2_in = torch.cat([s2_, s2_layer1_out], dim=2)

        s1_layer2_out = self.auto_rnn_bilstm(self.lstm_1, s1_layer2_in, l1)
        s2_layer2_out = self.auto_rnn_bilstm(self.lstm_1, s2_layer2_in, l2)

        s1_layer3_in = torch.cat([s1_, s1_layer1_out + s1_layer2_out], dim=2)
        s2_layer3_in = torch.cat([s2_, s2_layer1_out + s2_layer2_out], dim=2)

        s1_layer3_out = self.auto_rnn_bilstm(self.lstm_2, s1_layer3_in, l1)
        s2_layer3_out = self.auto_rnn_bilstm(self.lstm_2, s2_layer3_in, l2)

        s1_layer3_maxout = self.max_along_time(s1_layer3_out, l1)
        s2_layer3_maxout = self.max_along_time(s2_layer3_out, l2)

        features = torch.cat([s1_layer3_maxout,
                              s2_layer3_maxout,
                              torch.abs(s1_layer3_maxout - s2_layer3_maxout),
                              s1_layer3_maxout * s2_layer3_maxout], dim=1)

        classifier = self.classifier(features)

        return self.final_layer(classifier), self.domain_layer(GradientReversalFunction.apply(classifier, alpha))

    def load_params(self,
                    params: list):
        for p, avg_p in zip(self.parameters(), params):
            p.data = deepcopy(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params
