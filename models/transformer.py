import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


# Based on "A Decomposable Attention Model for Natural Language Inference" (https://arxiv.org/abs/1606.01933)
class Transformer(nn.Module):
    def __init__(self,
                 embedding_matrix: np.ndarray = None,
                 num_classes: int = 3,
                 num_embeddings: int = 196245,
                 embedding_dim: int = 300,
                 hidden_dim: int = 200):
        super(Transformer, self).__init__()

        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        if embedding_matrix is not None:
            self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding_layer.weight.requires_grad = False

        self.input_layer = nn.Linear(self.embedding_dim, self.hidden_dim, bias=False)

        self.mlp_f = self._mlp_layers(input_dim=self.hidden_dim, output_dim=self.hidden_dim)
        self.mlp_g = self._mlp_layers(input_dim=2 * self.hidden_dim, output_dim=self.hidden_dim)
        self.mlp_h = self._mlp_layers(input_dim=2 * self.hidden_dim, output_dim=self.hidden_dim)
        self.output = self.final_linear = nn.Linear(self.hidden_dim, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.normal_(std=1e-2)

    @staticmethod
    def _mlp_layers(input_dim: int,
                    output_dim: int):
        layers = [nn.Dropout(p=0.2),
                  nn.Linear(input_dim, output_dim),
                  nn.ReLU(),
                  nn.Dropout(p=0.2),
                  nn.Linear(output_dim, output_dim),
                  nn.ReLU()]
        return nn.Sequential(*layers)

    def forward(self,
                s1: torch.Tensor,
                s2: torch.Tensor,
                l1: torch.Tensor,
                l2: torch.Tensor):
        s1_ = self.input_layer(self.embedding_layer(s1))
        s2_ = self.input_layer(self.embedding_layer(s2))

        len_s1 = s1_.size(1)
        len_s2 = s2_.size(1)

        f1_ = self.mlp_f(s1_)
        f2_ = self.mlp_f(s2_)

        score1_ = torch.bmm(f1_, torch.transpose(f2_, 1, 2))
        p1_ = F.softmax(score1_.view(-1, len_s2), dim=0).view(-1, len_s1, len_s2)

        score2_ = torch.transpose(score1_.contiguous(), 1, 2).contiguous()
        p2_ = F.softmax(score2_.view(-1, len_s1), dim=0).view(-1, len_s2, len_s1)

        g1_ = self.mlp_g(torch.cat((s1_, torch.bmm(p1_, s2_)), 2))
        g2_ = self.mlp_g(torch.cat((s2_, torch.bmm(p2_, s1_)), 2))

        s1_out = g1_.sum(1).squeeze(1)
        s2_out = g2_.sum(1).squeeze(1)

        h = self.mlp_h(torch.cat((s1_out, s2_out), 1))
        return self.final_linear(h)

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data = deepcopy(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params
