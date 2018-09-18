import numpy as np

import torch
import torch.nn as nn

from copy import deepcopy


class MLP(nn.Module):
    def __init__(self,
                 embedding_matrix: np.ndarray = None,
                 num_classes: int = 3,
                 num_embeddings: int = 196245,
                 embedding_dim: int = 300,
                 hidden_dim: int = 300):
        super(MLP, self).__init__()

        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        if embedding_matrix is not None:
            self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding_layer.weight.requires_grad = False

        layers = [nn.Linear(in_features=4*self.embedding_dim, out_features=hidden_dim), nn.ReLU(),
                  nn.Linear(in_features=hidden_dim, out_features=hidden_dim), nn.ReLU(),
                  nn.Linear(in_features=hidden_dim, out_features=hidden_dim), nn.ReLU(),
                  nn.Dropout(0.1),
                  nn.Linear(in_features=hidden_dim, out_features=num_classes)]

        self.output = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.normal_(std=1e-2)

    def forward(self,
                s1: torch.Tensor,
                s2: torch.Tensor,
                l1: torch.Tensor,
                l2: torch.Tensor):
        s1_ = nn.Dropout(0.1)(self.embedding_layer(s1)).sum(dim=1)
        s2_ = nn.Dropout(0.1)(self.embedding_layer(s2)).sum(dim=1)
        input_ = torch.cat([s1_, s2_, s1_.sub(s2_), s1_.mul(s2_)], dim=1)
        return self.output(input_)

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data = deepcopy(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params
