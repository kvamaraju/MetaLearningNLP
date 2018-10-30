import copy
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# This code is based on https://github.com/huggingface/pytorch-openai-transformer-lm


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return 1.78718727865 * (x * torch.sigmoid(x) - 0.20662096414)


class LayerNorm(nn.Module):
    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        assert rf == 1
        self.rf = rf
        self.nf = nf

        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.w = Parameter(w)
        self.b = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
        return x.view(*size_out)


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, n_head=12, attn_pdrop=0.1, resid_pdrop=0.1, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        assert n_state % n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b - 1e9 * (1 - self.b)
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return self.resid_dropout(a)


class MLP(nn.Module):
    def __init__(self, n_state, embedding_dim=768, resid_pdrop=0.1):
        super(MLP, self).__init__()
        nx = embedding_dim
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, embedding_dim=768, n_head=12, attn_pdrop=0.1, resid_pdrop=0.1, scale=False):
        super(Block, self).__init__()
        nx = embedding_dim
        self.attn = Attention(nx=nx, n_ctx=n_ctx, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, scale=scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(n_state=4 * nx, embedding_dim=embedding_dim, resid_pdrop=resid_pdrop)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        return self.ln_2(n + m)


class TransformerModel(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray = None, embedding_dim=768, n_head=12, n_layer=12, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1, num_embeddings=40990, n_ctx=512):
        super(TransformerModel, self).__init__()
        self.num_embeddings = num_embeddings

        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        if embedding_matrix is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embed.weight.requires_grad = False

        self.drop = nn.Dropout(embd_pdrop)
        block = Block(n_ctx=n_ctx, embedding_dim=embedding_dim, n_head=n_head, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])

        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x):
        x = x.view(-1, x.size(-2), x.size(-1))
        e = self.embed(x)
        h = e.sum(dim=2)
        for block in self.h:
            h = block(h)
        return h


class ClfHead(nn.Module):
    def __init__(self, clf_token, n_class, embedding_dim=768, clf_pdrop=0.1):
        super(ClfHead, self).__init__()
        self.embedding_dim = embedding_dim
        self.clf_token = clf_token
        self.dropout = nn.Dropout(clf_pdrop)
        self.linear = nn.Linear(embedding_dim, n_class)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        clf_h = h.view(-1, self.embedding_dim)
        flat = x[..., 0].contiguous().view(-1)
        clf_h = clf_h[flat == self.clf_token, :]
        clf_h = self.dropout(clf_h)
        return self.linear(clf_h)


class Transformer2(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, clf_token, embedding_dim=768, n_head=12, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1, clf_pdrop=0.1, num_embeddings=40990, seq_len=120, n_ctx=100):
        super(Transformer2, self).__init__()
        self.num_embeddings = num_embeddings
        self.seq_len = seq_len
        self.transformer = TransformerModel(embedding_matrix=embedding_matrix,
                                            embedding_dim=embedding_dim,
                                            n_head=n_head,
                                            embd_pdrop=embd_pdrop,
                                            attn_pdrop=attn_pdrop,
                                            resid_pdrop=resid_pdrop,
                                            num_embeddings=num_embeddings,
                                            n_ctx=n_ctx)
        self.task_head = ClfHead(clf_token=clf_token, n_class=3, embedding_dim=embedding_dim, clf_pdrop=clf_pdrop)

    def forward(self,
                s1: torch.Tensor,
                s2: torch.Tensor,
                l1: torch.Tensor,
                l2: torch.Tensor):

        if torch.cuda.is_available():
            x = torch.stack([torch.cat((s1[i, :l1[i]],
                                        s2[i, 1:l2[i] - 1],
                                        torch.from_numpy(np.array([self.num_embeddings - 1 for _ in range(self.seq_len + 2 - l1[i] - l2[i])])).cuda()), 0)
                             for i in range(s1.shape[0])])
        else:
            x = torch.stack([torch.cat((s1[i, :l1[i]],
                                        s2[i, 1:l2[i] - 1],
                                        torch.from_numpy(np.array([self.num_embeddings - 1 for _ in range(self.seq_len + 2 - l1[i] - l2[i])]))), 0)
                             for i in range(s1.shape[0])])
        h = self.transformer(x)
        return self.task_head(h, x)
