import torch.nn as nn
import torch.nn.functional as F


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        return -1.0 * b.mean()
