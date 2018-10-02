import os
import shutil
import torch

import numpy as np

from models.mlp import MLP
from models.transformer import Transformer
from models.bilstm import ResBiLSTM


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor,
             target: torch.Tensor,
             topk: tuple = (1,)) -> list:
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_directory(name: str,
                  type_net: str) -> str:
    name += f'_{type_net}'
    directory = f'logs/{name}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def save_checkpoint(state: object,
                    is_best: bool,
                    name: str,
                    filename: str):
    directory = f'runs/{name}/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, filename)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'runs/{name}/model_best.pth.tar')


def resume_from_checkpoint(resume_path: str,
                           model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer) -> (int, float, int, torch.nn.Module, torch.optim.Optimizer):
    if os.path.isfile(resume_path):
        print(f"=> loading checkpoint '{resume_path}'")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{resume_path}' (epoch {checkpoint['epoch']})")

        return checkpoint['epoch'], checkpoint['best_prec1'], checkpoint['total_steps'], model, optimizer

    return 0, 0., 0, model, optimizer


def construct_optimizer(optimizer: str,
                        model: torch.nn.Module,
                        lr: float) -> torch.optim.Optimizer:
    if optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    elif optimizer == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer == 'adamax':
        return torch.optim.Adamax(model.parameters(), lr=lr)
    elif optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer == 'rprop':
        return torch.optim.Rprop(model.parameters(), lr=lr)
    elif optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        return torch.optim.Adadelta(model.parameters(), lr=lr)


def construct_model(model_type: str,
                    weight_matrix: np.ndarray) -> torch.nn.Module:
    if model_type == 'mlp':
        model = MLP(num_embeddings=weight_matrix.shape[0],
                    embedding_matrix=weight_matrix)
    elif model_type == 'transformer':
        model = Transformer(num_embeddings=weight_matrix.shape[0],
                            embedding_matrix=weight_matrix)
    elif model_type == 'lstm':
        model = ResBiLSTM(num_embeddings=weight_matrix.shape[0],
                          embedding_matrix=weight_matrix)
    else:
        model = None
    return model
