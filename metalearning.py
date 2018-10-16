from collections import OrderedDict
from typing import Iterable

import torch
import torch.nn as nn

import numpy as np

from distributed import reduce_gradients


class ClassifierTask(nn.Module):
    def __init__(self):
        super(ClassifierTask, self).__init__()
        if torch.cuda.is_available():
            self.loss_fn = torch.nn.CrossEntropyLoss().cuda()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self,
                module: torch.nn.Module,
                inputs: tuple,
                **kwargs):

        s1 = inputs[0]
        s2 = inputs[1]
        target = inputs[2]
        l1 = inputs[3]
        l2 = inputs[4]

        if torch.cuda.is_available():
            s1 = s1.cuda()
            s2 = s2.cuda()
            target = target.cuda()
            l1 = l1.cuda()
            l2 = l2.cuda()

        output = module(s1, s2, l1, l2, **kwargs)
        return self.loss_fn(output, target)


class GradientTaskModule(nn.Module):
    def __init__(self):
        super(GradientTaskModule, self).__init__()

    def forward(self,
                task: ClassifierTask,
                module: torch.nn.Module,
                input_batch: Iterable,
                params_for_grad: list,
                compute_grad: bool = True,
                create_graph: bool = False,
                gradient_average: float = 1., ):

        criterion = task(module=module,
                         inputs=input_batch)
        if compute_grad:
            return torch.autograd.grad(criterion / gradient_average, params_for_grad,
                                       create_graph=create_graph)


class MAML(nn.Module):
    def __init__(self,
                 module: torch.nn.Module,
                 inner_lr: float,
                 second_order: bool = False,
                 distributed: bool = False):
        super(MAML, self).__init__()
        self.module = module
        self.inner_lr = inner_lr
        self.second_order = second_order
        self.gradient_task = GradientTaskModule()
        self.distributed = distributed
        if self.distributed:
            for p in self.module.state_dict().values():
                if torch.is_tensor(p):
                    torch.distributed.broadcast(p, 0)

    def forward(self,
                tasks: list,
                train_batch: list,
                val_loaders: list):

        meta_grad_dict = grad_dict(self.module, clone=True)
        meta_param_dict = param_dict(self.module, clone=True)

        assert len(tasks) == len(train_batch) == len(val_loaders)

        for i, _ in enumerate(train_batch):
            load_param_dict(self.module, meta_param_dict)
            params_for_grad = get_params_for_grad(self.module)

            self.inner_loop(task=tasks[i],
                            loader=train_batch[i],
                            params_for_grad=params_for_grad)

            self.outer_loop(task=tasks[i],
                            loader=val_loaders[i],
                            params_for_grad=params_for_grad)

            update_grad_dict(module=self.module,
                             grad_state_dict=meta_grad_dict)

        if self.distributed:
            reduce_gradients(self.module)

    def inner_loop(self,
                   task: ClassifierTask,
                   loader: Iterable,
                   params_for_grad: list):
        training_status = self.training
        self.module.train()
        for input_batch in loader:
            with torch.enable_grad():
                grads = self.gradient_task(task=task,
                                           module=self.module,
                                           input_batch=input_batch,
                                           params_for_grad=params_for_grad,
                                           create_graph=self.second_order and self.training)
                new_params = grad_step_params(params=params_for_grad,
                                              grads=grads,
                                              lr=self.inner_lr,
                                              inplace=False)
            set_params_with_grad(module=self.module,
                                 params=new_params)
        self.module.train(training_status)

    def outer_loop(self,
                   task: ClassifierTask,
                   loader: Iterable,
                   params_for_grad: list):
        if not self.second_order:
            params_for_grad = get_params_for_grad(module=self.module)

        for batch in loader:
            grads = self.gradient_task(task=task,
                                       module=self.module,
                                       input_batch=batch,
                                       params_for_grad=params_for_grad,
                                       compute_grad=self.training,
                                       gradient_average=len(batch[0]))
            set_grads_for_params(params=params_for_grad,
                                 grads=grads)
            set_params_with_grad(module=self.module,
                                 params=params_for_grad)


class Reptile(nn.Module):
    def __init__(self,
                 module: torch.nn.Module,
                 inner_lr: float,
                 second_order: bool = False,
                 distributed: bool = False):
        super(Reptile, self).__init__()
        self.module = module
        self.inner_lr = inner_lr
        self.second_order = second_order
        self.gradient_task = GradientTaskModule()
        self.distributed = distributed
        if self.distributed:
            for p in self.module.state_dict().values():
                if torch.is_tensor(p):
                    torch.distributed.broadcast(p, 0)

    def forward(self,
                tasks: list,
                train_batch: list,
                val_loaders: list):

        assert len(tasks) == len(train_batch)

        i = np.random.choice(range(len(tasks)))

        params_for_grad = get_params_for_grad(self.module)
        new_params = self.inner_loop(task=tasks[i],
                                     loader=train_batch[i],
                                     params_for_grad=params_for_grad)

        grads = [x - y for x, y in zip(params_for_grad, new_params)]

        set_grads_for_params(params=params_for_grad, grads=grads)

        if self.distributed:
            reduce_gradients(self.module)

    def inner_loop(self,
                   task: ClassifierTask,
                   loader: Iterable,
                   params_for_grad: list) -> list:

        training_status = self.training
        self.module.train()

        for input_batch in loader:
            with torch.enable_grad():
                grads = self.gradient_task(task=task,
                                           module=self.module,
                                           input_batch=input_batch,
                                           params_for_grad=params_for_grad,
                                           create_graph=self.second_order and self.training)
                new_params = grad_step_params(params=params_for_grad,
                                              grads=grads,
                                              lr=self.inner_lr,
                                              inplace=False)
            set_params_with_grad(module=self.module,
                                 params=new_params)
        self.module.train(training_status)
        return new_params


class MetaTrainWrapper(nn.Module):
    def __init__(self,
                 module: nn.Module,
                 inner_lr: float,
                 use_maml: bool = True,
                 optim: torch.optim.Optimizer = None,
                 second_order: bool = False,
                 distributed: bool = False,
                 world_size: int = 1,
                 rank: int = -1):
        super(MetaTrainWrapper, self).__init__()
        self.module = module
        self.optim = optim
        self.distributed = distributed
        self.init_distributed(world_size, rank)

        if use_maml:
            self.meta_module = MAML(module=self.module,
                                    inner_lr=inner_lr,
                                    second_order=second_order)
        else:
            self.meta_module = Reptile(module=self.module,
                                       inner_lr=inner_lr,
                                       second_order=second_order)

    def train(self, mode=True):
        assert self.optim is not None
        return super(MetaTrainWrapper, self).train(mode)

    def update_lr(self, lr: float):
        for group in self.optim.param_groups:
            group['lr'] = lr

    def forward(self,
                tasks: list,
                train_batch: list,
                val_loaders: list):
        for group in self.optim.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = p.grad.data.contiguous()
        self.optim.zero_grad()

        if self.training:
            self.meta_module(tasks=tasks,
                             train_batch=train_batch,
                             val_loaders=val_loaders)
            self.optim.step()
        else:
            with torch.no_grad():
                self.meta_module(tasks=tasks,
                                 train_loaders=train_batch,
                                 val_loaders=val_loaders)

    def init_distributed(self, world_size=1, rank=-1):
        if self.distributed:
            torch.distributed.init_process_group(backend='gloo', world_size=world_size,
                                                 init_method='file://distributed.dpt', rank=rank)


def transpose_list(list_of_lists: list) -> list:
    return [tuple(zip(*x)) if isinstance(x[0], tuple) else list(x) for x in zip(*list_of_lists)]


def reduce_sum(tens: list) -> float:
    t_sum = 0
    for t in tens:
        if len(t.size()) >= 1:
            if len(t.size()) > 1 or t.size(0) != 0:
                t = t.sum()
        t_sum += t
    return t_sum


def mean(lst: list) -> float:
    length = float(len(lst))
    return reduce_sum(lst) / length


def param_dict(self: torch.nn.Module,
               destination: OrderedDict = None,
               prefix: str = '',
               clone: bool = False) -> OrderedDict:
    if destination is None:
        destination = OrderedDict()
    for name, param in self._parameters.items():
        if param is not None:
            destination[prefix + name] = param.detach().data.clone() if clone else param.data
    for name, module in self._modules.items():
        if module is not None:
            param_dict(self=module,
                       destination=destination,
                       prefix=prefix + name + '.',
                       clone=clone)
    return destination


def load_param_dict(module: torch.nn.Module,
                    copy: OrderedDict,
                    strict: bool = True):
    own_state = param_dict(module)
    for name, param in copy.items():
        if name in own_state:
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError(f'While copying the parameter named {name}, '
                                   f'whose dimensions in the model are {own_state[name].size()} and '
                                   f'whose dimensions in the checkpoint are {param.size()}.')
        elif strict:
            raise KeyError(f'unexpected key "{name}" in state_dict')
    if strict:
        missing = set(own_state.keys()) - set(copy.keys())
        if len(missing) > 0:
            raise KeyError(f'missing keys in state_dict: "{missing}"')


def grad_dict(self: torch.nn.Module,
              destination: OrderedDict = None,
              prefix: str = '',
              clone: bool = False) -> OrderedDict:
    if destination is None:
        destination = OrderedDict()
    for name, param in self._parameters.items():
        if param is not None:
            if param.grad is not None:
                destination[prefix + name] = param.grad.detach().data.clone() if clone else param.grad.data
            else:
                destination[prefix + name] = None
    for name, module in self._modules.items():
        if module is not None:
            grad_dict(self=module,
                      destination=destination,
                      prefix=prefix + name + '.',
                      clone=clone)
    return destination


def update_grad_dict(module: torch.nn.Module,
                     grad_state_dict: OrderedDict,
                     strict: bool = True,
                     accumulate: bool = True):
    own_state = grad_dict(module)
    for name, grad in own_state.items():
        if grad is None:
            continue
        if name in grad_state_dict:
            grad2update = grad_state_dict[name]
            try:
                if grad2update is None:
                    if grad is None:
                        continue
                    grad_state_dict[name] = grad.clone()
                else:
                    if accumulate:
                        grad2update.add_(grad)
                    else:
                        grad2update.copy_(grad)
            except Exception:
                raise RuntimeError(f'While copying the parameter named {name}, '
                                   f'whose dimensions in the fine tuned model are {grad.size()} and '
                                   f'whose dimensions in the meta model are {grad_state_dict[name].size()}.')
        elif strict:
            raise KeyError(f'unexpected key "{name}" in fine tuned model')


def load_grad_dict(module: torch.nn.Module,
                   gradient_dict: OrderedDict,
                   strict: bool = True):
    own_state = grad_dict(module)
    for name, grad in gradient_dict.items():
        if grad is None:
            continue
        if name in own_state:
            try:
                grad2update = own_state[name]
                if grad2update is None:
                    continue
                else:
                    grad2update.copy_(grad)

            except Exception:
                raise RuntimeError(f'While copying the parameter named {name}, '
                                   f'whose dimensions in the model are {own_state[name].size()} and '
                                   f'whose dimensions in the checkpoint are {grad.size()}.')
        elif strict:
            raise KeyError(f'unexpected key "{name}" in gradient_dict')
    if strict:
        missing = set(own_state.keys()) - set(gradient_dict.keys())
        if len(missing) > 0:
            raise KeyError(f'missing keys in gradient_dict: "{missing}"')


def grad_step_params(params: list,
                     grads: list,
                     lr: float,
                     inplace: bool = False) -> list:
    if inplace:
        for p, g in zip(params, grads):
            if g is None:
                continue
            p.data.add_(-lr * g.data)
        return []
    return [p - lr * g if g is not None else p for (p, g) in zip(params, grads)]


def set_grads_for_params(params: list,
                         grads: list):
    for p, g in zip(params, grads):
        if g is None:
            continue
        p.grad = g.contiguous()


def get_params_for_grad(module: torch.nn.Module,
                        destination: list = None) -> list:
    if destination is None:
        destination = []

    for _, p in module.named_parameters():
        if p is not None and p.requires_grad:
            destination.append(p)

    return destination


def set_params_with_grad(module: torch.nn.Module,
                         params: list):
    j = 0
    for _, (_, param) in enumerate(module.named_parameters()):
        if param is not None and param.requires_grad:
            param = params[j]
            j += 1


def zero_grad(optimizer: torch.optim.Optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad = p.grad.detach()
                p.grad.zero_()
