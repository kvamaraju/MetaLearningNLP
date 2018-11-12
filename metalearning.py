from collections import OrderedDict
from typing import Iterable

from losses.entropy import Entropy

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


class ClassifierTask2(nn.Module):
    def __init__(self):
        super(ClassifierTask2, self).__init__()
        if torch.cuda.is_available():
            self.loss_fn = torch.nn.CrossEntropyLoss().cuda()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        self.regularizer = Entropy()

    def forward(self,
                module: torch.nn.Module,
                inputs: tuple,
                task_number: int = 0,
                alpha: float = 0.1,
                **kwargs):

        s1 = inputs[0]
        s2 = inputs[1]
        target = inputs[2]
        l1 = inputs[3]
        l2 = inputs[4]

        target_ = torch.zeros(s1.size(0), dtype=torch.int64).fill_(int(task_number))

        if torch.cuda.is_available():
            s1 = s1.cuda()
            s2 = s2.cuda()
            target = target.cuda()
            l1 = l1.cuda()
            l2 = l2.cuda()
            target_ = target_.cuda()

        class_output, domain_output = module(s1, s2, l1, l2, **kwargs)
        return self.loss_fn(domain_output, target_), self.loss_fn(class_output, target) - alpha * self.regularizer(domain_output)


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
                gradient_average: float = 1.):

        criterion = task(module=module,
                         inputs=input_batch)
        if compute_grad:
            return torch.autograd.grad(criterion / gradient_average,
                                       params_for_grad,
                                       create_graph=create_graph)


class GradientTaskModule2(nn.Module):
    def __init__(self):
        super(GradientTaskModule2, self).__init__()

    def forward(self,
                task: ClassifierTask2,
                module: torch.nn.Module,
                input_batch: Iterable,
                domain_params_for_grad: list,
                classifier_params_for_grad: list,
                compute_grad: bool = True,
                create_graph: bool = False,
                gradient_average: float = 1.,
                task_number: int = 0):

        domain_criterion, classifier_criterion = task(module=module,
                                                      inputs=input_batch,
                                                      task_number=task_number)
        if compute_grad:
            return torch.autograd.grad(domain_criterion / gradient_average,
                                       domain_params_for_grad,
                                       create_graph=create_graph,
                                       retain_graph=True), \
                   torch.autograd.grad(classifier_criterion / gradient_average,
                                       classifier_params_for_grad,
                                       create_graph=create_graph,
                                       retain_graph=True)


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

        load_grad_dict(self.module, meta_grad_dict)

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
                 sample_task: bool = True,
                 distributed: bool = False):
        super(Reptile, self).__init__()
        self.module = module
        self.inner_lr = inner_lr
        self.sample_task = sample_task
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
        num_tasks = len(tasks)

        if self.sample_task:
            i = np.random.choice(range(num_tasks))

            params_for_grad = get_params_for_grad(self.module)
            new_params = self.inner_loop(task=tasks[i],
                                         loader=train_batch[i],
                                         params_for_grad=params_for_grad)

            grads = [x - y for x, y in zip(params_for_grad, new_params)]
        else:
            grads = [torch.zeros_like(g) for g in get_params_for_grad(self.module)]
            meta_param_dict = param_dict(self.module, clone=True)

            for i, _ in enumerate(train_batch):
                load_param_dict(self.module, meta_param_dict)
                params_for_grad = get_params_for_grad(self.module)

                new_params = self.inner_loop(task=tasks[i],
                                             loader=train_batch[i],
                                             params_for_grad=params_for_grad)

                grads = [z + x - y for x, y, z in zip(params_for_grad, new_params, grads)]

            grads = [g.div(num_tasks) for g in grads]
            params_for_grad = get_params_for_grad(self.module)

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
                                           create_graph=False)
                new_params = grad_step_params(params=params_for_grad,
                                              grads=grads,
                                              lr=self.inner_lr,
                                              inplace=False)
                set_params_with_grad(module=self.module,
                                     params=new_params)
        self.module.train(training_status)
        return new_params


class Reptile2(nn.Module):
    def __init__(self,
                 module: torch.nn.Module,
                 inner_lr: float,
                 sample_task: bool = True,
                 distributed: bool = False):
        super(Reptile2, self).__init__()
        self.module = module
        self.inner_lr = inner_lr
        self.sample_task = sample_task
        self.gradient_task = GradientTaskModule2()
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
        num_tasks = len(tasks)

        if self.sample_task:
            i = np.random.choice(range(num_tasks))

            domain_params_for_grad = [p for p in self.module.domain_layer.parameters() if p is not None
                                      and p.requires_grad]
            classifier_params_for_grad = [p for p in self.module.parameters() if p is not None
                                          and p.requires_grad
                                          and p not in set(self.module.domain_layer.parameters())]

            new_domain_params, new_classifier_params = self.inner_loop(task=tasks[i],
                                                                       loader=train_batch[i],
                                                                       domain_params_for_grad=domain_params_for_grad,
                                                                       classifier_params_for_grad=classifier_params_for_grad,
                                                                       task_number=i)
            domain_grads = [x - y for x, y in zip(domain_params_for_grad, new_domain_params)]
            classifier_grads = [x - y for x, y in zip(classifier_params_for_grad, new_classifier_params)]
        else:
            domain_grads = [torch.zeros_like(p) for p in self.module.domain_layer.parameters() if p is not None
                            and p.requires_grad]
            classifier_grads = [torch.zeros_like(p) for p in self.module.parameters() if p is not None
                                and p.requires_grad
                                and p not in set(self.module.domain_layer.parameters())]
            meta_param_dict = param_dict(self.module, clone=True)

            for i, _ in enumerate(train_batch):
                load_param_dict(self.module, meta_param_dict)

                domain_params_for_grad = [p for p in self.module.domain_layer.parameters() if p is not None
                                          and p.requires_grad]
                classifier_params_for_grad = [p for p in self.module.parameters() if p is not None
                                              and p.requires_grad
                                              and p not in set(self.module.domain_layer.parameters())]

                new_domain_params, new_classifier_params = self.inner_loop(task=tasks[i],
                                                                           loader=train_batch[i],
                                                                           domain_params_for_grad=domain_params_for_grad,
                                                                           classifier_params_for_grad=classifier_params_for_grad,
                                                                           task_number=i)

                domain_grads = [z + x - y for x, y, z in zip(domain_params_for_grad, new_domain_params, domain_grads)]
                classifier_grads = [z + x - y for x, y, z in zip(classifier_params_for_grad, new_classifier_params, classifier_grads)]

            domain_grads = [g.div(num_tasks) for g in domain_grads]
            classifier_grads = [g.div(num_tasks) for g in classifier_grads]

            load_param_dict(self.module, meta_param_dict)
            domain_params_for_grad = [p for p in self.module.domain_layer.parameters() if p is not None
                                      and p.requires_grad]
            classifier_params_for_grad = [p for p in self.module.parameters() if p is not None
                                          and p.requires_grad
                                          and p not in set(self.module.domain_layer.parameters())]

        set_grads_for_params(params=domain_params_for_grad, grads=domain_grads)
        set_grads_for_params(params=classifier_params_for_grad, grads=classifier_grads)

        if self.distributed:
            reduce_gradients(self.module)

    def inner_loop(self,
                   task: ClassifierTask2,
                   loader: Iterable,
                   domain_params_for_grad: list,
                   classifier_params_for_grad: list,
                   task_number: int) -> (list, list):

        training_status = self.training
        self.module.train()

        for input_batch in loader:
            with torch.enable_grad():
                domain_grads, classifier_grads = self.gradient_task(task=task,
                                                                    module=self.module,
                                                                    input_batch=input_batch,
                                                                    domain_params_for_grad=domain_params_for_grad,
                                                                    classifier_params_for_grad=classifier_params_for_grad,
                                                                    create_graph=False,
                                                                    task_number=task_number)
                new_domain_params = grad_step_params(params=domain_params_for_grad,
                                                     grads=domain_grads,
                                                     lr=self.inner_lr,
                                                     inplace=False)

                set_params_with_grad(module=self.module.domain_layer,
                                     params=new_domain_params)

                new_classifier_params = grad_step_params(params=classifier_params_for_grad,
                                                         grads=classifier_grads,
                                                         lr=self.inner_lr,
                                                         inplace=False)

                set_params_with_grad(module=self.module,
                                     params=new_classifier_params,
                                     blacklist=set([p for p in self.module.domain_layer.parameters()]))

        self.module.train(training_status)
        return new_domain_params, new_classifier_params


class MetaTrainWrapper(nn.Module):
    def __init__(self,
                 module: nn.Module,
                 inner_lr: float,
                 meta_module: str = 'reptile',
                 optim: torch.optim.Optimizer = None,
                 second_order: bool = False,
                 sample_task: bool = True,
                 distributed: bool = False,
                 world_size: int = 1,
                 rank: int = -1):
        super(MetaTrainWrapper, self).__init__()
        self.module = module
        self.optim = optim
        self.distributed = distributed
        self.init_distributed(world_size, rank)

        if meta_module == 'maml':
            self.meta_module = MAML(module=self.module,
                                    inner_lr=inner_lr,
                                    second_order=second_order)
        elif meta_module == 'reptile':
            self.meta_module = Reptile(module=self.module,
                                       inner_lr=inner_lr,
                                       sample_task=sample_task)
        elif meta_module == 'reptile2':
            self.meta_module = Reptile2(module=self.module,
                                        inner_lr=inner_lr,
                                        sample_task=sample_task)
        else:
            assert False

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
                         params: list,
                         blacklist: set = None):
    j = 0
    for _, param in module.named_parameters():
        if param is not None and param.requires_grad:
            if blacklist is None or (blacklist is not None and param not in blacklist):
                param = params[j]
                j += 1


def zero_grad(optimizer: torch.optim.Optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad = p.grad.detach()
                p.grad.zero_()
