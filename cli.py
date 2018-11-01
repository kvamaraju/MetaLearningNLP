import click
import time

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from copy import deepcopy

from utils import AverageMeter, accuracy, set_directory, construct_optimizer, construct_model, construct_model_r
from metalearning import ClassifierTask, MetaTrainWrapper, zero_grad

from datasets.mnli import *


def train_single_epoch(train_loader: Iterable,
                       model: torch.nn.Module,
                       criterion: torch.nn.modules.loss,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       total_steps: int,
                       print_freq: int,
                       writer: SummaryWriter,
                       num_batches: int = None) -> int:

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    ema_loss, steps = 0, 0

    for i, (s1, s2, target, l1, l2) in enumerate(train_loader):
        steps += 1
        total_steps += 1

        if torch.cuda.is_available():
            target = target.cuda(async=True)
            s1 = s1.cuda()
            s2 = s2.cuda()
            l1 = l1.cuda()
            l2 = l2.cuda()

        s1_var = torch.autograd.Variable(s1)
        s2_var = torch.autograd.Variable(s2)
        l1_var = torch.autograd.Variable(l1)
        l2_var = torch.autograd.Variable(l2)
        target_var = torch.autograd.Variable(target)

        zero_grad(optimizer)

        output = model(s1_var, s2_var, l1_var, l2_var)
        loss = criterion(output, target_var)

        loss.backward()
        optimizer.step()

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), s1.size(0))
        top1.update(prec1, s1.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(f' Epoch: [{epoch}][{i}/{len(train_loader)}]\t' +
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' +
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t' +
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

        if num_batches is not None:
            if steps > num_batches:
                break

    if writer is not None:
        writer.add_scalar('train/loss', losses.avg, epoch)
        writer.add_scalar('train/acc', top1.avg, epoch)

    return total_steps


def train_single_epoch_with_gradient_reversal(train_loader: Iterable,
                                              val_loader: Iterable,
                                              model: torch.nn.Module,
                                              criterion: torch.nn.modules.loss,
                                              optimizer: torch.optim.Optimizer,
                                              epoch: int,
                                              alpha: float,
                                              total_steps: int,
                                              print_freq: int,
                                              writer: SummaryWriter,
                                              num_batches: int = None) -> int:

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    ema_loss, steps = 0, 0

    len_dataloader = len(train_loader)
    num_loaders = len(train_loader) // len(val_loader) + 1

    train_iter = iter(train_loader)

    val_generator = (deepcopy(val_loader) for _ in range(num_loaders))
    val_iter = iter(next(val_generator))

    for i in range(len_dataloader):
        s1_train, s2_train, target_train, l1_train, l2_train = next(train_iter)
        try:
            s1_val, s2_val, target_val, l1_val, l2_val = next(val_iter)
        except:
            val_iter = iter(next(val_generator))
            s1_val, s2_val, target_val, l1_val, l2_val = next(val_iter)
        steps += 1
        total_steps += 1

        if torch.cuda.is_available():
            target = target_train.cuda(async=True)
            s1 = s1_train.cuda()
            s2 = s2_train.cuda()
            l1 = l1_train.cuda()
            l2 = l2_train.cuda()
        else:
            target = target_train
            s1 = s1_train
            s2 = s2_train
            l1 = l1_train
            l2 = l2_train

        s1_var = torch.autograd.Variable(s1)
        s2_var = torch.autograd.Variable(s2)
        l1_var = torch.autograd.Variable(l1)
        l2_var = torch.autograd.Variable(l2)
        target_var = torch.autograd.Variable(target)

        zero_grad(optimizer)

        class_output, domain_output = model(s1_var, s2_var, l1_var, l2_var, alpha=alpha)
        loss_train = criterion(class_output, target_var)
        loss_domain = criterion(domain_output, torch.zeros_like(target_var))

        prec1 = accuracy(class_output.data, target, topk=(1,))[0]
        losses.update(loss_train.item(), s1.size(0))
        top1.update(prec1, s1.size(0))

        if torch.cuda.is_available():
            target = target_val.cuda(async=True)
            s1 = s1_val.cuda()
            s2 = s2_val.cuda()
            l1 = l1_val.cuda()
            l2 = l2_val.cuda()
        else:
            target = target_val
            s1 = s1_val
            s2 = s2_val
            l1 = l1_val
            l2 = l2_val

        s1_var = torch.autograd.Variable(s1)
        s2_var = torch.autograd.Variable(s2)
        l1_var = torch.autograd.Variable(l1)
        l2_var = torch.autograd.Variable(l2)
        target_var = torch.autograd.Variable(target)

        _, domain_output = model(s1_var, s2_var, l1_var, l2_var, alpha=alpha)
        loss_val = criterion(domain_output, torch.ones_like(target_var))

        loss = loss_train + loss_domain + loss_val
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(f' Epoch: [{epoch}][{i}/{len(train_loader)}]\t' +
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' +
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t' +
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

        if num_batches is not None:
            if steps > num_batches:
                break

    if writer is not None:
        writer.add_scalar('train/loss', losses.avg, epoch)
        writer.add_scalar('train/acc', top1.avg, epoch)

    return total_steps


def validate(val_loader: Iterable,
             model: torch.nn.Module,
             criterion: torch.nn.modules.loss,
             epoch: int,
             print_freq: int,
             writer: SummaryWriter) -> float:

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (s1, s2, target, l1, l2) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(async=True)
            s1 = s1.cuda()
            s2 = s2.cuda()
            l1 = l1.cuda()
            l2 = l2.cuda()

        s1_var = torch.autograd.Variable(s1)
        s2_var = torch.autograd.Variable(s2)
        l1_var = torch.autograd.Variable(l1)
        l2_var = torch.autograd.Variable(l2)
        target_var = torch.autograd.Variable(target)

        output = model(s1_var, s2_var, l1_var, l2_var)
        if isinstance(output, tuple):
            output = output[0]
        loss = criterion(output, target_var)

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), s1.size(0))
        top1.update(prec1, s1.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(f'Test: [{i}/{len(val_loader)}]\t' +
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' +
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t' +
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    print(f' * Prec@1 {top1.avg:.3f}')

    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/acc', top1.avg, epoch)

    return top1.avg


@click.group()
def cli():
    pass


@cli.command()
@click.option('--type', type=click.Choice(['mlp', 'transformer', 'lstm']), default='mlp')
@click.option('--optim', type=click.Choice(['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'rprop', 'sgd']), default='adam')
@click.option('--lr', default=0.001, type=float)
@click.option('--epochs', default=100, type=int)
@click.option('--batch-size', default=100, type=int)
@click.option('--print_freq', default=100, type=int)
@click.option('--device', type=int, default=0)
def train_mnli(**kwargs):
    dir = set_directory(name=kwargs['type'], type_net=kwargs['type'])
    writer = SummaryWriter(dir)

    train, dev_matched, dev_mismatched, vocab = prepare_mnli(root='datasets/data',
                                                             urls=['https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip'],
                                                             dir='MultiNLI',
                                                             name='MultiNLI',
                                                             data_path='datasets/data/MultiNLI/multinli_1.0',
                                                             max_len=60)

    weight_matrix = prepare_glove(glove_path="datasets/GloVe/glove.840B.300d.txt",
                                  vocab=vocab)

    train_loader = DataLoader(
        MultiNLIDataset(dataset=train),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    val_loader = [DataLoader(
        MultiNLIDataset(dataset=loader),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available()) for loader in [dev_matched, dev_mismatched]]

    model = construct_model(model_type=kwargs['type'],
                            weight_matrix=weight_matrix)

    num_parameters = sum([p.data.nelement() for p in model.parameters()])
    print(f'Number of model parameters: {num_parameters}')

    if torch.cuda.is_available():
        torch.cuda.set_device(kwargs['device'])

    if torch.cuda.is_available():
        model = model.cuda()
        loss_function = torch.nn.CrossEntropyLoss().cuda()
    else:
        loss_function = torch.nn.CrossEntropyLoss()

    optimizer = construct_optimizer(optimizer=kwargs['optim'],
                                    model=model,
                                    lr=kwargs['lr'])

    total_steps = 0

    cudnn.benchmark = True

    for epoch in tqdm(range(kwargs['epochs'])):
        total_steps = train_single_epoch(train_loader=train_loader,
                                         model=model,
                                         criterion=loss_function,
                                         optimizer=optimizer,
                                         epoch=epoch,
                                         total_steps=total_steps,
                                         print_freq=kwargs['print_freq'],
                                         writer=writer)

        for loader in val_loader:
            validate(val_loader=loader,
                     model=model,
                     criterion=loss_function,
                     epoch=epoch,
                     print_freq=kwargs['print_freq'],
                     writer=writer)


@cli.command()
@click.option('--type', type=click.Choice(['mlp', 'transformer', 'lstm']), default='mlp')
@click.option('--optim', type=click.Choice(['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'rprop', 'sgd']), default='adam')
@click.option('--k', default=5, type=int)
@click.option('--lr', default=0.001, type=float)
@click.option('--lr-kshot', default=0.0001, type=float)
@click.option('--epochs', default=100, type=int)
@click.option('--batch-size', default=100, type=int)
@click.option('--print-freq', default=100, type=int)
@click.option('--device', type=int, default=0)
def train_mnli_kshot(**kwargs):
    dir = set_directory(name=kwargs['type'], type_net=kwargs['type'])
    writer = SummaryWriter(dir)

    train, dev_matched_train, test, dev_matched_test, dev_mismatched_test, vocab = prepare_mnli_split(root='datasets/data',
                                                                                                      urls=['https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip'],
                                                                                                      dir='MultiNLI',
                                                                                                      name='MultiNLI',
                                                                                                      data_path='datasets/data/MultiNLI/multinli_1.0',
                                                                                                      train_genres=[['government', 'telephone', 'slate', 'travel']],
                                                                                                      test_genres=[['fiction']],
                                                                                                      max_len=60)

    weight_matrix = prepare_glove(glove_path="datasets/GloVe/glove.840B.300d.txt",
                                  vocab=vocab)

    train_loader = DataLoader(
        MultiNLIDataset(dataset=train[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(
        MultiNLIDataset(dataset=dev_matched_train[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    model = construct_model(model_type=kwargs['type'],
                            weight_matrix=weight_matrix)

    num_parameters = sum([p.data.nelement() for p in model.parameters()])
    print(f'Number of model parameters: {num_parameters}')

    if torch.cuda.is_available():
        torch.cuda.set_device(kwargs['device'])

    if torch.cuda.is_available():
        model = model.cuda()
        loss_function = torch.nn.CrossEntropyLoss().cuda()
    else:
        loss_function = torch.nn.CrossEntropyLoss()

    optimizer = construct_optimizer(optimizer=kwargs['optim'],
                                    model=model,
                                    lr=kwargs['lr'])

    cudnn.benchmark = True

    total_steps = 0

    for epoch in tqdm(range(kwargs['epochs'])):
        total_steps = train_single_epoch(train_loader=train_loader,
                                         model=model,
                                         criterion=loss_function,
                                         optimizer=optimizer,
                                         epoch=epoch,
                                         total_steps=total_steps,
                                         print_freq=kwargs['print_freq'],
                                         writer=writer)

        validate(val_loader=val_loader,
                 model=model,
                 criterion=loss_function,
                 epoch=epoch,
                 print_freq=kwargs['print_freq'],
                 writer=writer)

    print('Zero Shot Performance')

    train_loader = DataLoader(
        MultiNLIDataset(dataset=test[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    val_loader = [DataLoader(
        MultiNLIDataset(dataset=dataset[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available()) for dataset in [dev_matched_test, dev_mismatched_test]]

    validate(val_loader=train_loader,
             model=model,
             criterion=loss_function,
             epoch=epoch,
             print_freq=kwargs['print_freq'],
             writer=writer)

    for loader in val_loader:
        validate(val_loader=loader,
                 model=model,
                 criterion=loss_function,
                 epoch=epoch,
                 print_freq=kwargs['print_freq'],
                 writer=writer)

    if kwargs['k'] > 0:
        print(f"{kwargs['k']}-Shot Performance")
        optimizer = construct_optimizer(optimizer=kwargs['optim'],
                                        model=model,
                                        lr=kwargs['lr_kshot'])

        train_batcher = Batcher(loaders=[train_loader],
                                batch_size=1)

        for i, train_batch in enumerate(train_batcher):
            train_single_epoch(train_loader=train_batch[0],
                               model=model,
                               criterion=loss_function,
                               optimizer=optimizer,
                               epoch=i,
                               total_steps=0,
                               print_freq=kwargs['print_freq'],
                               num_batches=1,
                               writer=writer)

            for loader in val_loader:
                validate(val_loader=loader,
                         model=model,
                         criterion=loss_function,
                         epoch=epoch,
                         print_freq=kwargs['print_freq'],
                         writer=writer)

            if i >= kwargs['k']:
                break


@cli.command()
@click.option('--type', type=click.Choice(['mlp', 'transformer', 'lstm']), default='mlp')
@click.option('--optim', type=click.Choice(['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'rprop', 'sgd']), default='adam')
@click.option('--lr', default=0.001, type=float)
@click.option('--epochs', default=100, type=int)
@click.option('--batch-size', default=100, type=int)
@click.option('--print-freq', default=100, type=int)
@click.option('--device', type=int, default=0)
def train_mnli_gradient_reversal(**kwargs):
    dir = set_directory(name=kwargs['type'], type_net=kwargs['type'])
    writer = SummaryWriter(dir)

    train, dev_matched_train, test, dev_matched_test, dev_mismatched_test, vocab = prepare_mnli_split(root='datasets/data',
                                                                                                      urls=['https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip'],
                                                                                                      dir='MultiNLI',
                                                                                                      name='MultiNLI',
                                                                                                      data_path='datasets/data/MultiNLI/multinli_1.0',
                                                                                                      train_genres=[['government', 'telephone', 'slate', 'travel']],
                                                                                                      test_genres=[['fiction']],
                                                                                                      max_len=60)

    weight_matrix = prepare_glove(glove_path="datasets/GloVe/glove.840B.300d.txt",
                                  vocab=vocab)

    train_loader = DataLoader(
        MultiNLIDataset(dataset=train[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    test_loader = DataLoader(
        MultiNLIDataset(dataset=test[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(
        MultiNLIDataset(dataset=dev_matched_train[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    model = construct_model_r(model_type=kwargs['type'],
                              weight_matrix=weight_matrix)

    num_parameters = sum([p.data.nelement() for p in model.parameters()])
    print(f'Number of model parameters: {num_parameters}')

    if torch.cuda.is_available():
        torch.cuda.set_device(kwargs['device'])

    if torch.cuda.is_available():
        model = model.cuda()
        loss_function = torch.nn.CrossEntropyLoss().cuda()
    else:
        loss_function = torch.nn.CrossEntropyLoss()

    optimizer = construct_optimizer(optimizer=kwargs['optim'],
                                    model=model,
                                    lr=kwargs['lr'])

    cudnn.benchmark = True

    total_steps = 0

    for epoch in tqdm(range(kwargs['epochs'])):
        total_steps = train_single_epoch_with_gradient_reversal(train_loader=train_loader,
                                                                val_loader=test_loader,
                                                                model=model,
                                                                criterion=loss_function,
                                                                optimizer=optimizer,
                                                                epoch=epoch,
                                                                alpha=1e-2,
                                                                total_steps=total_steps,
                                                                print_freq=kwargs['print_freq'],
                                                                writer=writer)

        validate(val_loader=val_loader,
                 model=model,
                 criterion=loss_function,
                 epoch=epoch,
                 print_freq=kwargs['print_freq'],
                 writer=writer)

    print('Zero Shot Performance')

    train_loader = DataLoader(
        MultiNLIDataset(dataset=test[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    val_loader = [DataLoader(
        MultiNLIDataset(dataset=dataset[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available()) for dataset in [dev_matched_test, dev_mismatched_test]]

    validate(val_loader=train_loader,
             model=model,
             criterion=loss_function,
             epoch=epoch,
             print_freq=kwargs['print_freq'],
             writer=writer)

    for loader in val_loader:
        validate(val_loader=loader,
                 model=model,
                 criterion=loss_function,
                 epoch=epoch,
                 print_freq=kwargs['print_freq'],
                 writer=writer)


@cli.command()
@click.option('--type', type=click.Choice(['mlp', 'transformer', 'lstm']), default='mlp')
@click.option('--optim', type=click.Choice(['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'rprop', 'sgd']), default='adam')
@click.option('--use-maml', type=bool, default=False)
@click.option('--k', default=5, type=int)
@click.option('--lr-inner-meta', default=0.001, type=float)
@click.option('--lr-outer-meta', default=0.001, type=float)
@click.option('--num-inner-iterations', default=1, type=int)
@click.option('--lr-kshot', default=0.001, type=float)
@click.option('--epochs', default=20, type=int)
@click.option('--batch-size', default=100, type=int)
@click.option('--print-freq', default=100, type=int)
@click.option('--device', type=int, default=0)
def train_mnli_meta(**kwargs):
    train, dev_matched_train, test, dev_matched_test, dev_mismatched_test, vocab = prepare_mnli_split(root='datasets/data',
                                                                                                      urls=['https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip'],
                                                                                                      dir='MultiNLI',
                                                                                                      name='MultiNLI',
                                                                                                      data_path='datasets/data/MultiNLI/multinli_1.0',
                                                                                                      train_genres=[['government'], ['telephone'], ['slate'], ['travel']],
                                                                                                      test_genres=[['fiction']],
                                                                                                      max_len=60)

    weight_matrix = prepare_glove(glove_path="datasets/GloVe/glove.840B.300d.txt",
                                  vocab=vocab)

    train_loaders = [DataLoader(
        MultiNLIDataset(dataset=t),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available()) for t in train]

    val_matched_loaders = [DataLoader(
                           MultiNLIDataset(dataset=t),
                           batch_size=2000,
                           shuffle=True,
                           num_workers=1,
                           pin_memory=torch.cuda.is_available()) for t in dev_matched_train]

    model = construct_model(model_type=kwargs['type'],
                            weight_matrix=weight_matrix)

    num_parameters = sum([p.data.nelement() for p in model.parameters()])
    print(f'Number of model parameters: {num_parameters}')

    cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.cuda.set_device(kwargs['device'])

    if torch.cuda.is_available():
        model = model.cuda()
        loss_function = torch.nn.CrossEntropyLoss().cuda()
    else:
        loss_function = torch.nn.CrossEntropyLoss()

    optimizer = construct_optimizer(optimizer=kwargs['optim'],
                                    model=model,
                                    lr=kwargs['lr_outer_meta'])

    meta_model = MetaTrainWrapper(module=model,
                                  inner_lr=kwargs['lr_inner_meta'],
                                  use_maml=kwargs['use_maml'],
                                  optim=optimizer,
                                  second_order=True,
                                  sample_task=True)

    train_batcher = Batcher(loaders=train_loaders,
                            batch_size=kwargs['num_inner_iterations'])
    meta_model.train()
    for epoch in range(kwargs['epochs']):
        for train_batch in tqdm(train_batcher):
            meta_model(tasks=[ClassifierTask() for _ in range(len(train_loaders))],
                       train_batch=train_batch,
                       val_loaders=train_loaders)

        print(f'Epoch {epoch + 1} Validation')
        prec = []
        for loader in val_matched_loaders:
            prec.append(validate(val_loader=loader,
                                 model=model,
                                 criterion=loss_function,
                                 epoch=epoch,
                                 print_freq=kwargs['print_freq'],
                                 writer=None))
        print(f'Average Matched Precision is {np.mean(prec)}')

    train_loader = DataLoader(
        MultiNLIDataset(dataset=test[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    val_matched_loader = DataLoader(
        MultiNLIDataset(dataset=dev_matched_test[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    val_mismatched_loader = DataLoader(
        MultiNLIDataset(dataset=dev_mismatched_test[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    print('Zero Shot Performance')

    validate(val_loader=train_loader,
             model=model,
             criterion=loss_function,
             epoch=0,
             print_freq=kwargs['print_freq'],
             writer=None)

    validate(val_loader=val_matched_loader,
             model=model,
             criterion=loss_function,
             epoch=0,
             print_freq=kwargs['print_freq'],
             writer=None)

    validate(val_loader=val_mismatched_loader,
             model=model,
             criterion=loss_function,
             epoch=0,
             print_freq=kwargs['print_freq'],
             writer=None)

    if kwargs['k'] > 0:
        print(f"{kwargs['k']}-Shot Performance")
        optimizer = construct_optimizer(optimizer=kwargs['optim'],
                                        model=model,
                                        lr=kwargs['lr_kshot'])

        train_batcher = Batcher(loaders=[train_loader],
                                batch_size=1)

        for i, train_batch in enumerate(train_batcher):
            train_single_epoch(train_loader=train_batch[0],
                               model=model,
                               criterion=loss_function,
                               optimizer=optimizer,
                               epoch=i,
                               total_steps=0,
                               print_freq=kwargs['print_freq'],
                               num_batches=1,
                               writer=None)

            validate(val_loader=val_matched_loader,
                     model=model,
                     criterion=loss_function,
                     epoch=0,
                     print_freq=kwargs['print_freq'],
                     writer=None)

            validate(val_loader=val_mismatched_loader,
                     model=model,
                     criterion=loss_function,
                     epoch=0,
                     print_freq=kwargs['print_freq'],
                     writer=None)

            if i >= kwargs['k']:
                break


if __name__ == '__main__':
    cli()
