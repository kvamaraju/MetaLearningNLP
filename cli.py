import click
import time
from typing import Iterable

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from utils import AverageMeter, accuracy, set_directory, save_checkpoint, resume_from_checkpoint
from models.mlp import MLP

from datasets.mnli import *


def train_single_epoch(train_loader: Iterable,
                       model: torch.nn.Module,
                       criterion: torch.nn.modules.loss,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       total_steps: int,
                       print_freq: int,
                       writer: SummaryWriter,
                       epoch_size: int = None,
                       shape: list = None) -> int:

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    ema_loss, steps = 0, 0
    for i, (s1, s2, target) in enumerate(train_loader):
        steps += 1
        total_steps += 1

        if torch.cuda.is_available():
            target = target.cuda(async=True)
            s1 = s1.cuda()
            s2 = s2.cuda()

        if shape is None:
            s1_var = torch.autograd.Variable(s1)
            s2_var = torch.autograd.Variable(s2)
        else:
            s1_var = torch.autograd.Variable(s1.view(shape))
            s2_var = torch.autograd.Variable(s2.view(shape))
        target_var = torch.autograd.Variable(target)

        optimizer.zero_grad()

        output = model(s1_var, s2_var)
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

        if epoch_size is not None:
            if steps > epoch_size:
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
             writer: SummaryWriter,
             shape: list = None) -> float:

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (s1, s2, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda(async=True)
            s1 = s1.cuda()
            s2 = s2.cuda()

        if shape is None:
            s1_var = torch.autograd.Variable(s1)
            s2_var = torch.autograd.Variable(s2)
        else:
            s1_var = torch.autograd.Variable(s1.view(shape))
            s2_var = torch.autograd.Variable(s2.view(shape))
        target_var = torch.autograd.Variable(target)

        output = model(s1_var, s2_var)
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
@click.option('--type', type=click.Choice(['mlp']), default='mlp')
@click.option('--lr', default=0.0001, type=float)
@click.option('--start_epoch', default=0, type=int)
@click.option('--epochs', default=50, type=int)
@click.option('--epoch_size', default=20, type=int)
@click.option('--batch_size', default=50, type=int)
@click.option('--print_freq', default=100, type=int)
@click.option('--save_at', type=list, default=[1, 10, 50, 100])
@click.option('--resume', default='', type=str)
@click.option('--device', type=int, default=0)
@click.option('--multi_gpu', default=False)
def train_mnli(**kwargs):
    dir = set_directory(name='mlp', type_net='mlp')
    writer = SummaryWriter(dir)

    train, dev_matched, vocab = prepare_mnli(root='datasets/data',
                                             urls=['https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip'],
                                             dir='MultiNLI',
                                             name='MultiNLI',
                                             data_path='datasets/data/MultiNLI/multinli_1.0',
                                             max_len=50)

    weight_matrix = prepare_glove(glove_path="datasets/GloVe/glove.840B.300d.txt",
                                  vocab=vocab)

    train_loader = DataLoader(
        MultiNLIDataset(dataset=train),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(
        MultiNLIDataset(dataset=dev_matched),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    if kwargs['type'] == 'mlp':
        model = MLP(num_embeddings=weight_matrix.shape[0],
                    embedding_matrix=weight_matrix)
    else:
        model = None

    num_parameters = sum([p.data.nelement() for p in model.parameters()])
    print(f'Number of model parameters: {num_parameters}')

    if torch.cuda.is_available():
        torch.cuda.set_device(kwargs['device'])

    if kwargs['multi_gpu']:
        model = torch.nn.DataParallel(model).cuda()
    else:
        if torch.cuda.is_available():
            model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=kwargs['lr'])

    if kwargs['resume'] != '':
        kwargs['start_epoch'], best_prec1, total_steps, model, optimizer = resume_from_checkpoint(resume_path=kwargs['resume'],
                                                                                                  model=model,
                                                                                                  optimizer=optimizer)
    else:
        total_steps = 0
        best_prec1 = 0.

    cudnn.benchmark = True

    loss_function = torch.nn.CrossEntropyLoss().cuda()

    for epoch in range(kwargs['start_epoch'], kwargs['epochs']):
        total_steps = train_single_epoch(train_loader=train_loader,
                                         model=model,
                                         criterion=loss_function,
                                         optimizer=optimizer,
                                         epoch=epoch,
                                         total_steps=total_steps,
                                         print_freq=kwargs['print_freq'],
                                         epoch_size=kwargs['epoch_size']*kwargs['batch_size'],
                                         writer=writer)

        prec1 = validate(val_loader=val_loader,
                         model=model,
                         criterion=loss_function,
                         epoch=epoch,
                         print_freq=kwargs['print_freq'],
                         writer=writer)

        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': max(prec1, best_prec1),
            'optimizer': optimizer.state_dict(),
            'total_steps': total_steps
        }

        if epoch in kwargs['save_at']:
            name = f'checkpoint_{epoch}'
            filename = f'checkpoint_{epoch}.pth.tar'
        else:
            name = 'checkpoint'
            filename = 'checkpoint.pth.tar'

        save_checkpoint(state=state,
                        is_best=is_best,
                        name=name,
                        filename=filename)

    print('Best accuracy: ', best_prec1)


@cli.command()
@click.option('--type', type=click.Choice(['mlp']), default='mlp')
@click.option('--k', default=5, type=int)
@click.option('--lr', default=0.0001, type=float)
@click.option('--start_epoch', default=0, type=int)
@click.option('--epochs', default=50, type=int)
@click.option('--epoch_size', default=20, type=int)
@click.option('--batch_size', default=100, type=int)
@click.option('--print_freq', default=100, type=int)
@click.option('--save_at', type=list, default=[1, 10, 50, 100])
@click.option('--resume', default='', type=str)
@click.option('--device', type=int, default=0)
@click.option('--multi_gpu', default=False)
def train_mnli_kshot(**kwargs):
    dir = set_directory(name='mlp', type_net='mlp')
    writer = SummaryWriter(dir)

    train, dev_matched_train, test, dev_matched_test, vocab = prepare_mnli_split(root='datasets/data',
                                                                                 urls=['https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip'],
                                                                                 dir='MultiNLI',
                                                                                 name='MultiNLI',
                                                                                 data_path='datasets/data/MultiNLI/multinli_1.0',
                                                                                 train_genres=[['fiction', 'government', 'slate', 'telephone']],
                                                                                 test_genres=[['travel']],
                                                                                 max_len=50)

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

    model = MLP(num_embeddings=weight_matrix.shape[0],
                embedding_matrix=weight_matrix)

    num_parameters = sum([p.data.nelement() for p in model.parameters()])
    print(f'Number of model parameters: {num_parameters}')

    if torch.cuda.is_available():
        torch.cuda.set_device(kwargs['device'])

    if kwargs['multi_gpu']:
        model = torch.nn.DataParallel(model).cuda()
    else:
        if torch.cuda.is_available():
            model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=kwargs['lr'])

    if kwargs['resume'] != '':
        kwargs['start_epoch'], best_prec1, total_steps, model, optimizer = resume_from_checkpoint(resume_path=kwargs['resume'],
                                                                                                  model=model,
                                                                                                  optimizer=optimizer)
    else:
        total_steps = 0
        best_prec1 = 0.

    cudnn.benchmark = True

    loss_function = torch.nn.CrossEntropyLoss().cuda()

    for epoch in range(kwargs['start_epoch'], kwargs['epochs']):
        total_steps = train_single_epoch(train_loader=train_loader,
                                         model=model,
                                         criterion=loss_function,
                                         optimizer=optimizer,
                                         epoch=epoch,
                                         total_steps=total_steps,
                                         print_freq=kwargs['print_freq'],
                                         epoch_size=kwargs['epoch_size']*kwargs['batch_size'],
                                         writer=writer)

        prec1 = validate(val_loader=val_loader,
                         model=model,
                         criterion=loss_function,
                         epoch=epoch,
                         print_freq=kwargs['print_freq'],
                         writer=writer)

        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': max(prec1, best_prec1),
            'optimizer': optimizer.state_dict(),
            'total_steps': total_steps
        }

        if epoch in kwargs['save_at']:
            name = f'checkpoint_{epoch}'
            filename = f'checkpoint_{epoch}.pth.tar'
        else:
            name = 'checkpoint'
            filename = 'checkpoint.pth.tar'

        save_checkpoint(state=state,
                        is_best=is_best,
                        name=name,
                        filename=filename)

    print('Best accuracy: ', best_prec1)

    print('Zero Shot Performance')

    train_loader = DataLoader(
        MultiNLIDataset(dataset=test[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(
        MultiNLIDataset(dataset=dev_matched_test[0]),
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available())

    validate(val_loader=train_loader,
             model=model,
             criterion=loss_function,
             epoch=epoch,
             print_freq=kwargs['print_freq'],
             writer=writer)

    validate(val_loader=val_loader,
             model=model,
             criterion=loss_function,
             epoch=epoch,
             print_freq=kwargs['print_freq'],
             writer=writer)

    print(f"{kwargs['k']}-Shot Performance")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=kwargs['lr'])

    for epoch in range(kwargs['k']):
        train_single_epoch(train_loader=train_loader,
                           model=model,
                           criterion=loss_function,
                           optimizer=optimizer,
                           epoch=epoch,
                           total_steps=total_steps,
                           print_freq=kwargs['print_freq'],
                           epoch_size=kwargs['epoch_size']*kwargs['batch_size'],
                           writer=writer)

        validate(val_loader=val_loader,
                 model=model,
                 criterion=loss_function,
                 epoch=epoch,
                 print_freq=kwargs['print_freq'],
                 writer=writer)


if __name__ == '__main__':
    cli()
