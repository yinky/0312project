import argparse
import sys
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import copy
#import calibration as cal
import random
from torch.autograd import Variable
import torch.nn.functional as F

import resnet as resnet
from ece_loss import ECELoss
from temperature_scaling import ModelWithTemperature

model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet") and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='CIFAR10/100')
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size',
                    default=64,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 512)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--dataset',
                    help='',
                    default='cifar10',
                    choices=['cifar10','cifar100'],
                    type=str)
parser.add_argument('--save-dir',
                    dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./checkpoint/',
                    type=str)
parser.add_argument('--method',
                    help='method used for learning (CE, Label Smoothing, L1 Norm, Focal Loss)',
                    default='ce',
                    choices=['ce', 'ls', 'l1', 'focal'],
                    type=str)

parser.add_argument('--epsilon',
                    default=1.0,
                    type=float,
                    help='Coefficient of Label Smoothing')

parser.add_argument('--alpha',
                    default=0.05,
                    type=float,
                    help='Coefficient of L1 Norm')

parser.add_argument('--gamma',
                    default=1.0,
                    type=float,
                    help='Coefficient of Focal Loss')

parser.add_argument('--seed',
                    default=101,
                    type=int,
                    help='seed for validation data split')

best_prec1 = 0
args = parser.parse_args()
print(args)

if args.dataset == 'cifar10':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    num_classes = 10
    all_train_data1 = datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    all_train_data2 = datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    all_train_data3 = datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    test_data = datasets.CIFAR10(
        root='./data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

elif args.dataset == 'cifar100':
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])
    num_classes = 100
    all_train_data1 = datasets.CIFAR100(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    all_train_data2 = datasets.CIFAR100(
        root='./data',
        train=True,
        transform=transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    all_train_data3 = datasets.CIFAR100(
        root='./data',
        train=True,
        transform=transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    test_data = datasets.CIFAR100(
        root='./data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    
indices = np.random.RandomState(args.seed).permutation(len(all_train_data1.targets))
indices1 = indices[:6250]
indices2 = indices[6250:50000]
indices3 = indices[45000:]
train_labeled_data = torch.utils.data.Subset(all_train_data1, indices1)
train_unlabeled_data = torch.utils.data.Subset(all_train_data2,indices2)
val_data = torch.utils.data.Subset(all_train_data3, indices3)

train_labeled_loader = torch.utils.data.DataLoader(train_labeled_data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True)

train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_data,
    batch_size=7*args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_data,
    batch_size=256,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_data,
    batch_size=256,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True)

def main():
    model = resnet.__dict__[args.arch](num_classes=num_classes)
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], last_epoch= -1)


    num_epoch = args.epochs

    for epoch in range(0, num_epoch):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(model, criterion, optimizer, epoch)
        lr_scheduler.step()
        evaluate(model)

        #if epoch >= num_epoch-10:
        #    evaluate_TS(model)

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
def de_interleave(x, size): 
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def train(model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    #code by Fixmatch
    labeled = iter(train_labeled_loader)
    unlabeled = iter(train_unlabeled_loader)
    for i in range(len(train_labeled_loader)):
        inputs_s, targets_s = labeled.next()
        inputs_u_w, _ = unlabeled.next()
        inputs_u_s=inputs_u_w# inputs_u_w????????????, inputs_u_s?????????????????????????????????
        if torch.cuda.is_available():#?????????GPU????????????????????????????????????GPU???
            inputs_s, targets_s = inputs_s.cuda(), targets_s.cuda()
            inputs_u_w, inputs_u_s = inputs_u_w.cuda(), inputs_u_s.cuda()
        inputs = interleave(torch.cat((inputs_s, inputs_u_w, inputs_u_s)), 2*7+1)#?????????:?????????:?????????1:7:7
        logits = model(inputs)
        logits = de_interleave(logits, 2*7+1)
        #targets_s.size()[0]?????????batch??????????????????????????????????????????????????????
        logits_s = logits[:targets_s.size()[0]]
        logits_u_w, logits_u_s = logits[targets_s.size()[0]:].chunk(2)
        del logits

        Ls = F.cross_entropy(logits_s, targets_s, reduction='mean')

        pseudo_label = torch.softmax(logits_u_w.detach()/1, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(0.95).float()

        Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

        loss = Ls + 1 * Lu

        optimizer.zero_grad()#?????????????????????????????????
        loss.backward()#loss?????????tensor???????????????
        optimizer.step()
        
        output = logits_s.float()
        loss = loss.float()
        # measure accuracy and record loss
        # prec1 = accuracy(output.data, target_var)[0]
        prec1 = accuracy(output.data, targets_s)[0]
        #losses.update(loss.item(), input.size(0))
        losses.update(loss.item(), inputs_s.size(0))

        #top1.update(prec1.item(), input.size(0))
        top1.update(prec1.item(), inputs_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_labeled_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1))
    
    '''
    #code by lht/lyt
    labeled = iter(train_labeled_loader)
    unlabeled = iter(train_unlabeled_loader)
    for i in range(len(train_labeled_loader)):
        inputs_s, targets_s = labeled.next()#????????????batch
        inputs_u, _ = unlabeled.next()

        if torch.cuda.is_available():#?????????GPU?????????????????????????????????GPU???
            inputs_s, targets_s = inputs_s.cuda(), targets_s.cuda()
            inputs_u = inputs_u.cuda()

        logits_s = model(inputs_s)
        # logits_s = torch.nn.functional.softmax(logits_s, dim=1)
        Ls = criterion(logits_s, targets_s)#?????????????????????loss????????????????????????????????????????????????????????????????????????

        #tensor????????????????????????????????????????????????tensor???????????????/???
        #with torch.no_grad():#??????????????????????????????, ????????????
        logits_u = model(inputs_u)
        #logits_u = torch.nn.functional.softmax(logits_u, dim=1) #[512,10]
        #prob_u = torch.nn.functional.softmax(logits_u, dim=1).detach() #[512,10]

        p_max, idx = torch.max(logits_u, dim=1) #[512]
        #p_max, idx = torch.max(prob_u, dim=1) #[512]
        mask = p_max.ge(0.95)#???????????????0.95?????????????????????true????????????false [512]
        # logits_u = logits_u * mask.unsqueeze(1).expand(logits_u.shape[0],10)
        # idx = idx * mask
        idx = torch.zeros(logits_u.size()[0], num_classes).cuda().scatter_(1, idx.view(-1,1), 1)

        # Lu = criterion(logits_u, idx) #512 represent the batchsize
        Lu = - torch.mean(torch.sum(torch.nn.functional.log_softmax(logits_u, dim=1) * idx, dim=1) * mask)
        #loss = Ls #?????????????????????
        loss = Ls + 1*Lu #??????????????????
        print('Ls = ',Ls)
        print('Lu =',Lu)
        
        optimizer.zero_grad()#??????tensor????????????????????????????????????????????????
        loss.backward()#loss?????????tensor??????????????????
        optimizer.step()
    '''
    

    # for i, (input, target) in enumerate(train_labeled_loader):
    #     # measure data loading time
    #     data_time.update(time.time() - end)
    #
    #     target_bi = torch.zeros(input.size(0), num_classes).scatter_(1, target.view(-1,1).long(), 1)
    #     target_bi = target_bi.cuda()
    #     input_var = input.cuda()
    #     target_var = target.cuda()
    #
    #     output = model(input_var)
    #
    #     if args.method == 'ce':
    #         loss = criterion(output, target_var)
    #
    #     elif args.method == 'ls':
    #         epsilon = args.epsilon
    #         target_bi_smooth = (1.0 - epsilon) * target_bi + epsilon/num_classes
    #         loss = -torch.mean(torch.sum(torch.nn.functional.log_softmax(output, dim=1) * target_bi_smooth, dim=1)) ####################Label Smoothing
    #
    #     elif args.method == 'l1':
    #         loss_cla = criterion(output, target_var)
    #         loss_f1_norm = torch.mean(torch.norm(output,p=1,dim=1))
    #         loss = loss_cla + args.alpha * loss_f1_norm  ########################## L1 Norm
    #
    #     elif args.method == 'focal':
    #         target_var = target_var.view(-1,1)
    #         logpt = torch.nn.functional.log_softmax(output, dim=1)
    #         logpt = logpt.gather(1,target_var)
    #         logpt = logpt.view(-1)
    #         pt = Variable(logpt.exp().data)
    #         weights = (1-pt)**(args.gamma)
    #         loss = -torch.mean(weights * logpt)   ################################## Focal Loss
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

        # output = output.float()


def temperature_scale(logits, temp):
    """
    Perform temperature scaling on logits
    """
    temperature = temp.unsqueeze(1).expand(logits.size(0), logits.size(1)).cuda()
    return logits / temperature

def evaluate_TS(model):
    ece_criterion = ECELoss(n_bins=15).cuda()
    model.eval()
    with torch.no_grad():
        model_temp = ModelWithTemperature(model)
        #print('Searching the Temperature on Validation Data:',end='')
        best_t, best_ece = evaluate_scaling(model)
        best_t = best_t.detach().cuda()
        model_temp.temperature = best_t

        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            ece_before = ece_criterion(output, target)
            calibrated_output = model_temp.temperature_scale(output)
            ece_after = ece_criterion(calibrated_output, target)
            print('\nECE on Test Data After TS Calibration: ', round(ece_after.item(),4))
        
def evaluate(model):
    model.eval()
    correct = 0
    ece_criterion = ECELoss().cuda()
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            output_nosoftmax = model(data)
            output = torch.nn.functional.softmax(output_nosoftmax, dim=1)
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            ece = ece_criterion(output_nosoftmax, target)
    print('\nTest set: Accuracy: {:.2f}%    ECE (without post-hoc calibration): {:.4f}'.format(100. * correct /
                                                   len(test_loader.dataset), ece.item()))

def evaluate_scaling(model):
    model.eval()
    correct = 0
    ece_criterion = ECELoss().cuda()
    best_ece = 1000
    with torch.no_grad():
        for data, target in val_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            output_nosoftmax = model(data)
            output = torch.nn.functional.softmax(output_nosoftmax, dim=1)
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            for i in range(0, 500, 1):
                ece = ece_criterion(temperature_scale(output_nosoftmax, torch.ones(1) * i/100).cuda(), target)
                if ece < best_ece and ece != 0:
                    best_ece = ece
                    best_temp = torch.ones(1) * i/100
        print('\nSearched Temperature on Validation Data: ', round(best_temp.item(),4))
    return best_temp, best_ece

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
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


if __name__ == '__main__':
    main()
