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

# 新加的头文件
from cifar import DATASET_GETTERS
import torch.nn.functional as F

import resnet as resnet
from ece_loss import ECELoss
from temperature_scaling import ModelWithTemperature

model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet") and callable(resnet.__dict__[name]))

# 超参数的添加顺序: 
# 1.创建parser: parser=argparse.ArgumentParser(description='xxx')
# 2.添加参数: parser.add_argument(...)
# 3.再创建args: args=parser.parse_args()
# 4.main(args)

parser = argparse.ArgumentParser(description='CIFAR10/100')# 数据集选择，已尝试过CIFAR10, CIFAR100未尝试
# --arch resnet152会出现CUDA内存不足的情况,why?
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', #个人理解为线程数，在SEU超算中心上，应设置为0或3。
                    '--workers',
                    default=3,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', #执行多少轮，每一轮会把所有的图像都执行一次。
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', #每个epoch中，一次喂多少张图像。缺省为有监督+无监督=64+448=512
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 512)')
parser.add_argument('--lr',# 会随着epoch进行而衰减，缺省是是第1~100轮0.1, 第101-150轮0.01, 第151-200轮 0.001。
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
parser.add_argument('--weight-decay',# 改动过?
                    '--wd',
                    default=5e-4,
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
parser.add_argument('--method', #选择什么样的loss函数，缺省为交叉熵
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
parser.add_argument('--seed',# 随机数种子，若固定可以得到确切的实验结果。
                    default=101,
                    type=int,
                    help='seed for validation data split')
parser.add_argument('--num-labeled',
                    default=4000,
                    type=int,
                    help='number of labeled data')
parser.add_argument("--expand-labels", action="store_true",
                    help="expand labels to fit eval steps")
parser.add_argument('--eval-step', default=1024, type=int,
                    help='number of eval steps to run')

args = parser.parse_args()
print(args)# 将超参数的信息打印在屏幕上

if args.dataset == 'cifar10':
    num_classes = 10# 数据集的类别数量。
elif args.dataset == 'cifar100':
    num_classes = 100

labeled_dataset, unlabeled_dataset, test_dataset, val_dataset = DATASET_GETTERS[args.dataset](args, '../../data')
train_sampler = torch.utils.data.RandomSampler

train_labeled_loader = torch.utils.data.DataLoader(labeled_dataset,
    sampler=train_sampler(labeled_dataset),
    batch_size=args.batch_size,
    num_workers=args.workers,
    pin_memory=True,
    drop_last=True# 设为true时，将结尾不满一个batch的数据直接丢弃
    )

train_unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset,
    sampler=train_sampler(unlabeled_dataset),#train_unlabeled_data,
    batch_size=args.batch_size,
    num_workers=args.workers,
    pin_memory=True,
    drop_last=True
    )

test_loader = torch.utils.data.DataLoader(test_dataset,
    sampler=torch.utils.data.SequentialSampler(test_dataset),
    batch_size=8*args.batch_size,
    num_workers=args.workers,
    pin_memory=True
    )

val_loader = torch.utils.data.DataLoader(val_dataset,
    sampler=torch.utils.data.SequentialSampler(val_dataset),
    batch_size=8*args.batch_size,
    num_workers=args.workers,
    pin_memory=True
    )

def main():
    model = resnet.__dict__[args.arch](num_classes=num_classes)#模型的基础参数，来源于超参数
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()#loss函数->交叉熵损失函数

    optimizer = torch.optim.SGD(model.parameters(),# 优化器->随机梯度下降
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    num_epoch = args.epochs #num_epoch从超参数中读入

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], last_epoch= -1)# 关键点的学习率*0.1

    for epoch in range(0, num_epoch):# train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))# 输出当前的学习率
        train(model, criterion, optimizer, epoch)# train函数为核心函数
        lr_scheduler.step()# ?
        evaluate(model)# 计算准确率。
        
        if epoch >= num_epoch-10:
            evaluate_TS(model)

'''
def interleave(x, size):#该函数重新定义了数组的长和宽，有何作用？
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size): #该函数重新定义了数组的长和宽，有何作用？
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
'''

def train(model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()#AverageMeter类定义在main.py最后
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()# switch to train model
    end = time.time()# 记录每个batch的开始时间

    #code by Fixmatch
    labeled = iter(train_labeled_loader)
    unlabeled = iter(train_unlabeled_loader)
    for i in range(len(train_unlabeled_loader)):
        try:
            inputs_s, targets_s = labeled.next()
        except StopIteration:
            labeled = iter(train_labeled_loader)
            inputs_s, targets_s = labeled.next()

        (inputs_u_w, inputs_u_s), _ = unlabeled.next()
        if torch.cuda.is_available():#如果有GPU，则需把要计算的数据放到GPU上
            inputs_s, targets_s = inputs_s.cuda(), targets_s.cuda()
            inputs_u_w, inputs_u_s = inputs_u_w.cuda(), inputs_u_s.cuda()
        BatchSize = targets_s.size()[0]
        #inputs = interleave(torch.cat((inputs_s, inputs_u_w, inputs_u_s)), 15)#有标记:无标记(弱):无标记(强)1:7:7
        inputs = torch.cat((inputs_s, inputs_u_w, inputs_u_s))

        logits = model(inputs)
        #logits = de_interleave(logits, 15)
        logits_s = logits[:BatchSize]
        logits_u_w, logits_u_s = logits[BatchSize:].chunk(2)
        del logits

        Ls = F.cross_entropy(logits_s, targets_s, reduction='mean')
        '''
        pseudo_label = torch.softmax(logits_u_w.detach()/1, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(0.95).float()#弱增强置信度超过0.95的图片记为true
        Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
        '''
        pseudo_label = torch.softmax(logits_u_w.detach()/1, dim=-1).detach()
        consistency_criterion = nn.KLDivLoss(reduction='batchmean').cuda() #"wdb: 这句可以放到函数外面作为全局变量"?
        Lu = consistency_criterion(logits_u_s, pseudo_label)

        loss = Ls + 1 * Lu #损失函数, 形式由Fixmatch文章中给出

        optimizer.zero_grad()#反向传播之前需要先清零
        loss.backward()#loss是一个tensor，默认相加
        optimizer.step()
        
        output = logits_s.float()
        loss = loss.float()
        prec1 = accuracy(output.data, targets_s)[0]
        losses.update(loss.item(), inputs_s.size(0))

        top1.update(prec1.item(), inputs_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)#train函数每个batch的运行时间=结束时间-开始时间
        end = time.time()# 重新记录开始时间

        if i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' # train函数(每个batch)运行时间
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' # 貌似无用
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' #平均loss值
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format( #准确率
                      epoch,
                      i,
                      len(train_unlabeled_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1))

def evaluate(model):
    model.eval()
    correct = 0 #模型分类正确的图片数量
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

if __name__ == '__main__':
    main()
