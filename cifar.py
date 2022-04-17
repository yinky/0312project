import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

#regularization=正则化, standardization=标准化, normalize=归一化
#transforms.Normalize()对图片里的每个像素值进行归一化,将三维张量(RGB，范围都是0~255)值映射到三维[-1,1]
#parameters是经验值，该分布来自于imageNet数据集.
#imageNet数据集有100万张图片, 1000个类, 数据集很大，可以消除过拟合，这也是深度神经网络火的原因。
#image=(image-mean)/std, mean为均值, std为标准差
cifar10_mean = (0.4914,0.4822,0.4465)
cifar10_std = (0.2471,0.2435,0.2616)
cifar100_mean = (0.5071,0.4867,0.4408)
cifar100_std = (0.2675,0.2565,0.2761)
mnist_mean = (0.1307,0.1307,0.1307)
mnist_std = (0.3081,0.3081,0.3081)
svhn_mean = (0.485,0.456,0.406)
svhn_std = (0.229,0.224,0.225)
normal_mean = (0.5,0.5,0.5)
normal_std = (0.5,0.5,0.5)


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, val_idx = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    val_dataset = CIFAR10SSL(
        root, val_idx, train=True,
        transform=transform_labeled)

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, val_dataset


def get_cifar100(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(root, train=True, download=True)

    # 添加val_idx, x_u_split返回的是3个参数
    train_labeled_idxs, train_unlabeled_idxs, val_idx = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    # 添加
    val_dataset = CIFAR100SSL(
        root, val_idx, train=True,
        transform=transform_labeled)

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, val_dataset


def get_mnist(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28,
                              padding=int(28*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])

    base_dataset = datasets.MNIST(root, train=True, download=False)

    if args.full == False:
        train_labeled_idxs, train_unlabeled_idxs, val_idx = x_u_split(
            args, base_dataset.targets)

        train_labeled_dataset = MNISTSSL(
            root, train_labeled_idxs, train=True,
            transform=transform_labeled)
        train_unlabeled_dataset = MNISTSSL(
            root, train_unlabeled_idxs, train=True,
            transform=TransformFixMatchMNIST(mean=mnist_mean, std=mnist_std))
        val_dataset = MNISTSSL(
            root, val_idx, train=True,
            transform=transform_labeled)
        test_dataset = datasets.MNIST(
            root, train=False, download=False,
            transform=transform_val)

        train_labeled_dataset.data = train_labeled_dataset.data.numpy()
        train_unlabeled_dataset.data = train_unlabeled_dataset.data.numpy()
        val_dataset.data = val_dataset.data.numpy()

        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, val_dataset
    else:
        train_labeled_idxs = np.array(range(len(base_dataset.targets)))

        train_labeled_dataset = MNISTSSL(
            root, train_labeled_idxs, train=True,
            transform=transform_labeled)
        test_dataset = datasets.MNIST(
            root, train=False, download=False,
            transform=transform_val)

        train_labeled_dataset.data = train_labeled_dataset.data.numpy()
        return train_labeled_dataset, test_dataset


def get_svhn(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std=svhn_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std=svhn_std)
    ])

    base_dataset = datasets.SVHN(root, download=True)

    if args.full == False:
        train_labeled_idxs, train_unlabeled_idxs, val_idx = x_u_split(
            args, base_dataset.labels)

        train_labeled_dataset = SVHNSSL(
            root, train_labeled_idxs, train=True,
            transform=transform_labeled)
        train_unlabeled_dataset = SVHNSSL(
            root, train_unlabeled_idxs, train=True,
            transform=TransformFixMatch(mean=svhn_mean, std=svhn_std))
        val_dataset = SVHNSSL(
            root, val_idx, train=True,
            transform=transform_labeled)
        test_dataset = datasets.SVHN(
            root, split='test', download=True,
            transform=transform_val)

        train_labeled_dataset.data = np.swapaxes(train_labeled_dataset.data, 1, 2)
        train_labeled_dataset.data = np.swapaxes(train_labeled_dataset.data, 2, 3)
        train_unlabeled_dataset.data = np.swapaxes(train_unlabeled_dataset.data, 1, 2)
        train_unlabeled_dataset.data = np.swapaxes(train_unlabeled_dataset.data, 2, 3)
        val_dataset.data = np.swapaxes(val_dataset.data, 1, 2)
        val_dataset.data = np.swapaxes(val_dataset.data, 2, 3)

        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, val_dataset
    else:
        train_labeled_idxs = np.array(range(len(base_dataset.labels)))

        train_labeled_dataset = SVHNSSL(
            root, train_labeled_idxs, train=True,
            transform=transform_labeled)

        test_dataset = datasets.SVHN(
            root, split='test', download=True,
            transform=transform_val)

        train_labeled_dataset.data = np.swapaxes(train_labeled_dataset.data, 1, 2)
        train_labeled_dataset.data = np.swapaxes(train_labeled_dataset.data, 2, 3)
        return train_labeled_dataset, test_dataset

def x_u_split(args, labels):

    # 新加, 因为貌似找不到将num_classes通过args传入的方法
    if args.dataset == 'cifar10':
        num_classes = 10# 数据集的类别数量。
    elif args.dataset == 'cifar100':
        num_classes = 100

    label_per_class = args.num_labeled // num_classes # 保证每个类的label数量相同
    labels = np.array(labels)# 将列表转化为数组?
    labeled_idx = []
    unlabeled_idx = []
    #unlabeled_idx = np.array(range(len(labels)))# unlabelled=全部的数据集
    val_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    for i in range(num_classes):
        idx = np.where(labels == i)[0] #training集合第i类的左端点
        #idx = np.random.choice(idx, label_per_class + 100, False)
        idx = np.random.choice(idx, 50000//num_classes, False)
        
        labeled_idx.extend(idx[:label_per_class])
        unlabeled_idx.extend(idx[:45000//num_classes])
        val_idx.extend(idx[45000//num_classes:])

    labeled_idx = np.array(labeled_idx)# 将列表转化为数组?
    unlabeled_idx = np.array(unlabeled_idx)
    val_idx = np.array(val_idx)

     #如果随机选择有标签的总数量!=args.num_labeled，则报异常。注意args.num_labeled应设置为类别的倍数
    assert len(labeled_idx) == args.num_labeled
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(args.batch_size * args.eval_step / args.num_labeled)# math.ceil为小数上浮取整函数
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)# 打乱labeled次序, 若为多维数组shuffle只打乱第1维

    return labeled_idx, unlabeled_idx, val_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class TransformFixMatchMNIST(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=int(28*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=int(28*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNISTSSL(datasets.MNIST):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class SVHNSSL(datasets.SVHN):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'mnist': get_mnist,
                   'svhn': get_svhn}
