import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.nn.modules import loss

from PIL import Image
from torch.autograd import Variable
from dataset import *

# All models
#reactnet-18
from models.Qa_reactnet_18_bn import birealnet18 as Qa_reactnet_18_bn
from models.Qa_reactnet_18_none import birealnet18 as Qa_reactnet_18_none
from models.Qa_reactnet_18_bf import birealnet18 as Qa_reactnet_18_bf

from models.Qaw_reactnet_18_bn import birealnet18 as Qaw_reactnet_18_bn
from models.Qaw_reactnet_18_none import birealnet18 as Qaw_reactnet_18_none
from models.Qaw_reactnet_18_bf import birealnet18 as Qaw_reactnet_18_bf

#reactnet-A
from models.Qa_reactnet_A_bn import reactnet as Qa_reactnet_A_bn
from models.Qa_reactnet_A_none import reactnet as Qa_reactnet_A_none
from models.Qa_reactnet_A_bf import reactnet as Qa_reactnet_A_bf

from models.Qaw_reactnet_A_bn import reactnet as Qaw_reactnet_A_bn
from models.Qaw_reactnet_A_none import reactnet as Qaw_reactnet_A_none
from models.Qaw_reactnet_A_bf import reactnet as Qaw_reactnet_A_bf


#lighting data augmentation
imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

#label smooth
class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class DistributionLoss(loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher output.

    output must be a pair of (model_output, real_output), both NxC tensors.
    The rows of real_output must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, model_output, real_output):

        self.size_average = True

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        if real_output.requires_grad:
            raise ValueError("real network output should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        real_output_soft = F.softmax(real_output, dim=1)
        del model_output, real_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        if self.size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        # model_output_log_prob = model_output_log_prob.squeeze(2)
        return cross_entropy_loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_checkpoint(state, is_best, save):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def setup_model_dataloader(args):

    # Dataset 
    if_imagenet = False
    
    if args.dataset == 'imagenet':
        print('* Dataset = ImageNet')
        train_loader, val_loader = imagenet_dataloaders(args.batch_size, args.data, args.workers)
        classes = 1000
        if_imagenet = True
    
    elif args.dataset == 'cifar10':
        print('* Dataset = CIFAR10')
        train_loader, val_loader = cifar10_dataloaders(args.batch_size, args.data, args.workers)
        classes = 10
    
    elif args.dataset == 'cifar100':
        print('* Dataset = CIFAR100')
        train_loader, val_loader = cifar100_dataloaders(args.batch_size, args.data, args.workers)
        classes = 100
    
    else:
        raise ValueError('unknow dataset')

    # architecture
    if args.arch == 'reactnet-18':
        print('* Model = ReActNet-18')
        if args.binary_w:
            print('* Binarize both activation and weights')
            if args.bn_type == 'bn':
                print('* with BN')
                model = Qaw_reactnet_18_bn(num_classes=classes, imagenet=if_imagenet)
            elif args.bn_type == 'none':
                print('* without BN')
                model = Qaw_reactnet_18_none(num_classes=classes, imagenet=if_imagenet)
            elif args.bn_type == 'bf':
                print('* BN-Free')
                model = Qaw_reactnet_18_bf(num_classes=classes, imagenet=if_imagenet)

        else:
            print('* Binarize only activation')
            if args.bn_type == 'bn':
                print('* with BN')
                model = Qa_reactnet_18_bn(num_classes=classes, imagenet=if_imagenet)
            elif args.bn_type == 'none':
                print('* without BN')
                model = Qa_reactnet_18_none(num_classes=classes, imagenet=if_imagenet)
            elif args.bn_type == 'bf':
                print('* BN-Free')
                model = Qa_reactnet_18_bf(num_classes=classes, imagenet=if_imagenet)


    elif args.arch == 'reactnet-A':
        print('* Model = reactnet-A')
        if args.binary_w:
            print('* Binarize both activation and weights')
            if args.bn_type == 'bn':
                print('* with BN')
                model = Qaw_reactnet_A_bn(num_classes=classes)
            elif args.bn_type == 'none':
                print('* without BN')
                model = Qaw_reactnet_A_none(num_classes=classes)
            elif args.bn_type == 'bf':
                print('* BN-Free')
                model = Qaw_reactnet_A_bf(num_classes=classes)

        else:
            print('* Binarize only activation')
            if args.bn_type == 'bn':
                print('* with BN')
                model = Qa_reactnet_A_bn(num_classes=classes)
            elif args.bn_type == 'none':
                print('* without BN')
                model = Qa_reactnet_A_none(num_classes=classes)
            elif args.bn_type == 'bf':
                print('* BN-Free')
                model = Qa_reactnet_A_bf(num_classes=classes)

    return train_loader, val_loader, model, classes



