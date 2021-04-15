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
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed

import timm 
from utils import *
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.models as models
from agc import adaptive_clip_grad

parser = argparse.ArgumentParser("normalize free BNN")
################################# basic settings ###################################### 
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--arch', type=str, default='reactnet', help='architecture')
parser.add_argument('--bn_type', type=str, default='bn', help='[w/w.o bn or nf-module]')
parser.add_argument('--binary_w', action="store_true", help="whether binarize weight")
parser.add_argument('--resume', action="store_true", help="whether resume training")
parser.add_argument('--pretrained', type=str, default=None, help='pretrained weight')
parser.add_argument('--loss_type', type=str, default='kd', help='[kd, ce, ls]')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--teacher', type=str, default='resnet34', help='path of ImageNet')
parser.add_argument('--teacher_weight', type=str, default=None, help='pretrained teacher weight')
################################# training settings ###################################### 
parser.add_argument('--epochs', type=int, default=120, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--agc', action="store_true", help="whether using agc")
parser.add_argument('--clip_value', type=float, default=0.04, help='lambda for AGC')
################################# other settings ###################################### 
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N', help='number of data loading workers (default: 4)')

args = parser.parse_args()

def main():
    global args
    print(args)

    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled=True

    train_loader, val_loader, model_student, CLASSES = setup_model_dataloader(args)
    model_student = nn.DataParallel(model_student).cuda()
    print(model_student)

    # load teacher model
    if args.loss_type == 'kd':
        print('* Loading teacher model')
        if not 'nfnet' in args.teacher:
            model_teacher = models.__dict__[args.teacher](pretrained=True)
            classes_in_teacher = model_teacher.fc.out_features
            num_features = model_teacher.fc.in_features
        else:
            model_teacher = timm.create_model(args.teacher, pretrained=True)
            classes_in_teacher = model_teacher.head.fc.out_features
            num_features = model_teacher.head.fc.in_features

        if not classes_in_teacher == CLASSES:
            print('* change fc layers in teacher')
            if not 'nfnet' in args.teacher:
                model_teacher.fc = nn.Linear(num_features, CLASSES)
            else:
                model_teacher.head.fc = nn.Linear(num_features, CLASSES)
            print('* loading pretrained teacher weight from {}'.format(args.teacher_weight))
            pretrain_teacher = torch.load(args.teacher_weight, map_location='cpu')['state_dict']
            model_teacher.load_state_dict(pretrain_teacher)

        model_teacher = nn.DataParallel(model_teacher).cuda()
        for p in model_teacher.parameters():
            p.requires_grad = False
        model_teacher.eval()


    #criterion
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth).cuda()
    criterion_kd = DistributionLoss()

    #optimizer
    all_parameters = model_student.parameters()
    weight_parameters = []
    for pname, p in model_student.named_parameters():
        if p.ndimension() == 4 or 'conv' in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=args.learning_rate,)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
    start_epoch = 0
    best_top1_acc= 0

    if args.pretrained:
        print('* loading pretrained weight {}'.format(args.pretrained))
        pretrain_student = torch.load(args.pretrained)
        if 'state_dict' in pretrain_student.keys():
            pretrain_student = pretrain_student['state_dict']

        for key in pretrain_student.keys():
            if not key in model_student.state_dict().keys():
                print('unload key: {}'.format(key))

        model_student.load_state_dict(pretrain_student, strict=False)

    if args.resume:
        checkpoint_tar = os.path.join(args.save, 'checkpoint.pth.tar')
        if os.path.exists(checkpoint_tar):
            print('loading checkpoint {} ..........'.format(checkpoint_tar))
            checkpoint = torch.load(checkpoint_tar)
            start_epoch = checkpoint['epoch']
            best_top1_acc = checkpoint['best_top1_acc']
            model_student.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))
        else:
            raise ValueError('no checkpoint for resume')

    if args.loss_type == 'kd':
        if not classes_in_teacher == CLASSES:
            validate('teacher', val_loader, model_teacher, criterion, args)

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:

        if args.loss_type == 'kd':
            train_obj, train_top1_acc, train_top5_acc = train_kd(epoch, train_loader, model_student, model_teacher, criterion_kd, optimizer, scheduler)
        elif args.loss_type == 'ce':
            train_obj, train_top1_acc, train_top5_acc = train(epoch, train_loader, model_student, criterion, optimizer, scheduler)
        elif args.loss_type == 'ls':
            train_obj, train_top1_acc, train_top5_acc = train(epoch, train_loader, model_student, criterion_smooth, optimizer, scheduler)
        else:
            raise ValueError('unsupport loss_type')
        
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model_student, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model_student.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            }, is_best, args.save)

        epoch += 1

    training_time = (time.time() - start_t) / 3600
    print('total training time = {} hours'.format(training_time))
    print('* best acc = {}'.format(best_top1_acc))


def train_kd(epoch, train_loader, model_student, model_teacher, criterion, optimizer, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model_student.train()
    model_teacher.eval()
    end = time.time()
    scheduler.step()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits_student = model_student(images)
        logits_teacher = model_teacher(images)
        loss = criterion(logits_student, logits_teacher)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # clip gradient if necessary
        if args.agc:
            parameters_list = []
            for name, p in model_student.named_parameters():
                if not 'fc' in name:
                    parameters_list.append(p)
            adaptive_clip_grad(parameters_list, clip_factor=args.clip_value)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i%50 == 0:
            progress.display(i)

    return losses.avg, top1.avg, top5.avg

def train(epoch, train_loader, model_student, criterion, optimizer, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model_student.train()
    end = time.time()
    scheduler.step()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits_student = model_student(images)
        loss = criterion(logits_student, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # clip gradient if necessary
        if args.agc:
            parameters_list = []
            for name, p in model_student.named_parameters():
                if not 'fc' in name:
                    parameters_list.append(p)
            adaptive_clip_grad(parameters_list, clip_factor=args.clip_value)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i%50 == 0:
            progress.display(i)

    return losses.avg, top1.avg, top5.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i%50 == 0:
                progress.display(i)

        print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

if __name__ == '__main__':
    main()
