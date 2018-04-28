# -*- coding: utf-8 -*-
"""
@Time    : 2018/4/28 16:58
@Author  : Elvis
"""
"""
 pretrain.py
  
"""
import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.autograd import Variable
import os
import argparse
from utils.data_loader import DataLoader
from models.zsl_resnet import resnet50cnn
from utils.logger import progress_bar

# Learning rate parameters
BASE_LR = 0.01
NUM_CLASSES = 50  # set the number of classes in your dataset
num_train_classes = 40
NUM_ATTR = 75
base_dir = "/home/elvis/data/attribute/ai_challenger_zsl2018_test_b_20180423"
# superclass = "electronics"
# superclass = "vehicles"
superclass = "hairstyles"
DATA_DIR = os.path.join(base_dir, superclass, "pretrain")

BATCH_SIZE = 64
IMAGE_SIZE = 224

MODEL_NAME = "cnn_%s" % superclass
USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = MODEL_NAME + '.pth'

parser = argparse.ArgumentParser(description='PyTorch %s Training' % MODEL_NAME)
parser.add_argument('--lr', default=BASE_LR, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    default=False, help='resume from checkpoint')
parser.add_argument('--data', default=DATA_DIR, type=str, help='file path of the dataset')
args = parser.parse_args()

best_acc = 0.
start_epoch = 0
print("Model: " + MODEL_NAME)
if args.resume:
    print("==> Resuming from checkpoint...")
    MODEL_SAVE_FILE = "cnn_electronics_epoch5.pth"
    checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
    net = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    optimizer = checkpoint["optimizer"]
else:
    print("==> Building model...")
    net = resnet50cnn(num_classes=num_train_classes)

if USE_GPU:
    net.cuda()
    # net = torch.nn.DataParallel(net.module, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

log = open("./log/%s.txt" % MODEL_NAME, 'a')
print("==> Preparing data...")

train_path = os.path.join(DATA_DIR, "train")
val_path = os.path.join(DATA_DIR, "val")
train_loader = DataLoader(data_dir=train_path, image_size=IMAGE_SIZE,
                          batch_size=BATCH_SIZE, trainval='train').data_loader
test_loader = DataLoader(data_dir=val_path, image_size=IMAGE_SIZE,
                         batch_size=BATCH_SIZE, trainval='val').data_loader

criterion = nn.CrossEntropyLoss()


def train(epoch, net, optimizer):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # optimizer = lr_scheduler(optimizer, epoch, init_lr=0.002, decay_epoch=start_epoch)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if USE_GPU:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()

        out = net(inputs)
        loss = criterion(out, targets)
        # loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(train_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    log.write(str(epoch) + ' ' + str(correct / total) + ' ')


def test(epoch, net):
    global best_acc
    net.eval()
    test_loss, correct, total, loss = 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if USE_GPU:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out = net(inputs)
        loss = criterion(out, targets)

        test_loss = loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        acc = 100. * correct / total
        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), acc, correct, total))

    log.write(str(correct / total) + ' ' + str(test_loss) + ' ')

    acc = 100. * correct / total
    if epoch > 2 and acc > best_acc:
        print("Saving checkpoint")
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer
        }
        if not os.path.isdir("checkpoints"):
            os.mkdir('checkpoints')
        torch.save(state, "./checkpoints/" + MODEL_NAME + "_epoch%d.pth" % epoch)
        best_acc = acc


optimizer = optim.Adam(net.parameters(), weight_decay=0.0005)

for epoch in range(start_epoch, 400):
    train(epoch, net, optimizer)
    test(epoch, net)
    log.write("\n")
    log.flush()

log.close()


