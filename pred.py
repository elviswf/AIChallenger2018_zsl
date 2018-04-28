# -*- coding: utf-8 -*-
"""
@Time    : 2018/3/23 21:53
@Author  : Elvis
"""
import os
import numpy as np
from utils.data_loader import TestDataset
import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.autograd import Variable
from tqdm import tqdm

superclass = "vehicles"
NUM_CLASSES = 50  # set the number of classes in your dataset
num_train_classes = 40
# base_dir = "/home/elvis/data/attribute/ai_challenger_zsl2018_train_test_a_20180321"
base_dir = "/home/elvis/data/attribute/ai_challenger_zsl2018_test_b_20180423"
# superclass = "animals"
spcls = superclass[0].upper()
test_path = os.path.join(base_dir, "zsl_b_%s_test_20180321" % superclass)

IMAGE_SIZE = 224
gamma = 10
USE_GPU = torch.cuda.is_available()

MODEL_NAME = "zsl_%s_g1" % superclass
# MODEL_SAVE_FILE = MODEL_NAME + '.pth'
MODEL_SAVE_FILE = "zsl_vehicles_g1_epoch3.pt"   # choose a checkpoint to make predictions

print("Model: " + MODEL_NAME)
print("==> Resuming from checkpoint...")
checkpoint = torch.load("./checkpoints/" + MODEL_SAVE_FILE)
net = checkpoint["net"]
best_acc = checkpoint["acc"]
start_epoch = checkpoint["epoch"]
optimizer = checkpoint["optimizer"]
print("start_epoch: %d" % start_epoch)
print("acc: %f" % best_acc)

if USE_GPU:
    net.cuda()
    # net = torch.nn.DataParallel(net.module, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

print("==> Preparing data...")

test_dset = TestDataset(img_dir=test_path, image_size=IMAGE_SIZE)
test_loader = DataLoader(test_dset, batch_size=256, shuffle=False, num_workers=1)
# pin_memory=True # CUDA only

net.eval()

train_ids = np.load("data/%s_train_ids.npy" % superclass)
test_ids = np.load("data/%s_test_ids.npy" % superclass)
imgs_list = []
preds_list = []

for batch_idx, (inputs, img_names) in tqdm(enumerate(test_loader)):
    if USE_GPU:
        inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    out, attr = net(inputs)
    logit = out.data.cpu()
    # logit = torch.nn.functional.softmax(Variable(logit), dim=1).data
    _, predicted = torch.max(logit, 1)
    seen_prob, seen_class = torch.max(logit[:, train_ids], 1)
    unseen_prob, unseen_class = torch.max(logit[:, test_ids], 1)
    for i, spi in enumerate(seen_prob):
        predicted[i] = int(test_ids[unseen_class[i]])
        # if seen_prob[i] < unseen_prob[i] * gamma:
        #     predicted[i] = int(test_ids[unseen_class[i]])

    # pred_list = predicted
    for pred, img_name in zip(predicted, img_names):
        imgs_list.append(img_name)
        preds_list.append("Label_%s_%02d" % (spcls, int(pred) + 1))

with open("data/preds_%s.txt" % spcls, "a") as fw:
    for img_name, pred in zip(imgs_list, preds_list):
        line = img_name + " " + pred + "\n"
        fw.write(line)


