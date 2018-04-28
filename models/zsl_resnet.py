# -*- coding: utf-8 -*-
"""
@Time    : 2018/3/24 14:04
@Author  : Elvis
 zsl_resnet.py

"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet50


class AttriCNN(nn.Module):
    def __init__(self, cnn, w_attr, num_attr=123, num_classes=50):
        super(AttriCNN, self).__init__()
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])
        self.feat_size = cnn.fc.in_features

        self.fc0 = nn.Sequential(
            nn.Linear(self.feat_size, num_attr),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.feat_size, num_attr),
            nn.Dropout(0.5),
            nn.Sigmoid(),
        )

        self.fc2 = nn.Linear(num_attr, num_classes, bias=False)
        self.fc2.weight = nn.Parameter(w_attr, requires_grad=False)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.view(feat.shape[0], -1)
        attr = self.fc0(feat)
        wt = self.fc1(feat)
        xt = wt.mul(attr)
        attr_y = self.fc2(xt)  # xt (batch,   square sum root
        return attr_y, wt


class AttriCNN1(nn.Module):
    def __init__(self, cnn, w_attr, num_attr=312, num_classes=200):
        super(AttriCNN1, self).__init__()
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])
        self.feat_size = cnn.fc.in_features

        self.fc1 = nn.Sequential(
            nn.Linear(self.feat_size, num_attr, bias=False),
        )

        self.fc2 = nn.Linear(num_attr, num_classes, bias=False)
        self.fc2.weight = nn.Parameter(w_attr, requires_grad=False)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.view(feat.shape[0], -1)
        xt = self.fc1(feat)
        attr_y = self.fc2(xt)
        return attr_y, (feat, self.fc1[0].weight)


def attrWCNNg(num_attr=123, num_classes=50, superclass="animals"):
    cnn = resnet50(pretrained=False)
    w_attr = np.load("data/%s_attr.npy" % superclass)
    w_attr = torch.FloatTensor(w_attr)  # 312 * 150
    attCNN = AttriCNN(cnn=cnn, w_attr=w_attr, num_attr=num_attr, num_classes=num_classes)
    return attCNN


def attrWCNNg1(num_attr=123, num_classes=50, superclass="animals"):
    cnn = resnet50(pretrained=False)
    w_attr = np.load("data/%s_attr.npy" % superclass)
    w_attr = torch.FloatTensor(w_attr)  # 312 * 150
    attCNN = AttriCNN1(cnn=cnn, w_attr=w_attr, num_attr=num_attr, num_classes=num_classes)
    return attCNN


class RegLoss(nn.Module):
    def __init__(self, lamda1=0.1, lamda2=0.1, superclass="cub"):
        super(RegLoss, self).__init__()
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        wa = np.load("data/%s_attr.npy" % superclass)
        train_ids = np.load("data/%s_train_ids.npy" % superclass)
        test_ids = np.load("data/%s_test_ids.npy" % superclass)
        self.wa_seen = Variable(torch.FloatTensor(wa[train_ids, :]), requires_grad=False).cuda()
        self.wa_unseen = Variable(torch.FloatTensor(wa[test_ids, :]), requires_grad=False).cuda()

    def forward(self, out, targets, w):
        ce = F.cross_entropy(out, targets)
        xt, wt = w
        ws_seen = torch.matmul(self.wa_seen, wt)
        loss = ce + self.lamda1 * torch.mean(torch.mean(ws_seen ** 2, 1))
        return loss
