# -*- coding: utf-8 -*-
"""
@Time    : 2018/3/23 17:57
@Author  : Elvis
"""
"""
 data_loader.py
  
"""
from torchvision import datasets, transforms
import os
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class DataLoader(object):
    def __init__(self, data_dir, image_size, batch_size, trainval='train'):
        """
        this class is the normalize data loader of PyTorch.
        The target image size and transforms can edit here.
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.trainval = trainval

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # torchsample.transforms.Rotate(10),
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            ]),
        }
        self._init_data_sets()

    def _init_data_sets(self):
        self.dset = datasets.ImageFolder(self.data_dir, self.data_transforms[self.trainval])
        self.data_loader = torch.utils.data.DataLoader(
            self.dset, batch_size=self.batch_size, pin_memory=True,
            shuffle=(self.trainval == 'train'), num_workers=4)
        self.data_sizes = len(self.dset)
        self.data_classes = self.dset.classes


class TestDataset(Dataset):
    """Dataset for test folder

    Arguments:
        Path to image folder
        PIL transforms
    """

    def __init__(self, img_dir, image_size, transform=None):
        self.img_dir = img_dir
        self.image_size = image_size
        self.transform = transform
        self.imgs = sorted(os.listdir(self.img_dir))

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.imgs[index]))
        img = img.convert('RGB')
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        img = self.transform(img)
        return img, self.imgs[index]

    def __len__(self):
        return len(self.imgs)
