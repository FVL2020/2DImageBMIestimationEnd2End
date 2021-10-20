import numpy as np
import cv2
import random
import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
from utils import *

Detect_Path = '/home/benkesheng/BMI_DETECT/datasets/Image_train'
datasets = ['Image_train', 'Image_test']

def getStat(train_data):
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToPILImage(),
        Resize(224),
        transforms.Pad(224, fill=(255, 255, 255)),
        transforms.CenterCrop(224),
        # argue
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
    ])
    train_dataset = ImageFolder(root = Detect_Path, )