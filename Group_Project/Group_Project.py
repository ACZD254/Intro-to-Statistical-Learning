"""
@author: Soumitra Pandit
"""
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
from tqdm.notebook import tqdm
import torchvision.transforms as T
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import random
import PIL

import matplotlib.pyplot as plt

def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
seed_everything(42)
print('ENVIRONMENT READY')

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tfms = T.Compose([
    T.RandomCrop(128, padding=8, padding_mode='reflect'),
     #T.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    T.Resize((128, 128)),
    T.RandomHorizontalFlip(), 
    T.RandomRotation(10),
    T.ToTensor(), 
     T.Normalize(*imagenet_stats,inplace=True), 
    #T.RandomErasing(inplace=True)
])

valid_tfms = T.Compose([
     T.Resize((128, 128)), 
    T.ToTensor(), 
     T.Normalize(*imagenet_stats)
])

dataset = ImageFolder(root = r"C:\Users\smtrp\OneDrive\Desktop\DS502\ISLR\Group_Project\Segmented_Data")

dataset_size = len(dataset)
print(dataset_size)

dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A',
       11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K',
       21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U',
       31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e',
       41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 
       51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y',
       61: 'z'}

test_size = 200
nontest_size = len(dataset) - test_size

nontest_df, test_df = random_split(dataset, [nontest_size, test_size])
print(len(nontest_df), len(test_df))