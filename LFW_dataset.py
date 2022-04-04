import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import time
import torchvision.transforms as transforms

class LFW_dataset(torch.utils.data.Dataset):

  def __init__(self, split, images_dict, ids, transform):
    self.split = split
    self.images_dict = images_dict
    self.ids = ids
    self.transform = transform

  def __getitem__(self, index):
    id1 = self.ids[index]
    if len(self.images_dict[id1]) == 1:
      id2 = np.random.randint(0, len(self.ids))
      id2 = self.ids[id2]
      label = 0
    else:
      id2 = id1
      label = 1
    img1 = Image.open(self.images_dict[id1][0])
    img2 = Image.open(random.sample(self.images_dict[id2], 1)[0])
    # img1 = transforms.ToTensor()(img1)
    # img2 = transforms.ToTensor()(img2)
    img1 = self.transform(img1)
    img2 = self.transform(img2)
    if label == 0:
      label = torch.Tensor([1, 0])
    else:
      label = torch.Tensor([0, 1])
    img = torch.cat((img1, img2), 0)
    return img, label

  def __len__(self):
    return len(self.ids)

