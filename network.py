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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 6 input image channel, 16 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(6, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        # an affine operation: y = Wx + b
        # dimension=(dimen_of_input_image- Filter_size(int)+(2*padding))/stride_value + 1
        self.fc1 = nn.Linear(32 * 13 * 13, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x