import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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

from LFW_dataset import LFW_dataset
from network import Net
from alexnet import Alexnet
from tqdm import tqdm
from torch.autograd import Variable

# preprocessing the data
def preprocessing(images_dict):
    # 400 persons are in the test set and 400 persons are included in the
    # validation set. The remaining is used for training.
    count = 0
    for line in open("images_list.txt","r"):
        line = line.strip()
        person = line.split("/")[-2]
        if person not in images_dict:
            images_dict[person] = [line]
        else:
            images_dict[person].append(line)
        count += 1

    print("Number of unique persons = ", str(len(images_dict)))
    print("Number of total images = ", str(count))
    unique_ids = list(images_dict.keys())
    val_ids = unique_ids[-800:-400]
    test_ids = unique_ids[-400:]
    train_ids = unique_ids[:-800]
    return val_ids, train_ids, test_ids

def corr2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(nn.Conv2d(x, k) for x, k in zip(X, K))

def train(model,device,train_loader,optimizer, loss_func, num_epochs):
    for epoch in tqdm(range(num_epochs)):
        training_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = Variable(labels.float()) 
            # zeroes the gradient buffers of all parameters
            optimizer.zero_grad()
            # forward pass
            outputs = model(images) 
            # calculate the loss
            # loss = F.cross_entropy(outputs, labels)
            loss = loss_func(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # Does the update based on the calculated gradients
            optimizer.step()
            training_loss += loss.item()
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
        training_loss /= len(train_loader)
    return training_loss

# def test(model,device,test_loader):

#     return test_loss

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

def visualize_img(train_data):
    # obtain one batch of training images
    dataiter = iter(train_data)
    images, labels = dataiter.next()
    images = images.numpy() # convert images to numpy for display
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(10, 4))
    classes = [0,1]
    # display 20 images
    for idx in np.arange(10):
        ax = fig.add_subplot(2, 10/2, idx+1, xticks=[], yticks=[])
        imshow(images[idx][:2])
        ax.set_title(classes[labels[idx]])

def main():
    images_dict = {}
    batch_size = 32
    num_epochs = 30
    val_ids, train_ids, test_ids = preprocessing(images_dict)
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = LFW_dataset("train", images_dict, train_ids, train_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
    # val_dataset = LFW_dataset(split="val", )
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=32)
    # test_dataset = LFW_dataset(split="test", images_dict=images_dict, ids=test_ids)
    # test_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32)
    start_time = time.time()
        
        
    # train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    # model = Alexnet().to(device)
    # train_dir = './lfw/'
    # dataset_train = datasets.ImageFolder(train_dir, transform=train_transform)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    # visualize_img(train_dataloader)
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    loss_func = nn.MSELoss()
    # loss_func = nn.CrossEntropyLoss()
    training_loss = train(model,device,train_dataloader ,optimizer, loss_func, num_epochs)
    print('training loss = ', training_loss)
    # end_time = time.time()
    # elasped_time = end_time-start_time
    # print("Elasped time = ", elasped_time)

if __name__ == "__main__":
    main()