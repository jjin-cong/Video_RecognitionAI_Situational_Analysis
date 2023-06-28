import data_loader
import cnn
import classifier
import training

import sys, os

import numpy as np
import collections
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch import optim
from torch.autograd import Variable
from torch import nn
import time

from tqdm import tqdm_notebook as tqdm

def train_model(model, dataloader_dict, criterion, optimizer, num_epochs, device, CFG):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*20)

        # for phase in ['train']: 
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
        
            epoch_loss = 0.0
            epoch_corrects = 0
        
            for inputs,labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                # 야매코드 -> Label을 Tensor로 처음부터 store 하는 코드 완성되면 지우고 밑에 uncomment 하기
                labels_array = np.zeros((len(labels), len(labels[0])))
                for i in range(len(labels)):
                    video_array = np.array([int(j) for j in labels[i]])
                    labels_array[i,:] = video_array
                labels_array = np.reshape(labels_array, (50,))
                labels = torch.from_numpy(labels_array)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    batch_videos = cnn.separate_to_windows(inputs, CFG)
                    train_batch_videos = cnn.input_C3D(batch_videos)
                    model.to(device)
                    outputs = model(train_batch_videos)
                    outputs = torch.from_numpy(outputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.type(torch.LongTensor))
                    loss = Variable(loss, requires_grad = True)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item()*inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model