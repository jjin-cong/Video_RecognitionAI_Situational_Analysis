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
from tqdm import tqdm_notebook


def train_model(model, dataloader_dict, criterion, optimizer, num_epochs, device, CFG):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
        
            epoch_loss = 0.0
            epoch_corrects = 0

            k = 0
            
            for inputs,labels in dataloader_dict[phase]:
                k = k + 1
                #print(k,'/',len(dataloader_dict[phase]))

                batch_videos = cnn.separate_to_windows(inputs, CFG)
                train_batch_videos = cnn.input_C3D(batch_videos)    
                ### normalize 코드 필요, transform에서 넣어줘야할 듯###
                train_batch_videos = train_batch_videos.to(device)

                labels_array = cnn.get_labels(labels) #labels shape을 (batch size, 1)로 변경
                labels = torch.from_numpy(np.int64(labels_array))
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(train_batch_videos)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels) #outputs가 아닌 preds와 labels를 비교

                    if phase == 'train':
                        #check_1 = list(model.parameters())[1]
                        loss.backward()
                        #print(check_1)
                        optimizer.step()

                    #확인 요망 : 파이토치 교과서 264p inputs.size(0)라고 하면 안되지 않나
                    #epoch_loss += loss.item()*batch_videos.size(0) #batch loss의 평균 값을 위해 넣은 코드로 우리의 경우 batch_videos가 들어가야함
                    #epoch_loss += loss.item()*batch_videos.shape[0] #batch loss의 평균 값을 위해 넣은 코드로 우리의 경우 batch_videos가 들어가야함
                    epoch_loss += loss.item() #* len(train_batch_videos)
                    epoch_corrects += torch.sum(preds == labels.data)
                    
            epoch_loss = epoch_loss / (len(train_batch_videos) * k) #한 배치당 프레임 수로 나눔
            epoch_acc = epoch_corrects.double() / (len(train_batch_videos) * k) #한 배치당 프레임 수로 나눔

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model