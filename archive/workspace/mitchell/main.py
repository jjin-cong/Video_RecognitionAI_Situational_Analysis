# data_loader.py, cnn.py, classifier.py 에 있는 function들을 불러와 training 및 testing을 진행합니다
# .py를 쓸지 .ipynb를 쓸지 정하기

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


"""
Data Loading Part
"""

path_data = "C:/Users/user/OneDrive - goe.go.kr/문서/Alzalttakkalbeul/data" #개인 파일 위치 입력
path_data_prepared = os.path.join(path_data, "videos_prepared")
sys.path.append(path_data)
sys.path.append(path_data_prepared)

train_data = data_loader.DashcamDataset(
    video_dir = path_data_prepared,
)
test_data = data_loader.DashcamDataset(
    video_dir = path_data_prepared,
    train = False
)
train_dataloader = DataLoader(train_data, batch_size=5, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=5, shuffle = False)
dataloader_dict = {'train': train_dataloader, 'val': test_dataloader}


"""
CNN Part
"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'VIDEO_LENGTH':25 * 4, # 25프레임 * 4초
    'IMG_SIZE':128,
    'WINDOW_SIZE' : 10
}

"""
Training Part
"""

model = cnn.BaseModel()
model = model.to(device)

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr = 0.3, momentum = 0.9)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

num_epochs = 5
model = training.train_model(model, dataloader_dict, criterion, optimizer, num_epochs, device, CFG)