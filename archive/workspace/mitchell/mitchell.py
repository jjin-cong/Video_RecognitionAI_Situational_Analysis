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

# path_data_prepared = "data/videos_prepared/"

# path_pardir = os.pardir # ..
# path_data = os.path.join(path_pardir, "data") # ../data
path_data = "data"
path_data_prepared = os.path.join(path_data, "videos_prepared")
# sys.path.append(path_pardir)
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
test_dataloader = DataLoader(test_data, batch_size = 5, shuffle = False)
dataloader_dict = {'train': train_dataloader, 'val': test_dataloader}

iterator = iter(train_dataloader) # next(iterator)할 때마다 배치 하나씩 받을 수 있음

video_batch1, class_batch1 = next(iterator) # batch 1
# video_batch2, class_batch2 = next(iterator) # batch 1
# video_batch3, class_batch3 = next(iterator) # batch 1
# video_batch4, class_batch4 = next(iterator) # batch 1
# video_batch5, class_batch5 = next(iterator) # batch 1
# video_batch6, class_batch6 = next(iterator) # batch 1

print(video_batch1.shape)
# print(video_batch2.shape)
# print(video_batch3.shape)
# print(video_batch4.shape)
# print(video_batch5.shape)
# print(video_batch6.shape)

# 배치 별 이미지 2장씩 뽑아보기...

# img1of1 = video_batch1.permute(0,2,1,3,4)[0][0].permute(1,2,0).numpy().astype(np.int)
# img2of1 = video_batch1.permute(0,2,1,3,4)[1][0].permute(1,2,0).numpy().astype(np.int)

# img1of2 = video_batch2.permute(0,2,1,3,4)[0][0].permute(1,2,0).numpy().astype(np.int)
# img2of2 = video_batch2.permute(0,2,1,3,4)[1][0].permute(1,2,0).numpy().astype(np.int)

# img1of3 = video_batch3.permute(0,2,1,3,4)[0][0].permute(1,2,0).numpy().astype(np.int)
# img2of3 = video_batch3.permute(0,2,1,3,4)[1][0].permute(1,2,0).numpy().astype(np.int)

# img1of4 = video_batch4.permute(0,2,1,3,4)[0][0].permute(1,2,0).numpy().astype(np.int)
# img2of4 = video_batch4.permute(0,2,1,3,4)[1][0].permute(1,2,0).numpy().astype(np.int)

# img1of5 = video_batch5.permute(0,2,1,3,4)[0][0].permute(1,2,0).numpy().astype(np.int)
# img2of5 = video_batch5.permute(0,2,1,3,4)[1][0].permute(1,2,0).numpy().astype(np.int)

# img1of6 = video_batch6.permute(0,2,1,3,4)[0][0].permute(1,2,0).numpy().astype(np.int)
# img2of6 = video_batch6.permute(0,2,1,3,4)[1][0].permute(1,2,0).numpy().astype(np.int)


# plt.figure
# plt.imshow(img1of1)

# plt.figure()
# plt.imshow(img2of1)

# plt.figure()
# plt.imshow(img1of2)

# plt.figure()
# plt.imshow(img2of2)

# plt.figure()
# plt.imshow(img1of3)

# plt.figure()
# plt.imshow(img2of3)

# plt.figure()
# plt.imshow(img1of4)

# plt.figure()
# plt.imshow(img2of4)

# plt.figure()
# plt.imshow(img1of5)

# plt.figure()
# plt.imshow(img2of5)

# plt.figure()
# plt.imshow(img1of6)

# plt.figure()
# plt.imshow(img2of6)

"""
CNN Part
"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'VIDEO_LENGTH':25 * 4, # 25프레임 * 4초
    'IMG_SIZE':128,
    'WINDOW_SIZE' : 10
}

# train_data = data_loader.DashcamDataset(
#     video_dir = path_data_prepared,
# )
# train_dataloader = DataLoader(train_data, batch_size=5, shuffle=False)
# iterator = iter(train_dataloader) # next(iterator)할 때마다 배치 하나씩 받을 수 있음

# video_batch1, class_batch1 = next(iterator) # batch 1
# video_batch2, class_batch2 = next(iterator) # batch 1
# video_batch3, class_batch3 = next(iterator) # batch 1
# video_batch4, class_batch4 = next(iterator) # batch 1
# video_batch5, class_batch5 = next(iterator) # batch 1
# video_batch6, class_batch6 = next(iterator) # batch 1
class_batch1_array = cnn.get_labels(class_batch1) 

batch_videos = cnn.separate_to_windows(video_batch1, CFG)
train_batch_videos = cnn.input_C3D(batch_videos)
model = cnn.BaseModel()

model.to(device)
model.train()

train_batch_videos = train_batch_videos.to(device)

output = model(train_batch_videos)
print('batch shape : ', video_batch1.shape)
print('class_labels shape : ',class_batch1_array.shape)
print('train_batch shape : ',train_batch_videos.shape)
print('extracted feature shape : ',output.shape)

"""
Training Part
"""

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

num_epochs = 10
model = training.train_model(model, dataloader_dict, criterion, optimizer, num_epochs, device, CFG)
# num_epochs = 10
# count = 0
# loss_list = []
# iteration_list = []
# accuracy_list = []
# predictions_list = []
# labels_list = []

# for epoch in range(num_epochs):
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         train = Variable(images.view(100, 1, 28))
#         labels = Variable(labels)

#         outputs = model(train)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         count += 1
    
#     if not (count%50):
#         total = 0
#         correct = 0

print("Hello World")