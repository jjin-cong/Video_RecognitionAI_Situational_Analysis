# data_loader.py에서 가공한 데이터를 불러와 3D CNN모델에 적용 시킵니다

# Torch 기반 모델링, 특징 추출
# 128 x 128 로

import numpy as np
import cv2 #영상처리를 위해 opencv를 사용한다

import torch
import torch.nn as nn
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

def separate_to_windows(batch_videos, CFG):
    frames_windows = []
    num_of_videos = batch_videos.shape[0]

    for i in range(num_of_videos):
        single_video = batch_videos[i]
        single_video_input = single_video.squeeze()
        frames_windows.append(separate_to_windows10(single_video_input.numpy(), CFG))
    frames_windows = np.array(frames_windows)
    frames_windows = frames_windows.reshape(-1,frames_windows.shape[2],frames_windows.shape[3],frames_windows.shape[4],frames_windows.shape[5])
    return frames_windows


def separate_to_windows10(single_video, CFG):
    windows = []    
    single_window = []
    for i in range(CFG['VIDEO_LENGTH']):
        single_window.append(single_video[:,i,:,:])
        if (i+1) % 10 == 0:
            windows.append(single_window)
            single_window = []
    
    return np.array(windows)

def separate_to_windows_label(labels):
    y_true = []
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            y_true.append(labels[i],[j])
    
    return np.array(y_true)

def get_video(path, CFG):
    frames = []
    cap = cv2.VideoCapture(path) #영상 파일(객체) 가져오기, opencv로 영상을 읽는 방법
    for _ in range(CFG['VIDEO_LENGTH']): #100장을 가져오겠다는 의미 - 미리 정의한 하이퍼파라미터 참고
        _, img = cap.read() #영상 파일에서 프레임 읽기
        img = cv2.resize(img, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
        img = img / 255.
        frames.append(img)
    
    frames_array = np.array(frames).transpose(3,0,1,2)
    frames_window = separate_to_windows10(frames_array)

    return torch.FloatTensor(frames_window)

def get_labels(label_batch):
    labels = []
    for i in range(len(label_batch)):
        labels.append(list(label_batch[i]))
    
    labels_array = np.array(labels).reshape(1,-1)
    
    return labels_array

class BaseModel(nn.Module):
    def __init__(self, num_classes = 3):
        super(BaseModel, self).__init__() #상속된 Module의 __init__ 변수도 사용하겠다
        self.feature_extract = nn.Sequential(
            nn.Conv3d(3,8,(3,3,3)),
            nn.ReLU(),
            nn.BatchNorm3d(8), #batchnorm의 input channel을 변수로 준다
            nn.MaxPool3d(2),#height,width를 (2,2)로 maxpool 해준다
            nn.Conv3d(8,32,(2,2,2)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(30752, 1024),
            nn.ReLU(),
            nn.Linear(1024,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,3)
        )
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        out = np.zeros((x.size()[0], 3))
        # 좀 더  efficient 한 코드 짜기
        for i in range(x.size()[0]):
            y = self.classifier(x[i,:])
            out[i, :] = y.detach().numpy()

        return out

def input_C3D(separated_windows):
    separated_windows = torch.from_numpy(separated_windows)
    separated_windows = separated_windows.permute(0,2,1,3,4) 

    return separated_windows
