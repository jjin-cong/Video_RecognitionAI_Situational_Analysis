import numpy as np
import cv2 #영상처리를 위해 opencv를 사용한다

import torch
import torch.nn as nn
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'VIDEO_LENGTH':25 * 4, # 25프레임 * 4초
    'IMG_SIZE':128,
    'WINDOW_SIZE' : 10
}


def seperate_to_windows(batch_videos):
    frames_windows = []
    num_of_videos = batch_videos.shape[0]

    for i in range(num_of_videos):
        single_video = batch_videos[i]
        single_video_input = single_video.squeeze()
        frames_windows.append(seperate_to_windows10(single_video_input.numpy()))
    frames_windows = np.array(frames_windows)
    frames_windows = frames_windows.reshape(-1,frames_windows.shape[2],frames_windows.shape[3],frames_windows.shape[4],frames_windows.shape[5])
    return frames_windows


def seperate_to_windows10(single_video):
    windows = []    
    single_window = []
    for i in range(CFG['VIDEO_LENGTH']):
        single_window.append(single_video[:,i,:,:])
        if (i+1) % 10 == 0:
            windows.append(single_window)
            single_window = []
    
    return np.array(windows)

def seperate_to_windows_label(labels):
    y_true = []
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            y_true.append(labels[i],[j])
    
    return np.array(y_true)

def get_video(path):
    frames = []
    cap = cv2.VideoCapture(path) #영상 파일(객체) 가져오기, opencv로 영상을 읽는 방법
    for _ in range(CFG['VIDEO_LENGTH']): #100장을 가져오겠다는 의미 - 미리 정의한 하이퍼파라미터 참고
        _, img = cap.read() #영상 파일에서 프레임 읽기
        img = cv2.resize(img, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
        img = img / 255.
        frames.append(img)
    
    frames_array = np.array(frames).transpose(3,0,1,2)
    frames_window = seperate_to_windows10(frames_array)

    return torch.FloatTensor(frames_window)

def get_labels(label_batch):
    labels = []
    for i in range(len(label_batch)):
        labels.append(list(label_batch[i]))
    
    labels_array = np.array(labels).reshape(1,-1)
    
    return labels_array

class BaseModel_onlyfeature(nn.Module):
    def __init__(self, num_classes = 3):
        super(BaseModel_onlyfeature, self).__init__() #상속된 Module의 __init__ 변수도 사용하겠다
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
        #self.classifier = nn.Linear(1024,num_classes)
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)  
        #x = self.classifier(x)

        return x


def input_C3D(seperated_windows):
    seperated_windows = torch.from_numpy(seperated_windows)
    seperated_windows = seperated_windows.permute(0,2,1,3,4) 

    return seperated_windows




# train_data = DashcamDataset(
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


# class_batch1_array = get_labels(class_batch1) 

# batch_videos = seperate_to_windows(video_batch1)
# train_batch_videos = input_C3D(batch_videos)


# model_onlyfeature = BaseModel_onlyfeature()

# model_onlyfeature.to(device)
# model_onlyfeature.train()

# train_batch_videos = train_batch_videos.to(device)

# output = model_onlyfeature(train_batch_videos)



# print('batch shape : ', video_batch1.shape)
# print('class_labels shape : ',class_batch1_array.shape)
# print('train_batch shape : ',train_batch_videos.shape)
# print('extracted feature shape : ',output.shape)