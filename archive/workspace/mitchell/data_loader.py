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

# path_pardir = os.pardir # ..
# path_data = os.path.join(path_pardir, "data") # ../data
# path_data_prepared = os.path.join(path_data, "videos_prepared")
# sys.path.append(path_pardir)
# sys.path.append(path_data)
# sys.path.append(path_data_prepared)

def make_labels_of_videos(
    labels_list, 
    timestamps_list,
    framenum_per_window=10,
    maxlabelnum_per_window=3,
    framenum_per_video=100, # 데이터 받아서 셀 수 있게 변경?
    second_per_video=4 # 데이터 받아서 셀 수 있게 변경?
    ):
    """
        video별로, {{label_of_window}}를 지정하고 {{label_of_video}}로 변환해주는 함수
        * {{label}} : 각 window마다 해당하는 label (ex : "0" or "1" or "2")
        * {{labels_of_video}} : 각 video 별 window들의 label을 연결된 string으로 만든 것 (ex : "0000001110")
        * {{labels_of_videos}} : {{labels_of_video}}의 리스트 (ex : [ "0000001110", "0000001110", "0000001110" ])
    """
    fps = framenum_per_video/second_per_video
    windownum_per_video = int(framenum_per_video/framenum_per_window)

    labels_of_videos = []
    for label, timestamp in zip(labels_list, timestamps_list):
        timestamp = float(timestamp[:2])+(float(timestamp[2:])*1e-2)
        labels_of_video = list("0"*10)

        # 한 video에서 각 window의 시작 시간
        starttimes_of_windows = np.zeros((windownum_per_video)) 
        time_interval = framenum_per_window/fps
        for windowidx in range(len(starttimes_of_windows)):
            starttimes_of_windows[windowidx] =  windowidx * time_interval
    #     print(f"\n\n|-> starttimes_of_windows : {starttimes_of_windows}")

        # 사고 시작 시점(timestamp)에 해당되는 window의 index 찾기 : {{timestamp_idx}}
        timestamp_loc = timestamp < starttimes_of_windows
        if True not in timestamp_loc: # 마지막 window만 해당하는 경우를 위한 예외 처리 (모두 False로 나오기 때문)
            timestamp_idx = len(timestamp_loc)-1
            timestamp_loc[timestamp_idx] = True
        else: # 마지막 window만 해당하는 경우가 아닌 경우
            timestamp_idx = np.where(timestamp_loc == True)[0][0]-1 
            timestamp_loc[timestamp_idx] = True
    #     print(f"|-> timestamp_loc : {timestamp_loc}")
    #     print(f"|-> timestamp_idx : {timestamp_idx}")

        # window 당 최대 유효 label 수에 대한 처리 : {{maxlabelnum_per_window}}
        if collections.Counter(timestamp_loc)[True] > 3:
            timestamp_loc[timestamp_idx+3:] = False
    #         print(f"|-> timestamp_loc : {timestamp_loc} (after maxlabelnum)")

        # {{labels_of_video}} 변환 
        idx_valid = np.where(timestamp_loc == True)[0]
        labels_of_video[idx_valid[0]:idx_valid[-1]+1] = label*len(idx_valid)
    #     print(f"|-> idx_valid : {idx_valid}")
    #     print(f"|-> labels_of_video : {labels_of_video}")
        labels_of_videos.append("".join(labels_of_video))

#     print("\n= labels_of_videos ===========================================================")
#     print(labels_of_videos, "\n")
#     for result in labels_of_videos:
#         print(result)
    
    return labels_of_videos


class DashcamDataset(Dataset):
    def __init__(self, video_dir, video_res=128, transform=None, target_transform=None, train=True):
        if train == True:
            self.video_dir = os.path.join(video_dir, "train") # videos_prepared/train
        else:
            self.video_dir = os.path.join(video_dir, "test") # videos_prepared/test
        self.video_fnames = sorted(os.listdir(self.video_dir))
        for fname in self.video_fnames:
            if fname[-3:]!="mp4": self.video_fnames.remove(fname)
        
        self.video_res = video_res # default 128
        self.video_labels = [fname[-5] for fname in self.video_fnames]
        self.video_tstamps = [fname[-10:-6] for fname in self.video_fnames]
        self.video_tlabels = make_labels_of_videos(self.video_labels, self.video_tstamps)
        
        self.transform = transform # ?
        self.target_transform = target_transform # ?
        
    def __len__(self): # 데이터셋의 샘플 개수 반환
        return len(self.video_labels)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_fnames[idx])
        video_origin = torchvision.io.read_video(video_path, output_format="TCHW")[0]
        
        # video resolution resizing & permuting       
        video_resized = torch.zeros((video_origin.shape[0],video_origin.shape[1],self.video_res,self.video_res))
#         print("video_resized shape :", video_resized.shape)
        resize_transform = transforms.Resize(size=(self.video_res,self.video_res))
        for iii in range(len(video_origin)):
#             frame_resized = video_origin[iii]
#             frame_resized2 = resize_transform(frame_resized)
#             print(frame_resized2.shape)
#             plt.imshow(frame_resized2.permute(1,2,0))
#             video_resized[iii] = frame_resized2
            video_resized[iii] = resize_transform(video_origin[iii])
        video = video_resized.permute(1,0,2,3) # 차원 순서 변경 : TCHW -> CTHW
        
        label = self.video_tlabels[idx]
        
        if self.transform: video = self.transform(video)
        if self.target_transform: label = self.target_transform(self.video_labels)
            
        return video, label