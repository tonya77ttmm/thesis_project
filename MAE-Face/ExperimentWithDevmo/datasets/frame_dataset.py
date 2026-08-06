#DatasetModule
# This module is responsible for providing tensor(frame,label)s of the dataset 
# for the next stge feature extraction
#what data does it own? Dataset path,  how to pick frames, label of each frame
#what decisions does it make?
# a list of (framepath, label)
# It will return [image_tensor,label_tensor]
#    What changes frequently?
# What stays stable?
#FrameDataset owns samples, uses sampler.returns tensors

import os
from .frame_sample import FrameSample
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


#samples->tensor
class FrameDataset(Dataset):
    def __init__(self,samples):
        self.samples = samples
        self.img_size = 224


    def __len__(self):# PyTorch requires both:	__len__ /__getitem__， Otherwise DataLoader won’t work.
        return len(self.samples)

    def __getitem__(self, idx):
        sample=self.samples[idx]
        img_path=sample.path
        label=sample.label
        
        img = cv2.imread(img_path) # hwc
        # Convert BGR -> RGB    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        return img, torch.tensor(label).long(),sample.clip_id

        