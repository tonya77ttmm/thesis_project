from .sampler import Sampler
import numpy as np
class LabelO1Sampler(Sampler):
    def __init__(self,fps):
        super().__init__(fps)
    def sampling_strategy(self,label,all_frames):
        selected_frames=[]
        #每个视频随机取样30
        if label==0:
            n = min(30, len(all_frames))
            indices = np.random.choice(len(all_frames),size=n,replace=False)
            selected_frames = [all_frames[i] for i in indices]
        elif label==1: 
            # 45frames
            n = min(60, len(all_frames))
            indices = np.random.choice(len(all_frames),size=n,replace=False)
            selected_frames = [all_frames[i] for i in indices]

        return selected_frames,label