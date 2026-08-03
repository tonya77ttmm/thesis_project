from .Sampler import Sampler
class LabelO1Sampler(Sampler):
    def __init__(self,fps):
        super().__init__(fps)
    def sampling_strategy(self,label,all_frames):
        selected_frames=[]
        if label==0:
            target_idx=5*self.fps
            if len(all_frames)>target_idx:
                selected_frames.append(all_frames[target_idx])
            elif all_frames:
                selected_frames.append(all_frames[-1]) #fallback to last frame if video is too short
        elif label==1:
            target_idx=2*self.fps
            for i in range(0, len(all_frames), target_idx):
                selected_frames.append(all_frames[i])
        return selected_frames,label