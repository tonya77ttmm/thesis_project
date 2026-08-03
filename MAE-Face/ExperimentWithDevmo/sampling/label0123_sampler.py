from .Sampler import Sampler
class LabelO123Sampler(Sampler):
    def __init__(self,fps):
        super().__init__(fps)
    def sampling_strategy(self,label,all_frames):
        selected_frames=[]
        final_label=label
        if label==0:
            target_idx=5*self.fps
            if len(all_frames)>target_idx:
                selected_frames.append(all_frames[target_idx])
            elif all_frames:
                selected_frames.append(all_frames[-1]) #fallback to last frame if video is too short
        elif label == 1:
        # Ignored completely
            pass
        elif label==2:
            target_idx=2*self.fps
            for i in range(0, len(all_frames), target_idx):
                selected_frames.append(all_frames[i])
            final_label=1 #relabel 2 to 1
        elif label==3:
            target_idx=1*self.fps
            for i in range(0, len(all_frames), target_idx):
                selected_frames.append(all_frames[i])
            final_label=1 #relabel 3 to 1
        return selected_frames,final_label