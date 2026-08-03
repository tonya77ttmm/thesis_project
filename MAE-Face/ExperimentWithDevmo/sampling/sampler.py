#strategy has strategy(label, all_faces, self.fps)  


class Sampler:
    def __init__(self,fps):
        self.fps=fps
    def sampling_strategy(self,label, all_frames):
        return all_frames,label
#sampling strategy1: label 0 and 1, if label is 0take 1 frame in the middle of the video, which is around 5 seconds in, which is around frame 150 if fps=30 (5*30fps); if label is 1, take every 2 seconds, which is every 60 frames  if fps=30(2*30fps)


# def sampling_strategy_label_0123(label, all_frames):
#     final_label=label
#     if label==0:
#         target_idx=5*self.fps
#         if len(all_frames)>target_idx:
#             self.selected_frames.append(all_frames[target_idx])
#         elif all_frames:
#             self.selected_frames.append(all_frames[-1]) #fallback to last frame if video is too short
#     elif label == 1:
#         # Ignored completely
#         pass
#     elif label==2:
#         target_idx=2*self.fps
#         for i in range(0, len(all_frames), target_idx):
#             self.selected_frames.append(all_frames[i])
#         final_label=1 #relabel 2 to 1
#     elif label==3:
#         target_idx=1*self.fps
#         for i in range(0, len(all_frames), target_idx):
#             self.selected_frames.append(all_frames[i])
#         final_label=1 #relabel 3 to 1
#     return self.selected_frames,final_label
# def sampling_strategy_label_01(label, all_frames):
#     # selected_frames = []
#     if label==0:
#         target_idx=5*self.fps
#         if len(all_frames)>target_idx:
#             self.selected_frames.append(all_frames[target_idx])
#         elif all_frames:
#             self.selected_frames.append(all_frames[-1]) #fallback to last frame if video is too short
#     elif label==1:
#         target_idx=2*self.fps
#         for i in range(0, len(all_frames), target_idx):
#             self.selected_frames.append(all_frames[i])
#     return self.selected_frames,label

#sampling strategy2: label 0,1,2,3. if label is 0, take 1 frame in the middle of the video, which is around 5 seconds in, which is around frame 150 (5*30fps); if label is 1, ignore; if label is 2, take every 2second, which is every 60 frames (2*30fps); if label is 3, take every second, which is every 30 frames (1*30fps)
