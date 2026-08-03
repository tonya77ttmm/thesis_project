from .parser import Parser
from .frame_sample import FrameSample
#DaiSee fps=30 
#folder->parse folders->samples
class DaiseeParser(Parser):
    def __init__(self, root_dir, label_dict, sampler):
        super().__init__(root_dir, label_dict, sampler)
    
    def parse(self,):
        samples=[]
        for usr in os.listdir(self.root_dir):
            user_path = os.path.join(self.root_dir, usr)
            if not os.listdir(user_path): continue
            for extract in os.listdir(user_path):
                clip_name = extract + ".avi"
                label = self.label_dict.get(clip_name, None)
                if label is None: continue
                frame_dir = os.path.join(user_path, extract)
                open_face_dir = os.path.join(frame_dir, "openFaces")
                if not os.path.exists(open_face_dir): continue
                all_faces=self._get_all_frames(open_face_dir)
                selected_paths,final_label=self.sampler.sampling_strategy(label, all_faces)
                # 4. Store final file paths
                for face_file in selected_paths:
                    sample=FrameSample(os.path.join(open_face_dir, face_file), final_label)
                    samples.append(sample)                
        # print(f"Total filtered samples: {len(self.samples)}")  
        return samples
              
    
                # # 1. Get all frames
                # all_faces = [f for f in os.listdir(open_face_dir) if f.endswith('.bmp')]
                # # 2. Sort numerically based on that long number
                # # This handles filenames like face_1100011002287.bmp
                # all_faces.sort(key=lambda f: int(re.search(r'\d+', f).group()))
                # # 3. Apply Time Selection Logic
                # #seperate to 2 strategies

                # selected_paths,final_label=strategy(label, all_faces,self.fps)
                # # 4. Store final file paths
                # for face_file in selected_paths:
                #     self.samples.append((os.path.join(open_face_dir, face_file), final_label))
                       
                # selected_paths = []
                # if self.training:
                #     selected_paths, final_label = strategy(label, all_faces, self.fps)  # Use the provided strategy function
                # else:
                #     # For validation, we can take all frames or apply a different logic if needed
                #     # selected_paths, final_label = sampling_strategy_v1(label, all_faces, self.fps)  # Use the same strategy for consistency
                #     selected_paths = all_faces  # Or apply a different selection strategy 
                #     final_label = label  # Keep original label for validation  

                