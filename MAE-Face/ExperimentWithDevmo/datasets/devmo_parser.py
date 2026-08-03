from .parser import Parser
from .frame_sample import FrameSample
import os
#Devmo fps=15
class DevmoParser(Parser):
    def __init__(self, root_dir, label_dict, sampler):
        super().__init__(root_dir, label_dict, sampler)

    def parse(self,):    
        samples=[]
        for usr in os.listdir(self.root_dir):
            usr_path = os.path.join(self.root_dir, usr)
            if usr.endswith(".mp4") or usr.endswith(".json"):
                continue
            if os.path.isdir(usr_path):
                usr_name=usr+".mp4"
                if usr_name in self.label_dict:
                    label=self.label_dict.get(usr_name, None)
                    if label is None: continue
                    open_face_dir=os.path.join(usr_path,"openFaces")
                    if not os.path.exists(open_face_dir):continue
                    # 1. Get all frames
                    all_faces=self._get_all_frames(open_face_dir)
                    selected_paths,final_label=self.sampler.sampling_strategy(label, all_faces)
                     # 4. Store final file paths
                    for face_file in selected_paths:
                        sample=FrameSample(os.path.join(open_face_dir, face_file), final_label)
                        samples.append(sample)
        print(f"Total filtered samples: {len(samples)}")
        return samples
        
    