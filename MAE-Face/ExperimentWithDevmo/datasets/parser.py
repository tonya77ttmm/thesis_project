from abc import ABC, abstractmethod
import os,re
class Parser(ABC):
    def __init__(self, root_dir, label_dict, sampler):
        self.root_dir=root_dir
        self.label_dict=label_dict
        self.sampler=sampler
    @abstractmethod
    def parse(self,):
        pass
    
    def _get_all_frames(self,f_dir):
        #1.get all frames
        all_faces = [f for f in os.listdir(f_dir) if f.endswith('.bmp')]
        # 2. Sort numerically based on that long number
                # This handles filenames like face_1100011002287.bmp
        all_faces.sort(key=lambda f: int(re.search(r'\d+', f).group()))
        
        return all_faces
        