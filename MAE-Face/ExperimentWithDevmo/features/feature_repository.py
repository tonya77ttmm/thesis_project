import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
class FeatureRepository:
    def __init__(self,batch_size):
        self.batch_size=batch_size
        
    def store(self,dataset,feature_dir,save_prefix,extractor):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        num_samples = len(dataset)
        # feature_dir="./Features/Numpy_features"
        os.makedirs(feature_dir, exist_ok=True)
        feat_file = os.path.join(feature_dir, f"{save_prefix}_feats.npy")
        label_file = os.path.join(feature_dir, f"{save_prefix}_labels.npy")
        clip_file = os.path.join(feature_dir,f"{save_prefix}_clip_ids.npy"
)
        features_mmap = np.memmap(feat_file, dtype='float32', mode='w+', shape=(num_samples,extractor.feature_dim))
        labels_mmap = np.memmap(label_file, dtype='int64', mode='w+', shape=(num_samples,))
        print(f"Extracting {num_samples} frames to {feat_file}...")
        global_idx = 0
        clip_ids = np.empty(num_samples,dtype='<U100')
        
        for batch_frames, batch_labels, batch_clip_ids in dataloader:
            features = extractor.extract(batch_frames)
            # Convert to numpy and write to the specific "slice" of the file
            curr_batch_size = features.shape[0]
            features_mmap[global_idx : global_idx +curr_batch_size] = features.cpu().numpy()
            labels_mmap[global_idx : global_idx + curr_batch_size] = batch_labels.numpy()
            clip_ids[global_idx:global_idx+curr_batch_size] = np.array(batch_clip_ids)        
            global_idx += curr_batch_size
            if global_idx % 1000 == 0 or global_idx == num_samples:
                print(f"Processed {global_idx}/{num_samples} frames...")
        
            # Important: Ensure data is flushed to disk
        features_mmap.flush()
        labels_mmap.flush()
        np.save(clip_file, clip_ids)
        print(f"Finished! Files saved: {feat_file} and {label_file} and {clip_file}")



