import numpy as np
from torch.utils.data import Dataset
import torch
# ==========================================
# read the features and labels from the numpy files using np.memmap 
# return a PyTorch Dataset(feature and label) that can be used to create a DataLoader for training and validation
# ==========================================
class FeatureDataset(Dataset):
    def __init__(self, feat_path, label_path, num_samples, feat_dim=768):
        # --- HIGHLIGHT: Use np.memmap instead of np.load ---
        # We must provide the shape and dtype because raw memmaps don't store headers
        self.features = np.memmap(feat_path, dtype='float32', mode='r', 
                                  shape=(num_samples, feat_dim))
        self.labels = np.memmap(label_path, dtype='int64', mode='r', 
                                shape=(num_samples,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # --- HIGHLIGHT: Added .copy() ---
        # memmap arrays are read-only; .copy() brings it into memory as a standard array
        # so PyTorch can convert it to a tensor without issues.
        feature = torch.from_numpy(self.features[idx].copy()).float()
        label = torch.tensor(self.labels[idx]).long()
        return feature, label
