import torch
import time
import cv2
import os
import models_vit
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from torch.utils.data import Dataset
import numpy as np
import re

class FrameDataset(Dataset):
    def __init__(self, root_dir, label_dict, img_size=224, fps=30, training=True):
        self.samples = []
        self.img_size = img_size
        self.fps = fps
        self.training = training

        for usr in os.listdir(root_dir):
            user_path = os.path.join(root_dir, usr)
            if not os.path.isdir(user_path): continue

            for extract in os.listdir(user_path):
                clip_name = extract + ".avi"
                label = label_dict.get(clip_name, None)
                if label is None: continue

                frame_dir = os.path.join(user_path, extract)
                open_face_dir = os.path.join(frame_dir, "openFaces")
                
                if not os.path.exists(open_face_dir): continue

                # 1. Get all frames
                all_faces = [f for f in os.listdir(open_face_dir) if f.endswith('.bmp')]
                
                # 2. Sort numerically based on that long number
                # This handles filenames like face_1100011002287.bmp
                all_faces.sort(key=lambda f: int(re.search(r'\d+', f).group()))

                # 3. Apply Time Selection Logic
                selected_paths = []
                if self.training:
                    if label == 0:
                    # 5 seconds * 30 fps = index 150
                        target_idx = 5 * self.fps
                        if len(all_faces) > target_idx:
                            selected_paths.append(all_faces[target_idx])
                        elif all_faces:
                            selected_paths.append(all_faces[-1]) # Fallback

                    elif label == 1:
                        # Every 2 seconds * 30 fps = every 60th frame
                        step = 2 * self.fps
                        for i in range(0, len(all_faces), step):
                            selected_paths.append(all_faces[i])
                else:
                    # For validation, we can take all frames or apply a different logic if needed
                    selected_paths = all_faces  # Or apply a different selection strategy   

                # 4. Store final file paths
                for face_file in selected_paths:
                    self.samples.append((os.path.join(open_face_dir, face_file), label))

        print(f"Total filtered samples: {len(self.samples)}")

    def __len__(self):# PyTorch requires both:	__len__ /__getitem__， Otherwise DataLoader won’t work.
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = cv2.imread(img_path) # hwc
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1) # HWC to CHW
        img = torch.from_numpy(img).float() / 255.0 #normalize pixel values from [0, 255] to [0, 1] #what is fro

        return img, torch.tensor(label).long()
# for usr in os.listdir(train_dataset):
#     currUser=os.listdir(os.path.join(train_dataset, usr))
#     for extract in currUser:
#         clip_name=extract+".avi"
#         label=label_dict.get(clip_name, None)
#         if label is None:
#             print(f"Warning: No label found for {clip_name}. Skipping.")
#             continue
#         frames=os.listdir(os.path.join(train_dataset, usr, extract))
#         for frame in frames:
            
#             #exclude the avi file if frame is avi
#             #else read the frame and resize it to 224x224
#             #then convert it to tensor 
#             #then append it to the train_frames list
#             if frame.endswith('.avi'):
#                 continue
#             else:
#                 img_path=os.path.join(train_dataset, usr, extract, frame)
#                 img=cv2.imread(img_path)
#                 img=cv2.resize(img, (img_size, img_size))
#                 img=img.transpose(2, 0, 1) # HWC to CHW
#                 img=torch.from_numpy(img).float() / 255.0 # normalize to [0, 1]
#                 train_frames.append(img) # [3, 224, 224]
#                 train_labels.append(label)
               
# train_frames = torch.stack(train_frames) # [num_frames, 3, 224, 224]
# train_labels = torch.tensor(train_labels).long()
# print(f"Total frames loaded:{train_frames.shape[0]}")
#load the pretrained ViT model and extract features

def extract_features_single_file(model_name, batch_size, device, dataset_path, label_dict, save_subdir):
    """
    Extracts features and saves EACH frame as an individual .pt file.
    """
    # 1. Initialize Model
    model = getattr(models_vit, model_name)(
        global_pool=True,
        num_classes=num_heads,
        drop_path_rate=0.1,
        img_size=224,
    )
    
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

    # 2. Setup Directories
    # Creates e.g., ./Features/Training_features/
    output_dir = os.path.join("./Features", save_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Load Dataset
    dataset = FrameDataset(dataset_path, label_dict, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Starting extraction to {output_dir}...")
    
    global_frame_count = 0
    with torch.no_grad():
        for batch_frames, batch_labels in dataloader:
            batch_frames = batch_frames.to(device)
            # ViT forward pass
            _, features = model(batch_frames, ret_feature=True)
            
            # features is [B, 768], labels is [B]
            features = features.cpu()
            batch_labels = batch_labels.cpu()

            # 4. Save each frame individually
            for i in range(features.size(0)):
                single_feature = features[i]  # Shape: [768]
                single_label = batch_labels[i] # Shape: []
                
                save_file = os.path.join(output_dir, f"frame_{global_frame_count:07d}.pt")
                torch.save({
                    'features': single_feature, 
                    'labels': single_label
                }, save_file)
                
                global_frame_count += 1

            if global_frame_count % 1000 == 0:
                print(f"Processed {global_frame_count} frames...")

    print(f"Finished! Total frames saved: {global_frame_count}")


def extract_features_mmap(model_name, batch_size, device, dataset_path, label_dict, save_prefix):
    """
    Saves features into a single memory-mapped binary file. 
    Extremely space efficient and fast for Weighted Samplers.
    """
    # 1. Initialize Model
    model = getattr(models_vit, model_name)(
        global_pool=True, num_classes=num_heads, drop_path_rate=0.1, img_size=224,
    )
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device).eval()

    # 2. Setup Dataset
    dataset = FrameDataset(dataset_path, label_dict, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    num_samples = len(dataset)
    feat_dim = 768  # For ViT-Base
    os.makedirs("./Features/Numpy_features", exist_ok=True)

    # 3. Create the Memory-Mapped files (The "MMP" solution)
    # This allocates the space immediately without loading into RAM
    feat_file = os.path.join("./Features/Numpy_features", f"{save_prefix}_feats_cc.npy")
    label_file = os.path.join("./Features/Numpy_features", f"{save_prefix}_labels_cc.npy")
    
    features_mmap = np.memmap(feat_file, dtype='float32', mode='w+', shape=(num_samples, feat_dim))
    labels_mmap = np.memmap(label_file, dtype='int64', mode='w+', shape=(num_samples,))

    print(f"Extracting {num_samples} frames to {feat_file}...")

    global_idx = 0
    with torch.no_grad():
        for batch_frames, batch_labels in dataloader:
            batch_frames = batch_frames.to(device)
            _, features = model(batch_frames, ret_feature=True)
            
            # Convert to numpy and write to the specific "slice" of the file
            curr_batch_size = features.shape[0]
            features_mmap[global_idx : global_idx + curr_batch_size] = features.cpu().numpy()
            labels_mmap[global_idx : global_idx + curr_batch_size] = batch_labels.numpy()
            
            global_idx += curr_batch_size
            if global_idx % 1000 == 0 or global_idx == num_samples:
                print(f"Processed {global_idx}/{num_samples} frames...")

    # Important: Ensure data is flushed to disk
    features_mmap.flush()
    labels_mmap.flush()
    print(f"Finished! Files saved: {feat_file} and {label_file}")

def extract_features(model_name, batch_size,device,dataset_path, label_dict,save_path):
    #load the model to get features ???parameters
    model = getattr(models_vit, model_name)(
    global_pool=True,
    num_classes=num_heads,
    drop_path_rate=0.1,
    img_size=224,
    )
    checkpoint = torch.load(ckpt_path, map_location='cpu',weights_only=False)
    checkpoint_model = checkpoint['model']
    msg = model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    model.eval()

    #load dataset
    dataset = FrameDataset(dataset_path, label_dict, img_size)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False)
    # all_features=[]
    # all_labels = []
    features_path=os.makedirs("./Features",exist_ok=True)
    #get the features from the model
    with torch.no_grad():
        #get the features from the model
        for batch_index,(batch_frames, batch_labels) in enumerate(dataloader):
            batch_frames = batch_frames.to(device)
            outputs, features = model(batch_frames, ret_feature=True)
            features_save_path=os.path.join("./Features",save_path)
            feature_batch_path=os.path.join(features_save_path,f"features_batch_{batch_index}.pt")
            torch.save({'features':features.cpu(),'labels':batch_labels},feature_batch_path)
           
    #         all_features.append(features.cpu())#features shape: [batch_size, 768]
    #         all_labels.append(batch_labels)#labels shape:[batch_size]
            
    # features=torch.cat(all_features,dim=0)#features shape: [num_frames, 768]
    # labels = torch.cat(all_labels, dim=0)#labels shape: [num_frames]
    # return TensorDataset(features, labels) #TensorDataset zips them together like:

if __name__=="__main__":
    batch_size=64
    #settings for video, model load path
    ckpt_path="./models/MAE/mae_face_pretrain_vit_base.pth"
    model_name = 'vit_base_patch16'
    num_heads=2
    device='cuda'
    img_size = 224
    #load the training dataset and labels
    train_dataset_path="../confusion_dataset/DAiSEE/DataSet/Train/"
    train_Labels="../confusion_dataset/DAiSEE/Labels/TrainLabels_confusion.csv"
    train_df=pd.read_csv(train_Labels)
    train_label_dict=dict(zip(train_df['ClipID'], train_df['Confusion']))
    # extract_features_single_file(model_name, batch_size, device, train_dataset_path, train_label_dict, "Training_single_frame_features")
    #load the validation dataset and labels
    val_dataset_path="../confusion_dataset/DAiSEE/DataSet/Validation/"
    val_Labels="../confusion_dataset/DAiSEE/Labels/ValidationLabels_confusion.csv"
    val_df=pd.read_csv(val_Labels)
    val_label_dict=dict(zip(val_df['ClipID'], val_df['Confusion']))

    
    # extract_features_single_file(model_name, batch_size, device, val_dataset_path, val_label_dict, "Val_single_frame_features")
    # extract_features(model_name,batch_size,device,train_dataset_path, train_label_dict,"Training_features")
    # extract_features(model_name,batch_size,device,val_dataset_path, val_label_dict,"Val_features")
    extract_features_mmap(model_name, batch_size, device, train_dataset_path, train_label_dict, "Train")
    extract_features_mmap(model_name, batch_size, device, val_dataset_path, val_label_dict, "Val")
    print("Feature extraction completed. You can now train the MLP models using the extracted features saved in the ./Features directory.")