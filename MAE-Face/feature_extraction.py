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

#sampling strategy1: label 0 and 1, if label is 0take 1 frame in the middle of the video, which is around 5 seconds in, which is around frame 150 (5*30fps); if label is 1, take every 2 seconds, which is every 60 frames (2*30fps)
def sampling_strategy_v1(label, all_frames, fps=30):
    selected_frames = []
    if label==0:
        target_idx=5*fps
        if len(all_frames)>target_idx:
            selected_frames.append(all_frames[target_idx])
        elif all_frames:
            selected_frames.append(all_frames[-1]) #fallback to last frame if video is too short
    elif label==1:
        target_idx=2*fps
        for i in range(0, len(all_frames), target_idx):
            selected_frames.append(all_frames[i])
    return selected_frames,label

#sampling strategy2: label 0,1,2,3. if label is 0, take 1 frame in the middle of the video, which is around 5 seconds in, which is around frame 150 (5*30fps); if label is 1, ignore; if label is 2, take every 2second, which is every 60 frames (2*30fps); if label is 3, take every second, which is every 30 frames (1*30fps)
def sampling_strategy_v2(label, all_frames, fps=30):
    selected_frames = []
    final_label=label
    if label==0:
        target_idx=5*fps
        if len(all_frames)>target_idx:
            selected_frames.append(all_frames[target_idx])
        elif all_frames:
            selected_frames.append(all_frames[-1]) #fallback to last frame if video is too short
    elif label == 1:
        # Ignored completely
        pass
    elif label==2:
        target_idx=2*fps
        for i in range(0, len(all_frames), target_idx):
            selected_frames.append(all_frames[i])
        final_label=1 #relabel 2 to 1
    elif label==3:
        target_idx=1*fps
        for i in range(0, len(all_frames), target_idx):
            selected_frames.append(all_frames[i])
        final_label=1 #relabel 3 to 1
    return selected_frames,final_label

class FrameDataset(Dataset):
    def __init__(self, root_dir, label_dict, strategy,img_size=224, fps=30, training=True, ):
        self.samples = []
        self.img_size = img_size
        self.fps = fps
        self.training = training

        for usr in os.listdir(root_dir):
            user_path = os.path.join(root_dir, usr)
            if not os.listdir(user_path): continue

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
                # selected_paths = []
                if self.training:
                    selected_paths, final_label = strategy(label, all_faces, self.fps)  # Use the provided strategy function
                else:
                    # For validation, we can take all frames or apply a different logic if needed
                    # selected_paths, final_label = sampling_strategy_v1(label, all_faces, self.fps)  # Use the same strategy for consistency
                    selected_paths = all_faces  # Or apply a different selection strategy 
                    final_label = label  # Keep original label for validation  

                # 4. Store final file paths
                for face_file in selected_paths:
                    self.samples.append((os.path.join(open_face_dir, face_file), final_label))
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
class FrameDatasetDevmo(FrameDataset):
    def __init__(self, root_dir, label_dict, strategy,img_size=224, fps=15, training=True, ):
        self.samples = []
        self.img_size = img_size
        self.fps = fps
        self.training = training
        for usr in os.listdir(root_dir):
            usr_path = os.path.join(root_dir, usr)
            if usr.endswith(".mp4") or usr.endswith(".json"):
                continue
            if os.path.isdir(usr_path):
                usr_name=usr+".mp4"
                if usr_name in label_dict:
                    label=label_dict.get(usr_name, None)
                    if label is None: continue
                    open_face_dir=os.path.join(usr_path,"openFaces")
                    if not os.path.exists(open_face_dir): continue
                        # 1. Get all frames
                    for f in os.listdir(open_face_dir):
                        if f.endswith('.bmp'):
                            self.samples.append((os.path.join(open_face_dir, f), label))
                # # 1. Get all frames
                # all_faces = [f for f in os.listdir(open_face_dir) if f.endswith('.bmp')]
                
                # # 2. Sort numerically based on that long number
                # # This handles filenames like face_1100011002287.bmp
                # all_faces.sort(key=lambda f: int(re.search(r'\d+', f).group()))


                # 3. Apply Time Selection Logic
                # selected_paths = []
                # if self.training:
                #     selected_paths, final_label = strategy(label, all_faces, self.fps)  # Use the provided strategy function
                # else:
                #     # For validation, we can take all frames or apply a different logic if needed
                #     # selected_paths, final_label = sampling_strategy_v1(label, all_faces, self.fps)  # Use the same strategy for consistency
                #     selected_paths = all_faces  # Or apply a different selection strategy 
                #     final_label = label  # Keep original label for validation  

                # 4. Store final file paths
                # for face_file in selected_paths:
                #     self.samples.append((os.path.join(open_face_dir, face_file), final_label))
        print(f"Total filtered samples: {len(self.samples)}")

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


def extract_features_mmap(model_name, batch_size, device, dataset_path, label_dict, save_prefix,strategy,training,dataset):
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
    # dataset = FrameDataset(dataset_path, label_dict, strategy=strategy,img_size=img_size,training=training)
    dataset=dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    num_samples = len(dataset)
    feat_dim = 768  # For ViT-Base
    os.makedirs("./Features/Numpy_features", exist_ok=True)

    # 3. Create the Memory-Mapped files (The "MMP" solution)
    # This allocates the space immediately without loading into RAM
    feat_file = os.path.join("./Features/Numpy_features", f"{save_prefix}_feats.npy")
    label_file = os.path.join("./Features/Numpy_features", f"{save_prefix}_labels.npy")
    
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
    train_Labels_v1="../confusion_dataset/DAiSEE/Labels/TrainLabels_confusion.csv"
    train_df_v1=pd.read_csv(train_Labels_v1)
    train_label_dict_v1=dict(zip(train_df_v1['ClipID'], train_df_v1['Confusion']))

    train_labels_v2="../confusion_dataset/DAiSEE/Labels/4_TrainLabels_confusion.csv"
    train_df_v2=pd.read_csv(train_labels_v2)
    train_label_dict_v2=dict(zip(train_df_v2['ClipID'], train_df_v2['Confusion']))

    # extract_features_single_file(model_name, batch_size, device, train_dataset_path, train_label_dict, "Training_single_frame_features")
    #load the validation dataset and labels
    val_dataset_path="../confusion_dataset/DAiSEE/DataSet/Validation/"
    val_Labels="../confusion_dataset/DAiSEE/Labels/ValidationLabels_confusion.csv"
    val_df=pd.read_csv(val_Labels)
    val_label_dict=dict(zip(val_df['ClipID'], val_df['Confusion']))

    devmo_dataset_path="../confusion_dataset/Devmo/devemo+/"
    devmo_train_df=pd.read_csv("../confusion_dataset/Devmo/devemo+/train.csv")
    devmo_train_label_dict=dict(zip(devmo_train_df['clipID'], devmo_train_df['label']))
    devmo_test_df=pd.read_csv("../confusion_dataset/Devmo/devemo+/test.csv")
    devmo_test_label_dict=dict(zip(devmo_test_df['clipID'], devmo_test_df['label']))

    devmo_fold_0_df=pd.read_csv("../confusion_dataset/Devmo/devemo+/fold_0_train.csv")
    devmo_fold_0_label_dict=dict(zip(devmo_fold_0_df['clipID'], devmo_fold_0_df['label']))
    devmo_fold_1_df=pd.read_csv("../confusion_dataset/Devmo/devemo+/fold_1_train.csv")
    devmo_fold_1_label_dict=dict(zip(devmo_fold_1_df['clipID'], devmo_fold_1_df['label']))
    devmo_fold_2_df=pd.read_csv("../confusion_dataset/Devmo/devemo+/fold_2_train.csv")
    devmo_fold_2_label_dict=dict(zip(devmo_fold_2_df['clipID'], devmo_fold_2_df['label']))
    devmo_fold_3_df=pd.read_csv("../confusion_dataset/Devmo/devemo+/fold_3_train.csv")
    devmo_fold_3_label_dict=dict(zip(devmo_fold_3_df['clipID'], devmo_fold_3_df['label']))
    devmo_fold_4_df=pd.read_csv("../confusion_dataset/Devmo/devemo+/fold_4_train.csv")
    devmo_fold_4_label_dict=dict(zip(devmo_fold_4_df['clipID'], devmo_fold_4_df['label']))

    devmo_fold_0_val_df=pd.read_csv("../confusion_dataset/Devmo/devemo+/fold_0_val.csv")
    devmo_fold_0_val_label_dict=dict(zip(devmo_fold_0_val_df['clipID'], devmo_fold_0_val_df['label']))
    devmo_fold_1_val_df=pd.read_csv("../confusion_dataset/Devmo/devemo+/fold_1_val.csv")
    devmo_fold_1_val_label_dict=dict(zip(devmo_fold_1_val_df['clipID'], devmo_fold_1_val_df['label']))
    devmo_fold_2_val_df=pd.read_csv("../confusion_dataset/Devmo/devemo+/fold_2_val.csv")
    devmo_fold_2_val_label_dict=dict(zip(devmo_fold_2_val_df['clipID'], devmo_fold_2_val_df['label']))
    devmo_fold_3_val_df=pd.read_csv("../confusion_dataset/Devmo/devemo+/fold_3_val.csv")
    devmo_fold_3_val_label_dict=dict(zip(devmo_fold_3_val_df['clipID'], devmo_fold_3_val_df['label']))
    devmo_fold_4_val_df=pd.read_csv("../confusion_dataset/Devmo/devemo+/fold_4_val.csv")
    devmo_fold_4_val_label_dict=dict(zip(devmo_fold_4_val_df['clipID'], devmo_fold_4_val_df['label']))
    # extract_features_single_file(model_name, batch_size, device, val_dataset_path, val_label_dict, "Val_single_frame_features")
    # extract_features(model_name,batch_size,device,train_dataset_path, train_label_dict,"Training_features")
    # extract_features(model_name,batch_size,device,val_dataset_path, val_label_dict,"Val_features")
    # extract_features_mmap(model_name, batch_size, device, train_dataset_path, train_label_dict_v1, "Train_v1",strategy=sampling_strategy_v1) 
    # extract_features_mmap(model_name, batch_size, device, train_dataset_path,train_label_dict_v2, "Train_v2",strategy=sampling_strategy_v2)
    # extract_features_mmap(model_name, batch_size, device, val_dataset_path, val_label_dict, "Val-all",strategy=None,training=False)  
    # devmoTrainDataset=FrameDatasetDevmo(devmo_dataset_path, devmo_train_label_dict, strategy=None,img_size=img_size,training=False)
    # devmoTestDataset=FrameDatasetDevmo(devmo_dataset_path, 
    # devmo_test_label_dict, strategy=None,img_size=img_size,training=False)

    #devmo fold train datasets
    # devmoFold0Dataset=FrameDatasetDevmo(devmo_dataset_path, devmo_fold_0_label_dict, strategy=None,img_size=img_size,training=False)
    # devmoFold1Dataset=FrameDatasetDevmo(devmo_dataset_path, devmo_fold_1_label_dict, strategy=None,img_size=img_size,training=False)
    # devmoFold2Dataset=FrameDatasetDevmo(devmo_dataset_path, devmo_fold_2_label_dict, strategy=None,img_size=img_size,training=False)
    # devmoFold3Dataset=FrameDatasetDevmo(devmo_dataset_path, devmo_fold_3_label_dict, strategy=None,img_size=img_size,training=False)
    # devmoFold4Dataset=FrameDatasetDevmo(devmo_dataset_path, devmo_fold_4_label_dict, strategy=None,img_size=img_size,training=False)

    #devmo fold val datasets
    devmoFold0ValDataset=FrameDatasetDevmo(devmo_dataset_path, devmo_fold_0_val_label_dict, strategy=None,img_size=img_size,training=False)
    devmoFold1ValDataset=FrameDatasetDevmo(devmo_dataset_path, devmo_fold_1_val_label_dict, strategy=None,img_size=img_size,training=False)
    devmoFold2ValDataset=FrameDatasetDevmo(devmo_dataset_path, devmo_fold_2_val_label_dict, strategy=None,img_size=img_size,training=False)
    devmoFold3ValDataset=FrameDatasetDevmo(devmo_dataset_path, devmo_fold_3_val_label_dict, strategy=None,img_size=img_size,training=False)
    devmoFold4ValDataset=FrameDatasetDevmo(devmo_dataset_path, devmo_fold_4_val_label_dict, strategy=None,img_size=img_size,training=False)


    # extract_features_mmap(model_name, batch_size, device, devmo_dataset_path, devmo_train_label_dict, "Devmo-train",strategy=None,training=False,dataset=devmoTrainDataset)
    # extract_features_mmap(model_name, batch_size, device, devmo_dataset_path, devmo_test_label_dict, "Devmo-test",strategy=None,training=False,dataset=devmoTestDataset)
    # extract_features_mmap(model_name, batch_size, device, devmo_dataset_path, devmo_fold_0_label_dict, "Devmo-fold0",strategy=None,training=False,dataset=devmoFold0Dataset)
    # extract_features_mmap(model_name, batch_size, device, devmo_dataset_path, devmo_fold_1_label_dict, "Devmo-fold1",strategy=None,training=False,dataset=devmoFold1Dataset)
    # extract_features_mmap(model_name, batch_size, device, devmo_dataset_path, devmo_fold_2_label_dict, "Devmo-fold2",strategy=None,training=False,dataset=devmoFold2Dataset)
    # extract_features_mmap(model_name, batch_size, device, devmo_dataset_path, devmo_fold_3_label_dict, "Devmo-fold3",strategy=None,training=False,dataset=devmoFold3Dataset)
    # extract_features_mmap(model_name, batch_size, device, devmo_dataset_path, devmo_fold_4_label_dict, "Devmo-fold4",strategy=None,training=False,dataset=devmoFold4Dataset)
    extract_features_mmap(model_name, batch_size, device, devmo_dataset_path, devmo_fold_0_val_label_dict, "Devmo-fold0-val",strategy=None,training=False,dataset=devmoFold0ValDataset)
    extract_features_mmap(model_name, batch_size, device, devmo_dataset_path, devmo_fold_1_val_label_dict, "Devmo-fold1-val",strategy=None,training=False,dataset=devmoFold1ValDataset)
    extract_features_mmap(model_name, batch_size, device, devmo_dataset_path, devmo_fold_2_val_label_dict, "Devmo-fold2-val",strategy=None,training=False,dataset=devmoFold2ValDataset)
    extract_features_mmap(model_name, batch_size, device, devmo_dataset_path, devmo_fold_3_val_label_dict, "Devmo-fold3-val",strategy=None,training=False,dataset=devmoFold3ValDataset)
    extract_features_mmap(model_name, batch_size, device, devmo_dataset_path, devmo_fold_4_val_label_dict, "Devmo-fold4-val",strategy=None,training=False,dataset=devmoFold4ValDataset)

    print("Feature extraction completed. You can now train the MLP models using the extracted features saved in the ./Features directory.")