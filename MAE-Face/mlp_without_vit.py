import os
import re
import cv2
import copy
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ==========================================
# 1. DATA SAMPLING STRATEGIES
# ==========================================

def sampling_strategy_v1(label, all_frames, fps=30):
    selected_frames = []
    if label == 0:
        target_idx = 5 * fps
        if len(all_frames) > target_idx:
            selected_frames.append(all_frames[target_idx])
        elif all_frames:
            selected_frames.append(all_frames[-1])
    elif label == 1:
        target_idx = 2 * fps
        for i in range(0, len(all_frames), target_idx):
            selected_frames.append(all_frames[i])
    return selected_frames, label


def sampling_strategy_v2(label, all_frames, fps=30):
    selected_frames = []
    final_label = label
    if label == 0:
        target_idx = 5 * fps
        if len(all_frames) > target_idx:
            selected_frames.append(all_frames[target_idx])
        elif all_frames:
            selected_frames.append(all_frames[-1])
    elif label == 1:
        pass  # Ignored completely
    elif label == 2:
        target_idx = 2 * fps
        for i in range(0, len(all_frames), target_idx):
            selected_frames.append(all_frames[i])
        final_label = 1
    elif label == 3:
        target_idx = 1 * fps
        for i in range(0, len(all_frames), target_idx):
            selected_frames.append(all_frames[i])
        final_label = 1
    return selected_frames, final_label


# ==========================================
# 2. YOUR ORIGINAL FRAMEDATASET (Reusable)
# ==========================================

class FrameDataset(Dataset):
    def __init__(self, root_dir, label_dict, strategy, img_size=224, fps=30, training=True):
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

                all_faces = [f for f in os.listdir(open_face_dir) if f.endswith('.bmp')]
                all_faces.sort(key=lambda f: int(re.search(r'\d+', f).group()))

                if self.training:
                    selected_paths, final_label = strategy(label, all_faces, self.fps)
                else:
                    _, final_label = strategy(label, all_faces, self.fps)
                    selected_paths = all_faces  

                for face_file in selected_paths:
                    self.samples.append((os.path.join(open_face_dir, face_file), final_label))
        print(f"Total filtered samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = torch.from_numpy(img).float() / 255.0
        return img, torch.tensor(label).long()


# ==========================================
# 3. STATIC CLASSIFIER ARCHITECTURE
# ==========================================

class EmotionMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate, num_classes=2):
        super().__init__()
        layers = []
        in_features = input_size

        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = h
        layers.append(nn.Linear(in_features, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ==========================================
# 4. SAMPLER & EVAL ENGINE
# ==========================================

def build_sampler(dataset):
    labels = [sample[1] for sample in dataset.samples]
    counts = np.bincount(labels)
    total = len(labels)
    
    class_weights = [total / (len(counts) * c) if c > 0 else 0 for c in counts]
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # --- HIGHLIGHT: Flatten batch from [B, 3, 224, 224] to [B, 150528] ---
            features = images.view(images.size(0), -1)
            
            outputs = model(features)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    prec = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    rec = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return f1, prec, rec, acc


# ==========================================
# 5. THE RUNTIME PIPELINE
# ==========================================

def train_mlp_direct_images(hidden_layer_variants, dropout_rate, num_classes, learning_rate, weight_decay, num_epochs, patience, device):
    os.makedirs("models/MLP_DirectImages", exist_ok=True)
    img_size = 224
    
    # Calculate total elements in flat image array (3 channels * 224 * 224)
    input_size = 3 * img_size * img_size  # Exactly 150,528 dimensions

    # --- Setup Data Paths ---
    train_dataset_path = "../confusion_dataset/DAiSEE/DataSet/Train/"
    train_labels_v2 = "../confusion_dataset/DAiSEE/Labels/4_TrainLabels_confusion.csv"
    val_dataset_path = "../confusion_dataset/DAiSEE/DataSet/Validation/"
    val_labels = "../confusion_dataset/DAiSEE/Labels/ValidationLabels_confusion.csv"

    train_df = pd.read_csv(train_labels_v2)
    train_label_dict = dict(zip(train_df['ClipID'], train_df['Confusion']))

    val_df = pd.read_csv(val_labels)
    val_label_dict = dict(zip(val_df['ClipID'], val_df['Confusion']))

    # --- Load Dataset Using Reusable Code Base ---
    print("Parsing frame directory layouts...")
    train_dataset = FrameDataset(train_dataset_path, train_label_dict, strategy=sampling_strategy_v2, img_size=img_size, training=True)
    val_dataset = FrameDataset(val_dataset_path, val_label_dict, strategy=sampling_strategy_v2, img_size=img_size, training=False)

    sampler = build_sampler(train_dataset)
       
    # High worker count helps streaming raw images into RAM efficiently
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True)

    # --- Grid Search Executions ---
    for h in hidden_layer_variants:
        print(f"\n🚀 Training Image-MLP Architecture: {h} | Inputs: {input_size}")
        model = EmotionMLP(input_size, h, dropout_rate, num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_f1 = 0
        best_acc = 0
        best_prec = 0
        best_rec = 0
        best_model_state = None

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            train_preds, train_labels_list = [], []

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # --- HIGHLIGHT: Flatten batch from [B, 3, 224, 224] to [B, 150528] ---
                features = images.view(images.size(0), -1)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels_list.extend(labels.cpu().numpy())

            # Evaluate performance metrics across epoch boundary
            avg_loss = running_loss / len(train_loader)
            t_f1 = f1_score(train_labels_list, train_preds, pos_label=1, zero_division=0)
            t_acc = accuracy_score(train_labels_list, train_preds)
            
            f1_conf, prec_conf, rec_conf, acc_conf = evaluate(model, val_loader, device)
            
            print(f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | Train F1: {t_f1:.4f} | Val F1: {f1_conf:.4f} | Val Acc: {acc_conf:.4f}")

            if f1_conf > best_f1:
                best_f1 = f1_conf
                best_acc = acc_conf
                best_prec = prec_conf
                best_rec = rec_conf
                best_model_state = copy.deepcopy(model.state_dict())

        # Save Checkpoint out
        hidden_str = "_".join(map(str, h))
        save_path = f"models/MLP_DirectImages/MLP_{hidden_str}_best.pth"

        torch.save({
            'model_state_dict': best_model_state,
            'best_val_f1': best_f1,
            'best_val_acc': best_acc,
            'best_val_prec': best_prec,
            'best_val_rec': best_rec
        }, save_path)
        print(f" Saved structural model file to: {save_path}")


if __name__ == "__main__":
    hidden_layer_variants = [
        [256],
        [512],
        [1024],
        [512, 256],
        [1024, 512]
    ]
    
    train_mlp_grid_search_params = {
        "hidden_layer_variants": hidden_layer_variants,
        "dropout_rate": 0.5,
        "num_classes": 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "num_epochs": 20, # Reduced epochs since fully flattening large pixel sets can overfit very rapidly
        "patience": 5,
        "device": 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    train_mlp_direct_images(**train_mlp_grid_search_params)