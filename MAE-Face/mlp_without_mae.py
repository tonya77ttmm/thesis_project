#import lib
#process video, cal time, load model
from collections import Counter
import torch
import time
import cv2
import os
import models_vit
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from feature_extraction import FrameDataset


# # ==========================================
# # CHANGE 1: New FeatureDataset for .npy MMAP
# # ==========================================
# class FeatureDataset(Dataset):
#     def __init__(self, feat_path, label_path, num_samples, feat_dim=768):
#         # --- HIGHLIGHT: Use np.memmap instead of np.load ---
#         # We must provide the shape and dtype because raw memmaps don't store headers
#         self.features = np.memmap(feat_path, dtype='float32', mode='r', 
#                                   shape=(num_samples, feat_dim))
#         self.labels = np.memmap(label_path, dtype='int64', mode='r', 
#                                 shape=(num_samples,))

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # --- HIGHLIGHT: Added .copy() ---
#         # memmap arrays are read-only; .copy() brings it into memory as a standard array
#         # so PyTorch can convert it to a tensor without issues.
#         feature = torch.from_numpy(self.features[idx].copy()).float()
#         label = torch.tensor(self.labels[idx]).long()
#         return feature, label

# train an MLP classifier with the features and labels
#define the MLP classifier
class EmotionMLP(nn.Module):
    def __init__(self, input_size,hidden_layers, dropout_rate, num_classes=2):
        super().__init__()
        layers=[]
        in_features=input_size

        for h in hidden_layers:
            layers.append(nn.Linear(in_features,h))
            layers.append(nn.BatchNorm1d(h)) # Added Batch Norm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features=h
        layers.append(nn.Linear(in_features,num_classes))
        self.model=nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

# class FocalLoss(nn.Module):
#     def __init__(self, alpha, gamma=2):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.ce = nn.CrossEntropyLoss(reduction='none')

#     def forward(self, logits, targets):
#         ce_loss = self.ce(logits, targets)
#         pt = torch.exp(-ce_loss)
#         loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss
#         return loss.mean()

# ==========================================
# CHANGE 2: Efficient Weight Computation
# ==========================================
def compute_class_weights(dataset, device):
    # Since labels are in a flat numpy array, we can use np.unique for speed
    labels = dataset.labels 
    counts = np.bincount(labels)
    total = len(labels)
    
    weights = [total / (len(counts) * c) if c > 0 else 0 for c in counts]
    print(f"Class counts: {counts}, Weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32).to(device)

def build_sampler(dataset, class_weights):
    # Map each sample's label to its weight
    # dataset.labels is the mmap array
    sample_weights = [class_weights[l].item() for l in dataset.labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
# def collate_fn(batch):
#     features_list, labels_list = zip(*batch)  # unzip batch

#     # Pad features with 0
#     features_padded = pad_sequence(features_list, batch_first=True)  # (batch, max_seq_len, feat_dim)

#     # Pad labels with -100 (special value to ignore in loss)
#     labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)  # (batch, max_seq_len)

#     return features_padded, labels_padded

def evaluate(model, loader, device, threshold=0.3):
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device) # Shape: [batch_size, 768]
            labels = labels.to(device)     # Shape: [batch_size]

            outputs = model(features)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics for the 'Confusion' class (pos_label=1)
    prec = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    rec = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return f1, prec, rec, acc

#grid search for hyperparameters
def train_mlp_grid_search(input_size,hidden_layers,dropout_rate,num_classes,learning_rate=1e-3, weight_decay=1e-5,num_epochs=200,patience=5,device='cuda'):
    # os.makedirs("saved_models",exist_ok=True)#exist_ok=True → don’t throw an error if the folder already exists.
    os.makedirs("models/MLP", exist_ok=True)
    # ===== DATASETS =====
    # ==========================================
    # CHANGE 3: Point to .npy files instead of dirs
    # ==========================================
    train_feat = "./Features/Numpy_features/Train_feats_cc.npy"
    train_lab  = "./Features/Numpy_features/Train_labels_cc.npy"
    val_feat   = "./Features/Numpy_features/Val_feats_cc.npy"
    val_lab    = "./Features/Numpy_features/Val_labels_cc.npy"

    train_count = os.path.getsize(train_feat) // (768 * 4)
    val_count = os.path.getsize(val_feat) // (768 * 4)

    train_dataset = FeatureDataset(train_feat, train_lab, num_samples=train_count)
    val_dataset = FeatureDataset(val_feat, val_lab, num_samples=val_count)


    # test_dataset = FeatureDataset("./Features/Test_features")

    # ===== IMBALANCE HANDLING =====
    class_weights = compute_class_weights(train_dataset, device)
    sampler = build_sampler(train_dataset, class_weights)
       
    # num_workers=4 helps fetch data from disk while GPU is training
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)


    # train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=64)
    # test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
    # ===== LOOP OVER ARCHITECTURES =====
    for h in hidden_layers:
        print(f"Training MLP with hidden layers: {h}, dropout rate: {dropout_rate}, learning rate: {learning_rate}")
        #create MLP model
        model=EmotionMLP(input_size,h,dropout_rate,num_classes).to(device)
        optimizer=optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #BUG maybe try weighted loss which are proportional to the class number in the training dataset
        # Pass the class weights you already calculated to the criterion
        criterion = nn.CrossEntropyLoss()
        # criterion=nn.CrossEntropyLoss(ignore_index=-100)
        #scheduler=ReduceLROnPlateau(optimizer,mode="max",factor=0.1,patience=3) #what is verbose=True?
        # criterion = FocalLoss(alpha=class_weights)
        # dataset=FeatureDataset(feature_dir="./features/Training_features")
        # loader=DataLoader(dataset,batch_size=4,shuffle=True, collate_fn=collate_fn) #load 4 files at a time, each file has 64 features, each bach has 256 features
        best_f1 = 0
        best_acc=0
        best_prec=0
        best_rec=0
        epochs_no_improve=0
        best_model_state=None
        for epoch in range(num_epochs):
            model.train()
            running_loss=0.0
            train_preds, train_labels = [], []
            for features, labels in train_loader:
                # features=features.view(-1,features.size(-1))
                # labels=labels.view(-1)
                # mask = labels != -100
                # features = features[mask].to(device)
                # labels = labels[mask].to(device)
                features = features.to(device) # Shape: [batch_size, 768]
                labels = labels.to(device)     # Shape: [batch_size]
                optimizer.zero_grad()
                outputs=model(features)
                loss=criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss+=loss.item()
                preds = outputs.argmax(dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                #print(f"Batch Loss: {loss.item():.4f}")
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
            # --- Epoch Sanity Check ---
            t_f1 = f1_score(train_labels, train_preds, pos_label=1, zero_division=0)
            t_acc = accuracy_score(train_labels, train_preds)
            t_prec= precision_score(train_labels, train_preds, pos_label=1, zero_division=0)
            t_rec= recall_score(train_labels, train_preds, pos_label=1, zero_division=0)
            print(f"TRAIN → Conf F1: {t_f1:.4f}, Conf Precision: {t_prec:.4f}, Conf Recall: {t_rec:.4f}, Accuracy: {t_acc:.4f}")
            
            # --- Validation Section ---
            # ===== VALIDATION =====
            f1_conf, prec_conf, rec_conf, acc_conf = evaluate(model, val_loader, device)
            print(f"VAL → Conf F1: {f1_conf:.4f}, Conf Precision: {prec_conf:.4f}, Conf Recall: {rec_conf:.4f}, Accuracy: {acc_conf:.4f}")

            # scheduler.step(f1_conf)

            # ===== EARLY STOPPING (based on F1!) =====
            if f1_conf > best_f1:
                best_f1 = f1_conf
                best_acc = acc_conf
                best_prec = prec_conf
                best_rec = rec_conf
                best_model_state = copy.deepcopy(model.state_dict())
                #epochs_no_improve = 0
            # else:
            #     epochs_no_improve += 1

            # if epochs_no_improve >= patience:
            #     print("Early stopping")
            #     break

        # ===== SAVE BEST MODEL =====
        hidden_str = "_".join(map(str, h))
        save_path = f"models/MLP/MLP_{hidden_str}_best.pth"

        torch.save({
            'model_state_dict': best_model_state,
            'best_val_f1': best_f1,
            'best_val_acc': best_acc,
            'best_val_prec': best_prec,
            'best_val_rec': best_rec
        }, save_path)

        print(f"Saved best model → {save_path}")

        # # ===== TEST EVALUATION =====
        # model.load_state_dict(best_model_state)

        # f1_conf, f1_non = evaluate(model, test_loader, device)

        # print(f"TEST → Conf F1: {f1_conf:.4f}, Non-conf F1: {f1_non:.4f}")

#             #save checkpoint every epoch
#             hidden_str="_".join(map(str,h))
#             epoch_filename=f"models/MLP/MLP_hidden_{hidden_str}_epoch_{epoch+1}.pth"
#             torch.save({
#     'epoch': epoch + 1,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'scheduler_state_dict': scheduler.state_dict(),
#     'loss': epoch_loss,
#     'best_val_loss': best_val_loss,
#     'epochs_no_improve': epochs_no_improve
# }, epoch_filename)
#             #early stopping
#             if epoch_loss<best_val_loss:
#                 best_val_loss=epoch_loss
#                 epochs_no_improve=0
#                 best_model_state=copy.deepcopy(model.state_dict())
#             else:
#                 epochs_no_improve+=1
#                 if epochs_no_improve>=patience:
#                     print(f"Early stopping triggered at epoch {epoch+1}")
#                     break
#             #Reduce LR on plateau
#             scheduler.step(epoch_loss)
#         #Save the best model for this hyperparameter combination
#         best_filename=f"models/MLP/MLP_hidden_{hidden_str}_best.pth"
#         torch.save({
#         'epoch': epoch+1,
#         'model_state_dict': best_model_state,
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': best_val_loss}, best_filename)
#         print(f"Best model saved as {best_filename}")
        
if __name__ == "__main__":
    input_size=768
     # Grid search architectures
    hidden_layer_variants = [
        [256],
        [512],
        [1024],
        [512, 256],
        [1024, 512]
    ]
    dropout_rate=0.5
    num_classes=2
   # learning_rate=1e-4,3,2
    learning_rate=1e-4
    weight_decay=1e-5
    # batch_size=4
    num_epochs=200
    patience=5
    device='cuda'
    train_mlp_grid_search(input_size,hidden_layer_variants,dropout_rate,num_classes,learning_rate,weight_decay,num_epochs,patience,device)
    # # val_features="./Features/Val_features"
    # print("MLP training completed. You can now evaluate the trained MLP models using the saved checkpoints.")
    # save_path = "models/MLP/MLP_256_best.pth"  # Update with the actual path to your saved model
    # checkpoint = torch.load(save_path)
    # print("Saved best val F1:", checkpoint['best_val_f1'])
    
    ckpt_paths=[f"./models/MLP/MLP_{"_".join(map(str, h))}_best.pth" for h in hidden_layer_variants]
    for ckpt_path in ckpt_paths:
        print({ckpt_path})
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        f1=checkpoint['best_val_f1']
        acc=checkpoint['best_val_acc']
        prec=checkpoint['best_val_prec']
        rec=checkpoint['best_val_rec']
        print(f"f1{f1}, acc{acc}, prec{prec}, rec{rec}  in {ckpt_path}")
        