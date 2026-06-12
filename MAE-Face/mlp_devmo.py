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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
import numpy as np


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
# Compute class weights based on the distribution of labels in the dataset.
# ==========================================
def compute_class_weights(dataset, device):
    # Since labels are in a flat numpy array, we can use np.unique for speed
    labels = dataset.labels 
    #bincount like bucket sorts
    #it counts how many times each class label appears in the labels array and returns an array of counts where the index corresponds to the class label. eg: np.bincount([0,0,0,1,1,2]) return([3,2,1])
    counts = np.bincount(labels)

    total = len(labels)
    
    #frequent class gets lower weight, rare class gets higher weight
    weights = [total / (len(counts) * c) if c > 0 else 0 for c in counts]
    print(f"Class counts: {counts}, Weights: {weights}")
    #torch.tensor converts list to tensor
    return torch.tensor(weights, dtype=torch.float32).to(device)

# def build_sampler(dataset, class_weights):
#     # Map each sample's label to its weight
#     # dataset.labels is the mmap array
#     sample_weights = [class_weights[l].item() for l in dataset.labels]
#     return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
# def collate_fn(batch):
#     features_list, labels_list = zip(*batch)  # unzip batch

#     # Pad features with 0
#     features_padded = pad_sequence(features_list, batch_first=True)  # (batch, max_seq_len, feat_dim)

#     # Pad labels with -100 (special value to ignore in loss)
#     labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)  # (batch, max_seq_len)

#     return features_padded, labels_padded

def evaluate_all_thresholds(model, loader, device, thresholds):
    model.eval()
    all_labels, all_probs = [], []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device) 
            labels = labels.to(device)     

            outputs = model(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    val_loss = running_loss / len(loader)

    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_roc = 0.5 

    thresh_metrics = {}
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        
        f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
        prec = precision_score(all_labels, preds, pos_label=1, zero_division=0)
        rec = recall_score(all_labels, preds, pos_label=1, zero_division=0)
        acc = accuracy_score(all_labels, preds)
        kappa = cohen_kappa_score(all_labels, preds)
        
        thresh_metrics[t] = {
            'f1': f1, 'prec': prec, 'rec': rec, 'acc': acc, 'kappa': kappa, 'auc': auc_roc
        }
        
    return thresh_metrics, val_loss

#grid search for hyperparameters


# def train_mlp_grid_search(input_size, hidden_layers, dropout_rate, num_classes, 
#                            learning_rate=1e-3, weight_decay=1e-5, num_epochs=200, 
#                            patience=5, device='cuda'):
    
#     os.makedirs("models/MLP/devmo+", exist_ok=True)
    
#     # Target directory where your pre-extracted fold files are located
#     data_dir = "./Features/Numpy_features"

#     # ===== LOOP OVER ARCHITECTURES =====
#     for h in hidden_layers:
#         hidden_str = "_".join(map(str, h))
#         print(f"==========================================================")
#         print(f"Training MLP Architecture: {h}")
#         print(f"==========================================================")
        
#         # ===== LOOP OVER THE 5 FOLDS =====
#         for fold in range(5):
#             print(f"\n--- Starting Fold {fold} ---")
            
#             # Construct filenames dynamically based on your naming convention
#             train_feat = os.path.join(data_dir, f"Devmo-fold{fold}_feats.npy")
#             train_lab  = os.path.join(data_dir, f"Devmo-fold{fold}_labels.npy")
#             val_feat   = os.path.join(data_dir, f"Devmo-fold{fold}-val_feats.npy")
#             val_lab    = os.path.join(data_dir, f"Devmo-fold{fold}-val_labels.npy")

#             # Dynamic sample count calculations per fold
#             train_count = os.path.getsize(train_feat) // (768 * 4)
#             val_count = os.path.getsize(val_feat) // (768 * 4)

#             # Datasets & Loaders for the current fold
#             train_dataset = FeatureDataset(train_feat, train_lab, num_samples=train_count)
#             val_dataset = FeatureDataset(val_feat, val_lab, num_samples=val_count)

#             train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
#             val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)

#             # Recalculate class weights based specifically on this fold's training split
#             class_weights = compute_class_weights(train_dataset, device)
            
#             # Initialize model and optimization assets per fold
#             model = EmotionMLP(input_size, h, dropout_rate, num_classes).to(device)
#             optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#             criterion = nn.CrossEntropyLoss(weight=class_weights)

#             # Training tracking states
#             best_f1 = 0
#             best_acc = 0
#             best_prec = 0
#             best_rec = 0
#             best_auc_roc = 0
#             best_kappa = 0
#             best_epoch = 0
#             # epochs_no_improve = 0
#             best_model_state = None

#             for epoch in range(num_epochs):
#                 model.train()
#                 running_loss = 0.0
#                 train_preds, train_labels = [], []
                
#                 for features, labels in train_loader:
#                     features = features.to(device) 
#                     labels = labels.to(device)     
                    
#                     optimizer.zero_grad()
#                     outputs = model(features)
#                     loss = criterion(outputs, labels)
#                     loss.backward()
#                     optimizer.step()
                    
#                     running_loss += loss.item()
#                     preds = outputs.argmax(dim=1)
#                     train_preds.extend(preds.cpu().numpy())
#                     train_labels.extend(labels.cpu().numpy())
                
#                 # --- Metrics Logging ---
#                 epoch_loss = running_loss / len(train_loader)
#                 t_f1 = f1_score(train_labels, train_preds, pos_label=1, zero_division=0)
#                 t_acc = accuracy_score(train_labels, train_preds)
                
#                 # --- Validation ---
#                 f1_conf, prec_conf, rec_conf, acc_conf, kappa = evaluate(model, val_loader, device, threshold=0.6)
                
#                 # Print status every 10 epochs (or every epoch if you prefer verbose output)
#                 if (epoch + 1) % 5 == 0 or epoch == 0:
#                     print(f"Epoch {epoch+1:03d} | Train Loss: {epoch_loss:.4f} | Train F1: {t_f1:.4f} | Val F1: {f1_conf:.4f} | Val Acc: {acc_conf:.4f}, Val Prec: {prec_conf:.4f}, Val Rec: {rec_conf:.4f}, Val Kappa: {kappa:.4f}")

#                 # ===== EARLY STOPPING & TRACKING (per fold!) =====
#                 if f1_conf > best_f1:
#                     best_f1 = f1_conf
#                     best_acc = acc_conf
#                     best_prec = prec_conf
#                     best_rec = rec_conf
#                     # best_auc_roc = auc_roc
#                     best_kappa = kappa
#                     best_model_state = copy.deepcopy(model.state_dict())
#                     best_epoch = epoch + 1
#                     # epochs_no_improve = 0
#                 # else:
#                 #     epochs_no_improve += 1

#                 # if epochs_no_improve >= patience:
#                 #     print(f"--> Early stopping triggered at epoch {epoch+1} for Fold {fold}.")
#                 #     break

#             # ===== SAVE BEST MODEL PER FOLD =====
#             # Appending fold index to the save path to avoid overwriting previous folds
#             save_path = f"models/MLP/devmo+/MLP_{hidden_str}_fold{fold}_best.pth"

#             torch.save({
#                 'model_state_dict': best_model_state,
#                 'architecture': h,
#                 'fold': fold,
#                 'best_val_f1': best_f1,
#                 'best_val_acc': best_acc,
#                 'best_val_prec': best_prec,
#                 'best_val_rec': best_rec,
#                 # 'best_val_auc_roc': best_auc_roc,
#                 'best_val_kappa': best_kappa,
#                 'best_epoch': best_epoch
#             }, save_path)

#             print(f"Saved best model for Fold {fold} → {save_path}")
#             print(f"Fold {fold} Results: Best Val F1: {best_f1:.4f} at Epoch {best_epoch}")
def train_mlp_grid_search(input_size, hidden_grid, lr_grid, wd_grid, drop_grid, thresh_grid, 
                           num_classes=2, num_epochs=60, device='cuda'):
    
    data_dir = "./Features/Numpy_features"
    os.makedirs("models/MLP/devmo+2", exist_ok=True)
    
    csv_results_records = []

    # ======================================================================
    # LAYER 1: Loop over different Network Hidden Layer Structures (e.g., [32] vs [64, 32])
    # ======================================================================
    for h in hidden_grid:
        hidden_str = "_".join(map(str, h))
        print(f"\n======================================================================")
        print(f"🔬 EXPLORING ARCHITECTURE STRUCTURE: {h}")
        print(f"======================================================================")
        
        # Tracks the absolute best hyperparameter combination for this specific structure
        structure_best_f1 = -1
        structure_best_model_state = None
        structure_best_meta = {}

        # ======================================================================
        # LAYERS 2-4: Loop over training-time hyperparameters (The "Combo")
        # All 5 folds will be processed under this fixed combination.
        # ======================================================================
        for lr in lr_grid:
            for wd in wd_grid:
                for drop in drop_grid:
                    print(f" Running Config -> LR: {lr} | WD: {wd} | Dropout: {drop}")
                    
                    # Accumulator to store evaluation scores from all 5 folds for every threshold
                    fold_metrics_accumulator = {t: {'f1':[], 'acc':[], 'prec':[], 'rec':[], 'kappa':[], 'auc':[]} for t in thresh_grid}
                    
                    # Temporary storage container for Fold 0's threshold-specific model snapshots
                    fold0_weights_snapshot = None

                    # ======================================================================
                    # LAYER 5: Execute 5-Fold Cross Validation
                    # ======================================================================
                    for fold in range(5):
                        # Construct filenames dynamically based on naming convention
                        train_feat = os.path.join(data_dir, f"Devmo-fold{fold}_feats.npy")
                        train_lab  = os.path.join(data_dir, f"Devmo-fold{fold}_labels.npy")
                        val_feat   = os.path.join(data_dir, f"Devmo-fold{fold}-val_feats.npy")
                        val_lab    = os.path.join(data_dir, f"Devmo-fold{fold}-val_labels.npy")

                        train_count = os.path.getsize(train_feat) // (768 * 4)
                        val_count = os.path.getsize(val_feat) // (768 * 4)

                        train_dataset = FeatureDataset(train_feat, train_lab, num_samples=train_count)
                        val_dataset = FeatureDataset(val_feat, val_lab, num_samples=val_count)

                        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
                        val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)

                        class_weights = compute_class_weights(train_dataset, device)
                        
                        # Initialize a clean model for this specific fold split
                        model = EmotionMLP(input_size, h, drop, num_classes).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
                        criterion = nn.CrossEntropyLoss(weight=class_weights)

                        # Tracks the highest historical metrics achieved within this isolated fold loop
                        fold_best_metrics = {t: {'f1': -1, 'acc':0, 'prec':0, 'rec':0, 'kappa':0, 'auc':0} for t in thresh_grid}
                        
                        # Memory pockets separating best weights for each decision boundary threshold
                        fold_best_weights_per_threshold = {t: None for t in thresh_grid}
                        
                        epochs_no_improve = 0 
                        early_stop_patience = 10

                        # --- Epoch Training Loop ---
                        for epoch in range(num_epochs):
                            model.train()
                            for features, labels in train_loader:
                                features, labels = features.to(device), labels.to(device)     
                                optimizer.zero_grad()
                                outputs = model(features)
                                loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()
                            
                            # Evaluate model across ALL thresholds in parallel
                            current_metrics, val_loss = evaluate_all_thresholds(model, val_loader, device, thresh_grid)
                            scheduler.step(val_loss)
                            
                            any_improvement = False
                            for t in thresh_grid:
                                # Check if this epoch set a new record for this specific threshold
                                if current_metrics[t]['f1'] > fold_best_metrics[t]['f1']:
                                    fold_best_metrics[t] = current_metrics[t]
                                    
                                    # CAPTURE BUG FIX: Save weights into the bucket mapped ONLY to this threshold
                                    fold_best_weights_per_threshold[t] = copy.deepcopy(model.state_dict())
                                    any_improvement = True 
                            
                            if any_improvement:
                                epochs_no_improve = 0
                            else:
                                epochs_no_improve += 1

                            if epochs_no_improve >= early_stop_patience:
                                break
                        # --- End of Epoch Loops for this Fold ---

                        # Append the highest performance achieved by this fold into our cross-validation tracking system
                        for t in thresh_grid:
                            for metric_name in fold_metrics_accumulator[t].keys():
                                fold_metrics_accumulator[t][metric_name].append(fold_best_metrics[t][metric_name])
                        
                        # SPECIAL STEP FOR FOLD 0: 
                        # We copy the entire threshold-weight dictionary into our snapshot variable. 
                        # This preserves Fold 0's optimal weights for all thresholds before they get cleared out of GPU RAM.
                        if fold == 0:
                            fold0_weights_snapshot = copy.deepcopy(fold_best_weights_per_threshold)
                    
                    # ======================================================================
                    # POST-FOLD PROCESSING: Analyze Cross-Validation results for this "Combo"
                    # ======================================================================
                    combo_best_threshold = None
                    combo_best_f1 = -1
                    combo_best_metrics_summary = {}

                    # Calculate the 5-fold average performance for each decision threshold
                    for t in thresh_grid:
                        avg_f1 = np.mean(fold_metrics_accumulator[t]['f1']) #fold 1,2,3,4,5 avg for this threshold
                        
                        # Compare thresholds against each other to find out which one performed best under this combo
                        if avg_f1 > combo_best_f1:
                            combo_best_f1 = avg_f1
                            combo_best_threshold = t
                            combo_best_metrics_summary = {
                                'f1': avg_f1,
                                'auc': np.mean(fold_metrics_accumulator[t]['auc']),
                                'acc': np.mean(fold_metrics_accumulator[t]['acc']),
                                'prec': np.mean(fold_metrics_accumulator[t]['prec']),
                                'rec': np.mean(fold_metrics_accumulator[t]['rec']),
                                'kappa': np.mean(fold_metrics_accumulator[t]['kappa'])
                            }

                    # Now that we know which threshold won the 5-fold cross-validation average, 
                    # we extract the corresponding model weights from our Fold 0 snapshot.
                    combo_best_fold0_weights = fold0_weights_snapshot[combo_best_threshold]

                    # Append exactly 1 row to our CSV record logging the optimal threshold performance for this combo
                    csv_results_records.append({
                        'architecture': hidden_str, 'learning_rate': lr, 'weight_decay': wd,
                        'dropout': drop, 'best_threshold': combo_best_threshold, 
                        'avg_val_f1': combo_best_metrics_summary['f1'],
                        'avg_val_auc': combo_best_metrics_summary['auc'], 
                        'avg_val_acc': combo_best_metrics_summary['acc'], 
                        'avg_val_prec': combo_best_metrics_summary['prec'],
                        'avg_val_rec': combo_best_metrics_summary['rec'], 
                        'avg_val_kappa': combo_best_metrics_summary['kappa']
                    })

                    # Check if this entire hyperparameter combination beats previous combinations tried under this architecture
                    if combo_best_f1 > structure_best_f1:
                        structure_best_f1 = combo_best_f1
                        structure_best_meta = {
                            'lr': lr, 'wd': wd, 'dropout': drop, 'best_threshold': combo_best_threshold,
                            'avg_f1': combo_best_metrics_summary['f1'], 'avg_auc': combo_best_metrics_summary['auc'], 
                            'avg_acc': combo_best_metrics_summary['acc']
                        }
                        # Save these specific weights as the absolute best version of this architecture structure
                        structure_best_model_state = combo_best_fold0_weights

        # Save exactly one `.pth` model file for the current structural layer architecture
        if structure_best_model_state is not None:
            save_path = f"models/MLP/devmo+2/MLP_{hidden_str}_best_structure_model.pth"
            torch.save({
                'model_state_dict': structure_best_model_state,
                'architecture': h,
                'hyperparameters': structure_best_meta
            }, save_path)
            print(f"\n💾 [SAVED MODEL] Top performing model file saved for structure {h} -> {save_path}")

    # Export unique combination statistics to the final CSV file
    results_df = pd.DataFrame(csv_results_records)
    csv_file_path = "models/MLP/devmo+2/mlp_grid_search_results.csv"
    results_df.to_csv(csv_file_path, index=False)
    print(f"\n📊 [SAVED CSV LOG] Grid search results file saved to: {csv_file_path}\n")
if __name__ == "__main__":
    input_size=768
     # Grid search architectures
    hidden_layer_variants = [
        [32],
        [64],
        [64, 32],
        [128],
        [128, 64],
        [256],
        [256, 128],
        [256, 128, 64],
        [512],
        [1024],
        [512, 256],
        [1024, 512]
    ]
    learning_rates = [1e-3, 1e-4]
    weight_decays = [1e-2,1e-3, 1e-4, 1e-5]
    thresholds = [0.3,0.4, 0.5, 0.6,0.7, 0.8]
    dropout_rates=[0.2,0.3, 0.4,0.5,0.6,0.7]
    num_classes=2
   # learning_rate=1e-4,3,2
    # learning_rate=1e-4
    # weight_decay=1e-3
    # batch_size=4
    num_epochs=100
    patience=10
    device='cuda'
    #read the best models for each architecture 
    for h in hidden_layer_variants:
        save_path = f"models/MLP/devmo+2/MLP_{'_'.join(map(str, h))}_best_structure_model.pth"
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location='cpu', weights_only=False)
            
            hyperparameters = checkpoint['hyperparameters']
            
            print(f"Architecture: {h} | lr: {hyperparameters['lr']} |wd: {hyperparameters['wd']} | dropout: {hyperparameters['dropout']} |threshold: {hyperparameters['best_threshold']} | Fold 0 Best Val F1: {hyperparameters['avg_f1']:.4f} | Val Acc: {hyperparameters['avg_acc']:.4f},  Val AUC-ROC: {hyperparameters['avg_auc']:.4f}")
        else:
            print(f"No saved model found for architecture {h} at path: {save_path}")
    

    # train_mlp_grid_search(
    #     input_size=input_size,
    #     hidden_grid=hidden_layer_variants,
    #     lr_grid=learning_rates,
    #     wd_grid=weight_decays,
    #     drop_grid=dropout_rates,
    #     thresh_grid=thresholds,
    #     num_classes=num_classes,
    #     num_epochs=num_epochs,
    #     device=device
    # )
    # train_mlp_grid_search(input_size,hidden_layer_variants,dropout_rate,num_classes,learning_rate,weight_decay,num_epochs,patience,device)
    # # val_features="./Features/Val_featureßs"
    # print("MLP training completed. You can now evaluate the trained MLP models using the saved checkpoints.")
    # save_path = "models/MLP/MLP_256_best.pth"  # Update with the actual path to your saved model
    # checkpoint = torch.load(save_path)
    # print("Saved best val F1:", checkpoint['best_val_f1'])
    
    
    # for h in hidden_layer_variants:
    #     ckpt_paths=[]
    #     for fold in range(5):
    #         ckpt_paths.append(f"./models/MLP/devmo+/MLP_{"_".join(map(str, h))}_fold{fold}_best.pth")
    #     #calculate the average of the 5 folds
    #     f1s=[]
    #     accs=[]
    #     precs=[]
    #     recs=[]
    #     kappas=[]
    #     for ckpt_path in ckpt_paths:
    #         checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    #         f1=checkpoint['best_val_f1']
    #         acc=checkpoint['best_val_acc']
    #         prec=checkpoint['best_val_prec']
    #         rec=checkpoint['best_val_rec']
    #         # auc_roc=checkpoint['best_val_auc_roc']
    #         kappa=checkpoint['best_val_kappa']
    #         epoch=checkpoint['best_epoch']
    #         f1s.append(f1)
    #         accs.append(acc)
    #         precs.append(prec)
    #         recs.append(rec)
    #         kappas.append(kappa)
    #     #average
    #     avg_f1=np.mean(f1s)
    #     avg_acc=np.mean(accs)
    #     avg_prec=np.mean(precs)
    #     avg_rec=np.mean(recs)
    #     avg_kappa=np.mean(kappas)
    #     print(f"Architecture: {h} | Average Val F1: {avg_f1:.4f} | Average Val Acc: {avg_acc:.4f} | Average Val Prec: {avg_prec:.4f} | Average Val Rec: {avg_rec:.4f} | Average Val Kappa: {avg_kappa:.4f}")

        # f1s=[]  
        # for ckpt_path in ckpt_paths:
        # print({ckpt_path})
        # checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # f1=checkpoint['best_val_f1']
        # acc=checkpoint['best_val_acc']
        # prec=checkpoint['best_val_prec']
        # rec=checkpoint['best_val_rec']
        # # auc_roc=checkpoint['best_val_auc_roc']
        # kappa=checkpoint['best_val_kappa']
        # epoch=checkpoint['best_epoch']
        # print(f"f1{f1}, acc{acc}, prec{prec}, rec{rec}, kappa{kappa}, epoch{epoch} in {ckpt_path}")
        