import copy

import os
import numpy as np
import torch
import pandas as pd

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset

from mlp_devmo import evaluate_all_thresholds


# =========================
# Dataset
# =========================
class FeatureDataset(Dataset):
    def __init__(self, feat_path, label_path, num_samples, feat_dim=768):
        self.features = np.memmap(feat_path, dtype='float32', mode='r',
                                  shape=(num_samples, feat_dim))
        self.labels = np.memmap(label_path, dtype='int64', mode='r',
                                shape=(num_samples,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx].copy()).float()
        y = torch.tensor(self.labels[idx]).long()
        return x, y


# =========================
# Model
# =========================
class EmotionMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate, num_classes=2):
        super().__init__()

        layers = []
        in_dim = input_size

        for h in hidden_layers:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
# metrics
# =========================
def test_model(model, loader, threshold, device):
    model.eval()
    all_y, all_p = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_y.extend(y.cpu().numpy())
            all_p.extend(probs.cpu().numpy())

    all_y = np.array(all_y)#为啥转换成np.array？因为sklearn的metrics函数需要numpy数组作为输入，而不是Python列表。将all_y和all_p转换成numpy数组可以确保它们具有正确的格式和类型，以便在计算指标时不会出现错误。此外，numpy数组还提供了更高效的计算和内存管理，特别是在处理大型数据集时。因此，将all_y和all_p转换成numpy数组是为了兼容sklearn的函数，并提高计算效率。 
    all_p = np.array(all_p)

    pred = (all_p >= threshold).astype(int)

    return {
        "f1": f1_score(all_y, pred),
        "acc": accuracy_score(all_y, pred),
        "prec": precision_score(all_y, pred),
        "rec": recall_score(all_y, pred),
        "kappa": cohen_kappa_score(all_y, pred),
        "auc": roc_auc_score(all_y, all_p),
    }


# =========================
# class weights
# =========================
def compute_class_weights(labels, device):
    labels = np.asarray(labels)
    counts = np.bincount(labels)

    total = len(labels)
    weights = [total / (len(counts) * c) if c > 0 else 0 for c in counts]

    return torch.tensor(weights, dtype=torch.float32).to(device)

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

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device
):
    model.train()

    total_loss = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def train_one_fold(num_epochs,model,train_loader, optimizer,criterion,device,val_loader,thresh_grid,scheduler, fold_best_metrics,fold_best_weights_per_threshold):
    epochs_no_improve = 0
    early_stop_patience = 10
    for epoch in range(num_epochs):
        #train model
        train_one_epoch(model,train_loader,optimizer, criterion,device)
        #evaluate model for this epoch
        current_metrics,val_loss=evaluate_all_thresholds(model, val_loader,device,thresh_grid)
        scheduler.step(val_loss)
        #early stop
        any_improvement=False
        for t in thresh_grid:
            #check if this epoch set a new record for this specific threshold
            if current_metrics[t]['f1']>fold_best_metrics[t]['f1']:
                fold_best_metrics[t]=current_metrics[t]
                fold_best_metrics[t]['epoch']=epoch
                fold_best_weights_per_threshold[t]=copy.deepcopy(model.state_dict())
                any_improvement=True
        if any_improvement:
            epochs_no_improve=0
        else:
            epochs_no_improve+=1
        if epochs_no_improve>=early_stop_patience:
            break

def evaluate_combo(architecture,lr,wd,drop,thresh_grid,device,train_dataset,input_size,num_epochs,num_classes):
    all_indices=np.arange(len(train_dataset))
    all_labels=np.concatenate([train_dataset.datasets[0].labels, train_dataset.datasets[1].labels])
    skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics_accumulator={t:{'f1':[], 'acc':[],'prec':[],'rec':[],'kappa':[],'auc':[],'epoch':[]} for t in thresh_grid}
    #epochs_acuumulator
    #{'t1':}
    fold0_weights_snapshot=None
    for fold,(train_idx,val_idx) in enumerate(skf.split(all_indices, all_labels)):
        train_subset=Subset(train_dataset,train_idx)
        val_subset=Subset(train_dataset,val_idx)
        train_loader= DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
        val_loader=DataLoader(val_subset, batch_size=64, num_workers=4)
        class_weights=compute_class_weights(all_labels[train_idx],device)
        model=EmotionMLP(input_size,architecture,drop,num_classes).to(device)
        optimizer=optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
        scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion=nn.CrossEntropyLoss(weight=class_weights)
        fold_best_metrics={t:{'f1':-1, 'acc':0, 'prec':0, 'rec':0, 'kappa':0, 'auc':0,'epoch':-1} for t in thresh_grid}
        fold_best_weights_per_threshold={t:None for t in thresh_grid}
        train_one_fold(num_epochs,model,train_loader,optimizer,criterion,device,val_loader,thresh_grid,scheduler, fold_best_metrics,fold_best_weights_per_threshold)
        for t in thresh_grid:
            for metric_name in fold_metrics_accumulator[t].keys():
                fold_metrics_accumulator[t][metric_name].append(fold_best_metrics[t][metric_name])
        if fold==0: 
            fold0_weights_snapshot=copy.deepcopy(fold_best_weights_per_threshold)
    
    #calculate 5-fold average performance for each threshold
    combo_best_f1=-1
    combo_best_threshold=None
    combo_best_epoch=None
    combo_best_metrics_summary={}
    for t in thresh_grid:
        avg_f1=np.mean(fold_metrics_accumulator[t]['f1'])
        if avg_f1>combo_best_f1:
            combo_best_f1=avg_f1
            combo_best_threshold=t
            combo_best_metrics_summary={
                'f1':avg_f1,
                'auc':np.mean(fold_metrics_accumulator[t]['auc']),
                'acc':np.mean(fold_metrics_accumulator[t]['acc']),
                'prec':np.mean(fold_metrics_accumulator[t]['prec']),
                'rec':np.mean(fold_metrics_accumulator[t]['rec']),
                'kappa':np.mean(fold_metrics_accumulator[t]['kappa']),
                'epoch':int(np.median(fold_metrics_accumulator[t]['epoch'])),
            }
    combo_best_fold0_weights=fold0_weights_snapshot[combo_best_threshold]
    return {'architecture': architecture, 'learning_rate': lr, 'weight_decay': wd, 'dropout': drop, 'best_threshold': combo_best_threshold, 'combo_best_f1': combo_best_metrics_summary['f1'], 'combo_best_auc': combo_best_metrics_summary['auc'], 'combo_best_acc': combo_best_metrics_summary['acc'], 'combo_best_prec': combo_best_metrics_summary['prec'], 'combo_best_rec': combo_best_metrics_summary['rec'], 'combo_best_kappa': combo_best_metrics_summary['kappa'], 'combo_best_epoch': combo_best_metrics_summary['epoch']},combo_best_fold0_weights


                                        
def final_training_and_evaluation(hidden_grid, train_dataset,devmo_test_dataset, daisee_test_dataset, device):
    results=[]
    for h in hidden_grid:
        print("\n====================")
        print("Structure:", h)

        ckpt_path=f"models/MLP/mixed/MLP_{'_'.join(map(str,h))}_best_structure_model.pth"
        ckpt=torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp=ckpt["hyperparameters"]
        model=EmotionMLP(input_size=768, hidden_layers=h, dropout_rate=hp["dropout"], num_classes=2).to(device)
        
        train_model(model, train_dataset, lr=hp["lr"], wd=hp["wd"], epochs=hp['structure_best_epoch'], device=device)
        test_loader_A=DataLoader(devmo_test_dataset, batch_size=64)
        metrics_A=test_model(model, test_loader_A, hp["best_threshold"], device)
        test_loader_B=DataLoader(daisee_test_dataset, batch_size=64)
        metrics_B=test_model(model, test_loader_B, hp["best_threshold"], device)

        save_path=f"models/MLP/mixed/MLP_{'_'.join(map(str,h))}_final_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': h,
            'hyperparameters': hp, 
        }, save_path)
        results.append({
            "structure": str(h),
            "metrics_A": metrics_A,
            "metrics_B": metrics_B
        })
        print("A:", metrics_A)
        print("B:", metrics_B)

    #save CSV
    df=pd.DataFrame(results)
    df.to_csv("A_B_joint_training_final_results.csv", index=False)


    
def train_mlp_grid_search(input_size, hidden_grid, lr_grid, wd_grid, drop_grid, thresh_grid, 
                           num_classes=2, num_epochs=60, device='cuda', train_dataset=None):
    
    data_dir = "./Features/Numpy_features"
    os.makedirs("models/MLP/mixed", exist_ok=True)
    
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

                    combo_results, combo_best_fold0_model=evaluate_combo(
                        architecture=h, lr=lr, wd=wd, drop=drop, thresh_grid=thresh_grid, device=device,
                        train_dataset=train_dataset, input_size=input_size, num_epochs=num_epochs, num_classes=num_classes
                    )
                    csv_results_records.append(combo_results)

                    # Check if this entire hyperparameter combination beats previous combinations tried under this architecture
                    if combo_results['combo_best_f1'] > structure_best_f1:
                        structure_best_f1 = combo_results['combo_best_f1']
                        structure_best_meta = {
                            'lr': lr, 'wd': wd, 'dropout': drop, 'best_threshold': combo_results['best_threshold'],
                            'structure_best_f1': combo_results['combo_best_f1'], 'structure_best_auc': combo_results['combo_best_auc'], 
                            'structure_best_acc': combo_results['combo_best_acc'],
                            'structure_best_prec': combo_results['combo_best_prec'], 'structure_best_rec': combo_results['combo_best_rec'],
                            'structure_best_kappa': combo_results['combo_best_kappa'], 'structure_best_epoch': combo_results['combo_best_epoch']
                        }
                        # Save these specific weights as the absolute best version of this architecture structure
                        structure_best_model_state = combo_best_fold0_model

        # Save exactly one `.pth` model file for the current structural layer architecture
        if structure_best_model_state is not None:

            save_path = f"models/MLP/mixed/MLP_{hidden_str}_best_structure_model.pth"
            torch.save({
                'model_state_dict': structure_best_model_state,
                'architecture': h,
                'hyperparameters': structure_best_meta
            }, save_path)
            print(f"\n💾 [SAVED MODEL] Top performing model file saved for structure {h} -> {save_path}")

    # Export unique combination statistics to the final CSV file
    results_df = pd.DataFrame(csv_results_records)
    csv_file_path = "models/MLP/mixed/mlp_grid_search_results.csv"
    results_df.to_csv(csv_file_path, index=False)
    print(f"\n📊 [SAVED CSV LOG] Grid search results file saved to: {csv_file_path}\n")
                   
# =========================
# train function (IMPORTANT)
# =========================
def train_model(model, train_dataset, lr, wd, epochs, device):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    all_labels=np.concatenate([train_dataset.datasets[0].labels, train_dataset.datasets[1].labels])
    class_weights = compute_class_weights(all_labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    train_loader=DataLoader(train_dataset, batch_size=64,shuffle=True)

    model.train()

    for ep in range(epochs):
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {ep+1}/{epochs} | loss={total_loss/len(train_loader):.4f}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    device = "cuda"
    data_dir = "./Features/Numpy_features"

    #训练中断，最后的时候在还原
    hidden_layer_variants = [
        [64], [128], [128, 64],
        [256], [256, 128], [256, 128, 64],
        [512], [1024], [512, 256], [1024, 512]
    ]
    # hidden_layer_variants = [
    #     [1024, 512]
    # ]
    learning_rates = [1e-3, 1e-4]
    weight_decays = [1e-2,1e-3, 1e-4]
    thresholds = [0.3,0.4, 0.5, 0.6,0.7, 0.8]
    dropout_rates=[0.2,0.3, 0.4,0.5]
    num_classes=2

    # =========================
    # devmo dataset
    # =========================
    A_train_feat = f"{data_dir}/Devmo-train_feats.npy"
    A_train_label = f"{data_dir}/Devmo-train_labels.npy"
    A_test_feat  = f"{data_dir}/Devmo-test_feats.npy"
    A_test_label = f"{data_dir}/Devmo-test_labels.npy"

    # =========================
    # daisee dataset
    # =========================
    B_train_feat = f"{data_dir}/Train_v2_feats_cc.npy"
    B_train_label = f"{data_dir}/Train_v2_labels_cc.npy"
    B_test_feat  = f"{data_dir}/Val_feats_cc.npy"
    B_test_label = f"{data_dir}/Val_labels_cc.npy"

    # counts
    num_A_train = os.path.getsize(A_train_feat) // (768 * 4)
    num_A_test  = os.path.getsize(A_test_feat) // (768 * 4)
    num_B_train = os.path.getsize(B_train_feat) // (768 * 4)
    num_B_test  = os.path.getsize(B_test_feat) // (768 * 4)

    train_dataset = ConcatDataset([
            FeatureDataset(A_train_feat, A_train_label, num_A_train),
            FeatureDataset(B_train_feat, B_train_label, num_B_train)
        ])

    #grid search for mixed_ train_dataset
    input_size = 768

    # train_mlp_grid_search(
    #     input_size=input_size,
    #     hidden_grid=hidden_layer_variants,
    #     lr_grid=learning_rates,
    #     wd_grid=weight_decays,
    #     drop_grid=dropout_rates,
    #     thresh_grid=thresholds,
    #     num_classes=num_classes,
    #     num_epochs=60,
    #     device=device,
    #     train_dataset=train_dataset
    # )


    #最后做
    devmo_test_dataset=FeatureDataset(A_test_feat, A_test_label, num_A_test)
    daisee_test_dataset=FeatureDataset(B_test_feat, B_test_label, num_B_test)

    final_training_and_evaluation(
        hidden_grid=hidden_layer_variants, 
        train_dataset=train_dataset,
        devmo_test_dataset=devmo_test_dataset,
        daisee_test_dataset=daisee_test_dataset,
        device=device
        )
   