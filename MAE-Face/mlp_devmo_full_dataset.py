import os
import numpy as np
import torch
import pandas as pd

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score


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

    all_y = np.array(all_y)
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
def compute_class_weights(dataset, device):
    labels = dataset.labels
    counts = np.bincount(labels)

    total = len(labels)
    weights = [total / (len(counts) * c) if c > 0 else 0 for c in counts]

    return torch.tensor(weights, dtype=torch.float32).to(device)


# =========================
# train function (IMPORTANT)
# =========================
def train_model(model, train_loader, lr, wd, epochs, device):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

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

    results = []

    hidden_layer_variants = [
        [32], [64], [64, 32], [128], [128, 64],
        [256], [256, 128], [256, 128, 64],
        [512], [1024], [512, 256], [1024, 512]
    ]

    # =========================
    # A dataset
    # =========================
    A_train_feat = f"{data_dir}/Devmo-train_feats.npy"
    A_train_label = f"{data_dir}/Devmo-train_labels.npy"
    A_test_feat  = f"{data_dir}/Devmo-test_feats.npy"
    A_test_label = f"{data_dir}/Devmo-test_labels.npy"

    # =========================
    # B dataset
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

    for h in hidden_layer_variants:

        print("\n====================")
        print("Structure:", h)

        # load hyperparams ONLY (no weights needed)
        ckpt_path = f"models/MLP/devmo+2/MLP_{'_'.join(map(str,h))}_best_structure_model.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = ckpt["hyperparameters"]

        # =========================
        # rebuild model (fresh)
        # =========================
        model = EmotionMLP(
            input_size=768,
            hidden_layers=h,
            dropout_rate=hp["dropout"],
            num_classes=2
        ).to(device)

        # =========================
        # A + B training
        # =========================
        train_dataset = ConcatDataset([
            FeatureDataset(A_train_feat, A_train_label, num_A_train),
            FeatureDataset(B_train_feat, B_train_label, num_B_train)
        ])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        train_model(
            model,
            train_loader,
            lr=hp["lr"],
            wd=hp["wd"],
            epochs=15,   # fast run for meeting
            device=device
        )

        # =========================
        # evaluation A
        # =========================
        test_loader_A = DataLoader(
            FeatureDataset(A_test_feat, A_test_label, num_A_test),
            batch_size=64
        )

        metrics_A = test_model(model, test_loader_A, hp["best_threshold"], device)

        # =========================
        # evaluation B
        # =========================
        test_loader_B = DataLoader(
            FeatureDataset(B_test_feat, B_test_label, num_B_test),
            batch_size=64
        )

        metrics_B = test_model(model, test_loader_B, hp["best_threshold"], device)

        # =========================
        # save
        # =========================
        results.append({
            "structure": str(h),

            "A_f1": metrics_A["f1"],
            "A_acc": metrics_A["acc"],
            "A_auc": metrics_A["auc"],
            "A_kappa": metrics_A["kappa"],

            #"prec": precision_score(all_y, pred),
            "A_rec": metrics_A["rec"],
            #"kappa": cohen_kappa_score(all_y, pred),
            "A_prec": metrics_A["prec"],

            "B_f1": metrics_B["f1"],
            "B_acc": metrics_B["acc"],
            "B_auc": metrics_B["auc"],
            "B_kappa": metrics_B["kappa"],
            "B_rec": metrics_B["rec"],
            "B_prec": metrics_B["prec"],
        })

        print("A:", metrics_A)
        print("B:", metrics_B)

    # =========================
    # save CSV
    # =========================
    df = pd.DataFrame(results)
    df.to_csv("A_B_joint_training_fast_results.csv", index=False)

    print("\nDONE")