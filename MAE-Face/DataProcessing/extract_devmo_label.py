import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# Load json
devmo_label_path = "../../confusion_dataset/Devmo/devemo+/devemo+.json"
with open(devmo_label_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract needed columns
rows = []

for item in data:
    filename = item["filename"]
    confusion_label = 1 if item["label"] == "Confusion" else 0

    rows.append({
        "clipID": filename,
        "label": confusion_label
    })

# Create dataframe
df = pd.DataFrame(rows)

print("Total samples:", len(df))
print("Label distribution:\n", df["label"].value_counts())

# =========================
# 1. Train / Test split (20% test)
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]   # ⭐保证0/1比例一致
)

print("\nTrain size:", len(train_df))
print("Test size:", len(test_df))

print("\nTrain label distribution:\n", train_df["label"].value_counts())
print("\nTest label distribution:\n", test_df["label"].value_counts())

# =========================
# 2. K-Fold on TRAIN ONLY
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

folds = []
save_dir = "../../confusion_dataset/Devmo/devemo+/"

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
    fold_train = train_df.iloc[train_idx]
    fold_val = train_df.iloc[val_idx]
    fold_train.to_csv(save_dir + f"fold_{fold}_train.csv", index=False)
    fold_val.to_csv(save_dir + f"fold_{fold}_val.csv", index=False)
    folds.append({
        "fold": fold,
        "train": fold_train,
        "val": fold_val
    })

    print(f"\nFold {fold}")
    print("Train:", len(fold_train), "Val:", len(fold_val))
    print("Train label dist:\n", fold_train["label"].value_counts())
    print("Val label dist:\n", fold_val["label"].value_counts())

# =========================
# 3. Save splits
# =========================

test_df.to_csv(save_dir + "test.csv", index=False)
train_df.to_csv(save_dir + "train.csv", index=False)

# for fold in folds:
#     fold["train"].to_csv(f"{save_dir}fold_{fold['fold']}_train.csv", index=False)
#     fold["val"].to_csv(f"{save_dir}fold_{fold['fold']}_val.csv", index=False)

# print("\nDone saving splits!")