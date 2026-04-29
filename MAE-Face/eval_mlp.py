# ...existing code...

import torch
from torch import nn
from torch.utils.data import DataLoader
from mlp import EmotionMLP, FeatureDataset, collate_fn

def evaluate_mlp(model_path, feature_dir, input_size, hidden_layers,
                 dropout_rate=0.1, num_classes=2, device='cuda', batch_size=4):

    # recreate model architecture that matches the saved checkpoint
    model = EmotionMLP(input_size, hidden_layers, dropout_rate, num_classes).to(device)

    # load checkpoint
    ckpt = torch.load(model_path, map_location=device)
    if ckpt.get('model_state_dict') is None:
        raise RuntimeError(f"No 'model_state_dict' in checkpoint {model_path}")
    model.load_state_dict(ckpt['model_state_dict'])

    dataset = FeatureDataset(feature_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')  # sum so we can average later

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0

#need to discuss evaluatate metrics #BUG
    # optional confusion counts for precision/recall/f1 (works for any num_classes)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    with torch.no_grad():
        for features, labels in loader:
            # features: (batch, seq_len, feat_dim), labels: (batch, seq_len)
            B, S, D = features.shape
            features = features.view(-1, D).to(device)           # (B*S, D)
            labels = labels.view(-1).to(device)                  # (B*S,)

            outputs = model(features)                            # (B*S, num_classes)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            mask = labels != -100
            if mask.sum().item() == 0:
                continue
            preds = outputs.argmax(dim=1)

            valid_labels = labels[mask]
            valid_preds = preds[mask]

            total_correct += (valid_preds == valid_labels).sum().item()
            total_valid += mask.sum().item()

            # update confusion counts
            for c in range(num_classes):
                tp[c] += int(((valid_preds == c) & (valid_labels == c)).sum().item())
                fp[c] += int(((valid_preds == c) & (valid_labels != c)).sum().item())
                fn[c] += int(((valid_preds != c) & (valid_labels == c)).sum().item())

    if total_valid == 0:
        raise RuntimeError("No valid labels in validation set (all padded).")

    avg_loss = total_loss / total_valid
    accuracy = total_correct / total_valid

    # per-class precision/recall/f1
    precision = []
    recall = []
    f1 = []
    for c in range(num_classes):
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(p); recall.append(r); f1.append(f)

    results = {
        'avg_loss_per_label': avg_loss,
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'total_valid_labels': total_valid
    }

    return results


# res = evaluate_mlp(
#     model_path="saved_models/MLP_hidden_512_256_best.pth",
#     feature_dir="./features/",
#     input_size=768,
#     hidden_layers=[512, 256],
#     dropout_rate=0.5,
#     num_classes=2,
#     device="cuda"
# )
# print(res)
if __name__ == "__main__":
    hidden_layer_variants = [
        [256],
        [512],
        [1024],
        [512, 256],
        [1024, 512]
    ]
    for h in hidden_layer_variants:
        hidden_str="_".join(map(str,h))
        model_path=f"models/MLP/MLP_hidden_{hidden_str}_best.pth"
        results = evaluate_mlp(
            model_path=model_path,
            feature_dir="./Features/Val_features",
            input_size=768,
            hidden_layers=h,
            dropout_rate=0.5,
            num_classes=2,
            device="cuda"
        )
        print(f"Results for MLP with hidden layers {h}:")
        print(results)