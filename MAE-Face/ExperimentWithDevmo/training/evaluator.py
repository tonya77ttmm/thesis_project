# =========================
# metrics
# =========================
from ..models.metrics import Metrics
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score

#  computes metrics

class Evaluator:
    def __init__(self,device,):
        self.device=device
        self.criterion = nn.CrossEntropyLoss()
    def test_model(self, model, loader, threshold):
        all_y, all_p,_=self.__run_inference(model,loader)

        pred = (all_p >= threshold).astype(int)

        return Metrics(
        f1= f1_score(all_y, pred),
        acc= accuracy_score(all_y, pred),
        prec= precision_score(all_y, pred),
        rec =recall_score(all_y, pred),
        kappa= cohen_kappa_score(all_y, pred),
        auc=roc_auc_score(all_y, all_p),
    )
    def __run_inference(self,model,loader):
        model.eval()
        all_labels, all_probs = [], []
        running_loss = 0.0
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device) 
                labels = labels.to(self.device)     

                outputs = model(features)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        val_loss = running_loss / len(loader)  

        return all_labels, all_probs, val_loss      
    def evaluate_all_thresholds(self,model, loader, thresholds):
        all_labels, all_probs,val_loss=self.__run_inference(model,loader)
        auc_roc = roc_auc_score(all_labels, all_probs)
        thresh_metrics = {}
        for t in thresholds:
            preds = (all_probs >= t).astype(int)
        
            f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
            prec = precision_score(all_labels, preds, pos_label=1, zero_division=0)
            rec = recall_score(all_labels, preds, pos_label=1, zero_division=0)
            acc = accuracy_score(all_labels, preds)
            kappa = cohen_kappa_score(all_labels, preds)
        
            thresh_metrics[t] = Metrics(f1=f1,prec=prec,rec=rec,acc=acc,kappa=kappa,auc=auc_roc)
        return thresh_metrics, val_loss




