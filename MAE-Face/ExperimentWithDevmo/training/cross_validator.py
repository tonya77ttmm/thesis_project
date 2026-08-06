import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from ..models.emotion_mlp import EmotionMLP
from torch.utils.data import Subset
from ..training.trainer import Trainer
from ..models.metrics import Metrics
import copy
import os

class CrossValidator:
    def __init__(self,train_dataset,device,thresh_grid,num_epochs,evaluator):
        self.train_dataset=train_dataset
        self.device=device
        self.thresh_grid=thresh_grid
        self.num_epochs=num_epochs
        self.evaluator=evaluator
    def evaluate_combo(self,architecture,lr,wd,drop,input_size,num_classes):
        all_indices=np.arange(len(self.train_dataset))
        all_labels=np.array(self.train_dataset.labels)
        groups=np.array(self.train_dataset.clip_ids)
        skf=StratifiedGroupKFold(n_splits=5,shuffle=True, random_state=42)
        fold_metrics_accumulator={t:[] for t in self.thresh_grid} #[Metrics1, Metrics2,...Metrics5]
        best_fold_f1 = -1
        best_fold_weights = None
        for fold,(train_idx,val_idx) in enumerate(skf.split(all_indices, all_labels, groups)):
            train_subset=Subset(self.train_dataset,train_idx)
            val_subset=Subset(self.train_dataset,val_idx)
            model=EmotionMLP(input_size,architecture,drop,num_classes).to(self.device)
            fold_best_weights_per_threshold,fold_best_metrics=self.__evaluate_one_fold(train_subset,val_subset,model,lr,wd)
            for t in self.thresh_grid:
                fold_metrics_accumulator[t].append(fold_best_metrics[t])
                print(f"Fold {fold} completed. Best metrics per threshold: {fold_best_metrics}")
            fold_best_threshold = max(self.thresh_grid,key=lambda t: fold_best_metrics[t].f1)
            fold_best_f1 = fold_best_metrics[fold_best_threshold].f1
            if fold_best_f1 > best_fold_f1:
                best_fold_f1 = fold_best_f1
                best_fold_weights = copy.deepcopy(fold_best_weights_per_threshold[fold_best_threshold])
           
        #calculate 5-fold average performance for each threshold
        combo_best_metrics,combo_best_threshold=self.__summarize_cv_results(fold_metrics_accumulator)
        # combo_best_fold0_weights=fold0_weights_snapshot[combo_best_threshold]
        return {'architecture': architecture, 'learning_rate': lr, 'weight_decay': wd, 'dropout': drop, 'best_threshold': combo_best_threshold, 'combo_best_metrics':combo_best_metrics},best_fold_weights


    def __evaluate_one_fold(self,train_dataset,val_dataset,model,lr,wd):
        trainer=Trainer(64,self.device,lr,wd,train_dataset,val_dataset,model,self.num_epochs,self.thresh_grid)
        fold_best_metrics={t:Metrics(-1,0,0,0,0,0,-1) for t in self.thresh_grid}
        fold_best_weights_per_threshold={t:None for t in self.thresh_grid} #but even threshold is different, they share the same weights, who to modify it 
        trainer.train_one_fold(fold_best_metrics,fold_best_weights_per_threshold,self.evaluator)
        return fold_best_weights_per_threshold,fold_best_metrics

    def __summarize_cv_results(self, fold_metrics_accumulator):
        combo_best_f1=-1
        combo_best_threshold=None
        combo_best_epoch=None
        combo_best_metrics_summary={}
        for t in self.thresh_grid:
            avg_f1=np.mean([m.f1 for m in fold_metrics_accumulator[t]])
            if avg_f1>combo_best_f1:
                combo_best_f1=avg_f1
                combo_best_threshold=t
                combo_best_metrics_summary=Metrics(
                    f1=avg_f1,
                    auc=np.mean([m.auc for m in fold_metrics_accumulator[t]]),
                    acc=np.mean([m.acc for m in fold_metrics_accumulator[t]]),
                    prec=np.mean([m.prec for m in fold_metrics_accumulator[t]]),
                    rec=np.mean([m.rec for m in fold_metrics_accumulator[t]]),
                    kappa=np.mean([m.kappa for m in fold_metrics_accumulator[t]]),
                    epoch=int(np.median([m.epoch for m in fold_metrics_accumulator[t]])),
                )
        return combo_best_metrics_summary, combo_best_threshold

