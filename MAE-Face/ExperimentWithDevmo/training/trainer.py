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
from ..models.metrics import Metrics
from .evaluator import Evaluator
#training model
class Trainer:
    def __init__(self,batch_size,device,lr,wd,train_dataset,val_dataset,model,num_epochs,thresh_grid):
        self.batch_size=batch_size
        self.device=device
        self.lr=lr
        self.wd=wd
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader=DataLoader(val_dataset,batch_size=self.batch_size,shuffle=True,num_workers=4)
        self.num_epochs=num_epochs
        class_weights = self.__compute_class_weights(train_dataset, self.device)
                # Initialize a clean model for this specific fold split
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.thresh_grid=thresh_grid

    def __compute_class_weights(self,dataset, device):
        # Since labels are in a flat numpy array, we can use np.unique for speed
        labels = []
        for _, y in dataset:
            labels.append(int(y))
        #bincount like bucket sorts
        #it counts how many times each class label appears in the labels array and returns an array of counts where the index corresponds to the class label. eg: np.bincount([0,0,0,1,1,2]) return([3,2,1])
        counts = np.bincount(labels)
        total = len(labels)
        #frequent class gets lower weight, rare class gets higher weight
        weights = [total / (len(counts) * c) if c > 0 else 0 for c in counts]
        #torch.tensor converts list to tensor
        return torch.tensor(weights, dtype=torch.float32).to(self.device)
    def __train_one_epoch(self,):
        self.model.train()
        total_loss = 0
        for features, labels in self.train_loader:
            features, labels = features.to(self.device), labels.to(self.device)     
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)  
     
    def train_one_fold( self, fold_best_metrics,fold_best_weights_per_threshold,evaluator):
        epochs_no_improve = 0
        early_stop_patience = 10
        for epoch in range(self.num_epochs):
            #train model
            self.__train_one_epoch()
            #evaluate model for this epoch
            current_metrics,val_loss=evaluator.evaluate_all_thresholds(self.model, self.val_loader,self.thresh_grid)
            self.scheduler.step(val_loss)
            #early stop
            any_improvement=False
            for t in self.thresh_grid:
            #check if this epoch set a new record for this specific threshold
                if current_metrics[t].f1>fold_best_metrics[t].f1:
                    fold_best_metrics[t]=current_metrics[t]
                    fold_best_metrics[t].epoch=epoch
                    fold_best_weights_per_threshold[t]=copy.deepcopy(self.model.state_dict())
                    any_improvement=True
            if any_improvement:
                epochs_no_improve=0
            else:
                epochs_no_improve+=1
            if epochs_no_improve>=early_stop_patience:
                break




