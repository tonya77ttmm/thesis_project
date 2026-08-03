# train an MLP classifier with the features and labels
#define the MLP classifier
from torch import nn
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
