import DataLoader from torch.utils.data
import torch
from ..models.metrics import Metrics
from ..models.emotion_mlp import EmotionMLP
from ..training.evaluator import Evaluator

class Tester:
    def __init__(self, test_dataset, device):
        self.device=device
        self.test_loader=DataLoader(test_dataset, batch_size=64)
        self.evaluator=Evaluator(device)
    def __load_model(self, ckpt_path, h):
        #load model to device
        ckpt=torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp=ckpt["hyperparameters"]
        model=EmotionMLP(input_size=768, hidden_layers=h, dropout_rate=hp["dropout"], num_classes=2).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model,hp["best_threshold"]
            
    def test_model(self,ckpt_path,h):
        model,t=self.__load_model(ckpt_path,h)
        metrics=self.evaluator.evaluate_threshold(model,self.test_loader,t)
        return metrics
       
