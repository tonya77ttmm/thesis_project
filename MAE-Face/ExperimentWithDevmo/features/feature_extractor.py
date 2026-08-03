import torch
class FeatureExtractor:
    def __init__(self,ckpt_path,device):
        self.ckpt_path=ckpt_path
        self.device=device
        self.model = None

    def _load_model(self,):
        checkpoint = torch.load(self.ckpt_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.to(self.device).eval()
        
    #iterator
    def extract(self, batch_frames):
        batch_frames = batch_frames.to(self.device)
        with torch.no_grad():
            _,features=self.model(batch_frames, ret_feature=True)
        return features