from ..models import models_vit
from .feature_extractor import FeatureExtractor
class MAEExtractor(FeatureExtractor):
    def __init__(self,ckpt_path,device,feature_dim):
        super().__init__(ckpt_path,device)
        model_name = 'vit_base_patch16'
        self.model = getattr(models_vit, model_name)(
            global_pool=True, num_classes=2, drop_path_rate=0.1, img_size=224,
        )
        self.feature_dim=feature_dim
        self._load_model()

