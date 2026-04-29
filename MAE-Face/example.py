# This script is a simple example for how to:
# 1. create the model
# 2. load the pre-trained weights
# 3. evaluate the model

import torch
import models_vit
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# model parameters
model_name = 'vit_base_patch16'
ckpt_path = './models/mae_face_pretrain_vit_base.pth'
global_pool = True # recommend: True for most cases, False if you want to evaluate the features from the pre-trained model without fine-tuning
num_heads = 12 # specify the number of classification heads for the downstream task

device = 'cuda'
batch_size = 64

# create model
model = getattr(models_vit, model_name)(
    global_pool=global_pool,
    num_classes=num_heads,
    drop_path_rate=0.1,
    img_size=224,
)

# load pre-trained weights
print(f"Load pre-trained checkpoint from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu',weights_only=False)
checkpoint_model = checkpoint['model']
msg = model.load_state_dict(checkpoint_model, strict=False)
print(msg) # print which weights are not loaded
model.to(device)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
# ])

# simple test for evaluation
# `features` can be used as the facial representation for downstream tasks
model.eval()
with torch.no_grad():
    inputs = torch.ones([batch_size, 3, 224, 224]).to(device)
    outputs, features = model(inputs, ret_feature=True)

    outputs, features = model(inputs, ret_feature=True)

print("Feature shape:", features.shape)
print("Features",features)
print("Output", outputs)


# Or fine-tune the model with your datasets for downstream tasks
# ......
