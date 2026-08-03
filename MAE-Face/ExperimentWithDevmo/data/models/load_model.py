import torch
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

model_path = (
    PROJECT_ROOT
    / "ExperimentWithDevmo"
    / "data"
    / "models"
    / "MLP_best_structure_model_256_128.pth"
)

print("Loading:", model_path)

checkpoint = torch.load(
    model_path,
    map_location="cpu",
    weights_only=False
)

#   # adjust import based on your structure
# from dataset.frame_dataset import FrameDataset
# from torch.utils.data import DataLoader


# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# hidden_str = "256_128"  # change to your saved model name

# model_path = (
#     f"MLP_best_structure_model_{hidden_str}.pth"
# )



# checkpoint = torch.load(

#     model_path,

#     map_location="cpu",
#     weights_only=False

# )

print("===== Model Architecture =====")

print(checkpoint["architecture"])

print("\n===== Hyperparameters =====")

for key, value in checkpoint["hyperparameters"].items():

    print(f"{key}: {value}")

print("\n===== Model State Dict =====")

state_dict = checkpoint["model_state_dict"]

print(f"Number of layers: {len(state_dict)}")

for name, param in state_dict.items():

    print(

        f"{name}: {tuple(param.shape)}"

    )