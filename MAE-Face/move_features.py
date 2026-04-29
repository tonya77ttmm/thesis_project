#move all features from Features to training_features
import os
import shutil

features_dir="./Features/"
training_dir="./Features/Training_features/"

os.makedirs(training_dir,exist_ok=True)

for filename in os.listdir(features_dir):
    filepath=os.path.join(features_dir,filename)
    if os.path.isfile(filepath):
        shutil.move(filepath, os.path.join(training_dir))
        
