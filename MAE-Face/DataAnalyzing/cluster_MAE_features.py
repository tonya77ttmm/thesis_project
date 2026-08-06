# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import os

# # load data
# feat_file = "../Features/Numpy_features/Train_v2_feats_cc.npy"
# label_file = "../Features/Numpy_features/Train_v2_labels_cc.npy"
# train_count = os.path.getsize(feat_file) // (768 * 4)
# #  self.features = np.memmap(feat_path, dtype='float32', mode='r', 
# #                                   shape=(num_samples, feat_dim))
# #         self.labels = np.memmap(label_path, dtype='int64', mode='r', 
# #                                 shape=(num_samples,))

# features = np.memmap(feat_file, dtype='float32', mode='r', shape=(train_count, 768))
# labels = np.memmap(label_file, dtype='int64', mode='r', shape=(train_count,))

# features = np.array(features)
# labels = np.array(labels)
# print(features.shape)
# print(features.dtype)

# print(features[:2])
# print("Running t-SNE... this may take a while")

# # optional: sample (speed up)
# idx = np.random.choice(len(features), 100, replace=False)
# features_sample = features[idx]
# labels_sample = labels[idx]

# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# X_2d = tsne.fit_transform(features_sample)

# plt.figure(figsize=(8,6))

# # class 0
# plt.scatter(X_2d[labels_sample==0, 0],
#             X_2d[labels_sample==0, 1],
#             s=5, alpha=0.5, label="Non-confusion")

# # class 1
# plt.scatter(X_2d[labels_sample==1, 0],
#             X_2d[labels_sample==1, 1],
#             s=5, alpha=0.5, label="Confusion")

# plt.legend()
# plt.title("t-SNE of ViT features")
# plt.show()
import numpy as np
import matplotlib
# Force a non-interactive backend to prevent GUI hanging
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import time

# analyze features extracted from  a MAE (Masked Autoencoder) model to see if the model has  learned to distinguish between  Confusion vs. Non-confusion.

# Load data
# feat_file = "../Features/Numpy_features/Train_feats_cc.npy"
# label_file = "../Features/Numpy_features/Train_labels_cc.npy"
feat_file = "../ExperimentWithDevmo/data/features/devmo_train_feats.npy"

label_file = "../ExperimentWithDevmo/data/features/devmo_train_labels.npy"

# Use the hardcoded shape since we know it's 5931 from your log
num_samples = os.path.getsize(feat_file) // (768 * 4)
feat_dim = 768

print("Loading memory maps...")
features_mmap = np.memmap(feat_file, dtype='float32', mode='r', shape=(num_samples, feat_dim))
labels_mmap = np.memmap(label_file, dtype='int64', mode='r', shape=(num_samples,))

# Pull into actual RAM memory
features = np.array(features_mmap)
labels = np.array(labels_mmap)

print(f"Data shape: {features.shape}")

# Optional: Since 5931 is small, let's use 1000 samples instead of just 100 
# to get a much better, more meaningful visualization.
sample_size = len(features)
print(f"Sampling {sample_size} points...")
idx = np.random.choice(len(features), sample_size, replace=False)
features_sample = features[idx]
labels_sample = labels[idx]

# Run t-SNE with a timer
print("Running t-SNE... (this should take less than 10 seconds)")
start_time = time.time()

# added n_jobs=-1 to use all CPU cores and speed it up further
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
X_2d = tsne.fit_transform(features_sample)

print(f"t-SNE finished in {time.time() - start_time:.2f} seconds!")

# Plotting
print("Generating plot...")
plt.figure(figsize=(8, 6))

# Class 0
plt.scatter(X_2d[labels_sample == 0, 0],
            X_2d[labels_sample == 0, 1],
            s=15, alpha=0.6, label="Non-confusion", c='blue')

# Class 1
plt.scatter(X_2d[labels_sample == 1, 0],
            X_2d[labels_sample == 1, 1],
            s=15, alpha=0.6, label="Confusion", c='red')

plt.legend()
plt.title("t-SNE of ViT features")

# Save instead of show to prevent GUI deadlock
output_image = "devmo_train_feats_cluster.png"
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"Success! Plot saved as '{output_image}' in your current folder.")