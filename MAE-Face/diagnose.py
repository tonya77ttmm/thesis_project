import torch
import os
import zipfile

def analyze_pt_file(filepath):
    print(f"--- Analyzing: {os.path.basename(filepath)} ---")
    if os.path.exists(filepath):
        print("✅ File found!")
    else:
        print("❌ File NOT found. Check if the folder names match exactly.")
        return

    # 1. Physical File Size
    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"Physical File Size: {file_size_kb:.2f} KB")
    
    # 2. Check if it's a Zip Archive (Modern PyTorch format)
    if zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, 'r') as z:
            print("Internal Files (Zip):")
            for member in z.infolist():
                print(f"  - {member.filename}: {member.file_size / 1024:.2f} KB")
    
    # 3. Load and check actual data size
    data = torch.load(filepath, map_location='cpu')
    feat = data['features']
    # Size in bytes = number of elements * bytes per element (float32 = 4 bytes)
    actual_bytes = feat.nelement() * feat.element_size()
    print(f"Actual Tensor Data Size: {actual_bytes / 1024:.2f} KB")
    print("\n")

# Use this on one of your 1-feature files and one 64-feature file
file_1="./Features/Training_single_frame_features/frame_1142465.pt"
file_64="./Features/Training_features/features_batch_0.pt"

analyze_pt_file(file_1)
analyze_pt_file(file_64)