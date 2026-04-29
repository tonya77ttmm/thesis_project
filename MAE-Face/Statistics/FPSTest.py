#import lib
#process video, cal time, load model
import torch
import time
import cv2
import models_vit

#settings for video, model load path
video_path="./Data/P32.mp4"
ckpt_path="./models/mae_face_pretrain_vit_base.pth"
model_name = 'vit_base_patch16'
num_heads=12
device='cuda'

#load video
cap=cv2.VideoCapture(video_path)
frames=[]
img_size=224
while True:
    ret, frame=cap.read()
    if not ret:
        break
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame=cv2.resize(frame,(img_size,img_size))
    frame=torch.tensor(frame).permute(2,0,1).float()/255.0
    frames.append(frame)
cap.release()
frames=torch.stack(frames)
print(f"Total frames loaded:{frames.shape[0]}")

#load the model???parameters
model = getattr(models_vit, model_name)(
    global_pool=True,
    num_classes=num_heads,
    drop_path_rate=0.1,
    img_size=224,
)
checkpoint = torch.load(ckpt_path, map_location='cpu',weights_only=False)
checkpoint_model = checkpoint['model']
msg = model.load_state_dict(checkpoint_model, strict=False)
model.to(device)
model.eval()

#test
with torch.no_grad():
    for batch_size in range(1, 65):
        total_frames=0
        start_time=time.time()
        for i in range(0, len(frames), batch_size):
            batch=frames[i:i+batch_size].to(device)
            _=model(batch, ret_feature=True)
            total_frames+=batch.shape[0]
        elapsed=time.time()-start_time
        fps=total_frames/elapsed
        print(f"base model batch_size:{batch_size} FPS:{fps:.2f}")