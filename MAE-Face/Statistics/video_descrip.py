import os
import statistics
from moviepy import VideoFileClip
from pathlib import Path



def get_video_durations(video_dir):
  

# 2. List all video files with .avi or .mp4
    video_files = [
    str(p) 
    for p in Path(video_dir).rglob("*") 
    if p.suffix.lower() in [".avi", ".mp4"]
]

    print(f"Total videos found: {len(video_files)}")

# 3. Get durations in seconds
    durations = []
    for video_path in video_files:
        try:
            clip = VideoFileClip(video_path)
            durations.append(clip.duration)
            clip.close()
        except Exception as e:
            print(f"Error reading {video_path}: {e}")

    if durations:
        avg_duration = sum(durations) / len(durations)
        median_duration = statistics.median(durations)
        try:
            mode_duration = statistics.mode(durations)
        except statistics.StatisticsError:
            mode_duration = "No unique mode"
        std_duration = statistics.stdev(durations) if len(durations) > 1 else 0

        print(f"Statistics for video durations:{video_dir}")
        print(f"Average duration: {avg_duration:.2f} seconds")
        print(f"Median duration: {median_duration:.2f} seconds")
        print(f"Mode duration: {mode_duration}")
        print(f"Standard deviation: {std_duration:.2f} seconds")
    else:
        print("No video durations could be calculated.")

if __name__ == "__main__":
    daisee_train_video_dir = "../confusion_dataset/DAiSEE/DataSet/Train"
    daisee_test_video_dir = "../confusion_dataset/DAiSEE/DataSet/Test"
    daisee_val_video_dir = "../confusion_dataset/DAiSEE/DataSet/Validation"
    daisee_video_dir="../confusion_dataset/DAiSEE/DataSet"
    devmo_video_dir="../confusion_dataset/Devmo"
    mfc_video_dir="../confusion_dataset/MFC-dataset"
    get_video_durations(daisee_train_video_dir)
    get_video_durations(daisee_test_video_dir)  
    get_video_durations(daisee_val_video_dir)
    get_video_durations(daisee_video_dir)
    get_video_durations(devmo_video_dir)
    get_video_durations(mfc_video_dir)
