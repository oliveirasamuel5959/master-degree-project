import os
import cv2
from pathlib import Path

def dataset_frames_labels(dataset_path: Path, rand_num: int):
  frames = []
  labels = []
  
  classes_names = os.listdir(dataset_path)
  
  for class_name in classes_names:
    videos_path = dataset_path / class_name
    video_path = list(videos_path.glob("*.mp4"))[rand_num]
  
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
      print("Error: Could not open the video file.")
      return
      
    ret, frame = cap.read()
    
    if not ret:
      break
    
    frames.append(frame)
    labels.append(video_path.name)
    
  return frames, labels

