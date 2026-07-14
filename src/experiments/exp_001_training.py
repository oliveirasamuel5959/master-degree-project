import os
import time
import sys
import torch

from pathlib import Path
from tqdm import tqdm

from torch import nn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.utils import make_grid

from src.data.loaders import read_train_test_data_txt
from src.data.dataset_preprocessing import VideoDataset
from src.data.dataset_preprocessing import video_preprocessing
from src.models.Models import Video3DCNN
from src.utils.training_utils import training_validation
from src.utils.training_utils import save_history_and_plots

from src.core.logger import get_logger, log_task

logger = get_logger(__name__)

# ==============
# Training Loop
# ==============
def process_train(
  dataset_root_dir,
  dataset_split_dir,
  output_root,
  model_name="CNN3D",
  num_classes=14,
  window_frames=16,
  learning_rate=0.001,
  epochs=10,
  batch_size=16,
  my_seed=42,
  device='cpu'
):
  
  # ================================================
  # Read the files train, validation and test splits
  # ================================================
  logger.info(f"Loading dataset from path {dataset_split_dir}...")
  
  train_files = read_train_test_data_txt(dataset_split_dir, "Anomaly_Train")
  val_files = read_train_test_data_txt(dataset_split_dir, "Anomaly_Test")
  # test_files = read_train_test_data_txt(dataset_split_dir.parent / "Action_Recognition_Splits", "test_002")
  
  # ===========================================================
  # Create train, val and test Dataset and apply transformation
  # ===========================================================
  logger.info(f"Creating video dataset from path {dataset_root_dir}")
  
  video_transforms = video_preprocessing(frame_resize=(112, 112))
  
  train_dataset = VideoDataset(
    root_dir=dataset_root_dir, 
    data_files=train_files, 
    num_frames=window_frames,
    transform=video_transforms
  )
  
  val_dataset = VideoDataset(
    root_dir=dataset_root_dir, 
    data_files=val_files, 
    num_frames=window_frames,
    transform=video_transforms
  )
  
  # test_dataset = VideoDataset(
  #   root_dir=dataset_root_dir, 
  #   data_files=test_files, 
  #   num_frames=window_frames,
  #   transform=video_transforms
  # )
  
  # ===============================================
  # Create Dataloader for model training in batches
  # ===============================================
  logger.info("Creating dataloader for training...")
  
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
  # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
  
  # =========================
  # Model Build
  # =========================
  if model_name == 'CNN3D':
    model = Video3DCNN(num_classes)
  
  # ==============
  # Train Loop
  # ==============
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
  model = model.to(device)
  
  # Store loss values for plotting
  history = {
    "epoch": [],
    "accuracy": [],
    "val_accuracy": [],
    "loss": [],
    "val_loss": []
  }
  
  best_val_loss = float('inf')
  
  logger.info("Model training...")
  time_start = time.time()
  
  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    
    for videos, labels in loop:
      videos, labels = videos.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(videos)
      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      num_batches += 1
      
      # Update progress bar with running average loss
      loop.set_postfix(loss=running_loss / num_batches)
      
    # Validation step
    val_loss, val_accuracy = training_validation(model, val_loader, loss_fn, device)
    
    # Checkpointing
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {running_loss/len(train_loader):.4f}, "
          f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
    history["epoch"].append(epoch)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)
    
  time_end = time.time()
  training_time = time_end - time_start
  logger.info("Training time: %.2f seconds", training_time)
    
  save_history_and_plots(history, output_root, "history")
    
    
def main():
  ROOT_DIR = Path.cwd()
  OUTPUT_DIR = ROOT_DIR / "outputs" / "experiments" / Path(__file__).stem
  
  if not OUTPUT_DIR.exists():
    print(f"Output root directory '{OUTPUT_DIR}' does not exist. Creating it.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  
  TRAIN_TEST_DATA = ROOT_DIR / "data" / "UCF_Crime" / "raw"
  TRAIN_TEST_SPLIT_FILES = ROOT_DIR / "data" / "UCF_Crime" / "processed" / "Anomaly_Detection_splits"

  # ====================
  # Run Training Process
  # ====================
  device = "cuda" if torch.cuda.is_available() else "cpu"
  
  process_train(
    dataset_root_dir=TRAIN_TEST_DATA,
    dataset_split_dir=TRAIN_TEST_SPLIT_FILES,
    output_root=OUTPUT_DIR,
    model_name="CNN3D",
    num_classes=14,
    window_frames=16,
    learning_rate=0.001,
    epochs=10,
    batch_size=16,
    my_seed=42,
    device=device
    
  )
  
if __name__ == "__main__":
  main()