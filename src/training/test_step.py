"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter

def test_step(
  model: torch.nn.Module, 
  dataloader: torch.utils.data.DataLoader, 
  loss_fn: torch.nn.Module,
  device: torch.device) -> Tuple[float, float]:
  
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    total_loss = 0.0
    correct = 0
    total = 0

    # Turn on inference context manager
    with torch.inference_mode():
      
      # Loop through DataLoader batches
      for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation", leave=False):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        logits = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(logits, y)
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size

        # Calculate and accumulate accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item() 
        total += batch_size
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = total_loss / total
    test_acc = correct / total
    
    return test_loss, test_acc