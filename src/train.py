"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter

from src.training.train_step import train_step
from src.training.test_step import test_step

def train(
  model: torch.nn.Module, 
  train_dataloader: torch.utils.data.DataLoader, 
  test_dataloader: torch.utils.data.DataLoader, 
  optimizer: torch.optim.Optimizer,
  loss_fn: torch.nn.Module,
  epochs: int,
  device: torch.device,
  writer: torch.utils.tensorboard.writer.SummaryWriter=None,
  scheduler: torch.optim.lr_scheduler._LRScheduler=None) -> Dict[str, List]:
  
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    results = {
      "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)
    
    # Loop through training and testing steps for a number of epochs
    # At the beginning of training (before loop)
    previous_lr = optimizer.param_groups[0]['lr']
    print(f"[Epoch 0] Initial Learning Rate: {previous_lr:.6f}")
    
    for epoch in tqdm(range(epochs)):
      
      train_loss, train_acc = train_step(
        model=model,
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
      )
      
      test_loss, test_acc = test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device
      )
      
      # Print out training
      print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.4f} | "
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_acc:.4f}"
      )
      
      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)
      
      ### New: Experiment TRACKING ###
      if writer:
        # Add results to SummaryWriter
        writer.add_scalars(
          main_tag="Loss", 
          tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
          global_step=epoch
        )
        
        writer.add_scalars(
          main_tag="Accuracy", 
          tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc}, 
          global_step=epoch
        )

        # Close the writer
        writer.close()
        
      else:
        pass
      
      if scheduler:
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']

        if current_lr != previous_lr:
          print(f"[Epoch {epoch+1}] Learning rate changed to: {current_lr:.6f}")
          previous_lr = current_lr
      
    # Return the filled results at the end of the epochs
    return results