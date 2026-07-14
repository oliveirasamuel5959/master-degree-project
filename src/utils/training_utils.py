import torch
import pandas as pd
import os
import matplotlib.pyplot as plt

def training_validation(model, val_loader, loss_fn, device):
  model.eval()
  val_loss = 0.0
  correct = 0
  
  with torch.no_grad():
    for videos, labels in val_loader:
      videos, labels = videos.to(device), labels.to(device)
      outputs = model(videos)
      loss = loss_fn(outputs, labels)
      val_loss += loss.item()
      
      _, predicted = torch.max(outputs, 1)
      correct += (predicted == labels).sum().item()
      
  avg_val_loss = val_loss / len(val_loader)
  accuracy = correct / len(val_loader.dataset)

  return avg_val_loss, accuracy


# ----------------------------------
# Plot and save history and plots
# ----------------------------------
def save_history_and_plots(history, output_dir, prefix):
  """

  Args:
      history (_type_): objeto retornado pelo model.fit
      output_dir (_type_): save directory
      prefix (_type_): ex.: "train" ou "val"
  """

  os.makedirs(output_dir, exist_ok=True)

  # ---------------------
  # History to Dataframe
  # ---------------------
  hist_df = pd.DataFrame(history)
  hist_df["epoch"] = hist_df.index + 1

  csv_path = os.path.join(output_dir, f"{prefix}_.csv")
  hist_df.to_csv(csv_path, index=False)

  # ---------------------
  # Accuracy plot
  # ---------------------
  fig_acc, ax_acc = plt.subplots(figsize=(7, 5))

  ax_acc.plot(hist_df["epoch"], hist_df["accuracy"], label="Train Accuracy")

  if "val_accuracy" in hist_df:
    ax_acc.plot(hist_df["epoch"], hist_df["val_accuracy"], label="Validation Accuracy")

  ax_acc.set_xlabel("Epoch")
  ax_acc.set_ylabel("Accuracy")
  ax_acc.set_title("Training and Validation Accuracy")
  ax_acc.legend()
  ax_acc.grid(True)

  acc_plot_path = os.path.join(output_dir, f"acc_plot.png")
  fig_acc.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close(fig_acc)

  # ---------------------
  # Loss plot
  # ---------------------
  fig_loss, ax_loss = plt.subplots(figsize=(7, 5))

  ax_loss.plot(hist_df["epoch"], hist_df["loss"], label="Train Loss")

  if "val_loss" in hist_df:
    ax_loss.plot(hist_df["epoch"], hist_df["val_loss"], label="Validation Loss")

  ax_loss.set_xlabel("Epoch")
  ax_loss.set_ylabel("Loss")
  ax_loss.set_title("Training and Validation Loss")
  ax_loss.legend()
  ax_loss.grid(True)

  loss_plot_path = os.path.join(output_dir, f"loss_plot.png")
  fig_loss.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close(fig_loss)

  print()
  print(f"[OK] History saved to {csv_path}")
  print(f"[OK] Accuracy plot saved to {acc_plot_path}")
  print(f"[OK] Loss plot saved to {loss_plot_path}")