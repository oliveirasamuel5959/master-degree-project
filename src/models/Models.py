import torch
from torch import nn

class Conv3DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, pool_kernel=(1, 2, 2)):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm3d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm3d(out_channels),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel)
    )

  def forward(self, x):
    return self.block(x)
  
  
class Video3DCNN(nn.Module):
  def __init__(self, num_classes, in_channels=3):
    super().__init__()

    self.features = nn.Sequential(
      # Input: (B, 3, 16, 224, 224)
      Conv3DBlock(in_channels, 16, pool_kernel=(1, 2, 2)),   # -> (B,32,16,112,112)
      Conv3DBlock(16, 32,  pool_kernel=(2, 2, 2)),           # -> (B,64,8,56,56)
      Conv3DBlock(32, 64, pool_kernel=(2, 2, 2)),           # -> (B,128,4,28,28)
      Conv3DBlock(64, 128, pool_kernel=(2, 2, 2)),          # -> (B,256,2,14,14)
    )

    # AdaptiveAvgPool3d makes the classifier independent of exact T/H/W
    self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Dropout(0.5),
      nn.Linear(128, 64),
      nn.ReLU(inplace=True),
      nn.Dropout(0.3),
      nn.Linear(64, num_classes)
    )

  def forward(self, x):
    # x: (B, C, T, H, W)
    x = self.features(x)
    x = self.global_pool(x)
    x = self.classifier(x)
    
    return x