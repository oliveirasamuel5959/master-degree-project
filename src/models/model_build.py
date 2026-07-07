import torch
from torch import nn

class BaseModel(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    
    # takes in 2 features and upscales to 5 features 
    self.layer_1 = nn.Linear(in_features=2, out_features=5)
    
    # takes in 5 features from previous layer and outputs a single feature (same shape as y)
    self.layer_2 = nn.Linear(in_features=5, out_features=1)
  
  # 3. Define a forward() method that outlines the forward pass
  def forward(self, x):
    return self.layer_2(self.layer_1(x)) # x -> layer_1 ->  layer_2 -> output 