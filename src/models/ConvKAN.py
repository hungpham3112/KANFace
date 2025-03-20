import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# sys.path.append('./kan_convolutional')
from ..layers.KANLinear import KANLinear

class ConvKAN(nn.Module):
    def __init__(self,  neuron_fun="mean", num_features=128):
        super(ConvKAN, self).__init__()
        # First convolutional layer updated to take 3 input channels
        self.conv1 = nn.Conv2d(3, 5, kernel_size=3, bias=True, padding=1)  # Output: (5, 110, 110)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, bias=True, padding=1)  # Output: (5, 108, 108)
        
        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # Reduces spatial dimensions by half
        
        # Flatten layer
        self.flatten = nn.Flatten()

        # Adjusting the input size for KANLinear based on reduced dimensions
        # (112x112 -> 56x56 after 1 maxpool -> 28x28 after 2 maxpool)
        self.kan1 = KANLinear(
            # Input size will be determined after the first forward pass
            out_dim=num_features,
        )
        # self.linear1 = nn.Linear(5 * 26 *26, 512, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.maxpool(x)        
        x = F.relu(self.conv2(x))  
        x = self.maxpool(x)        
        x = self.flatten(x)
        if self.kan1.in_dim is None:
            self.kan1.in_dim = x.size(1)
            self.kan1 = self.kan1.to(x.device)    
        x = self.kan1(x)
        
        # x = self.linear1(x)    
        return x