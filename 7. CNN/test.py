import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader

test_tensor = torch.Tensor(1,1,28,28)
print(test_tensor.shape)

layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
)

out = layer1(test_tensor)
print(out.shape)

layer2 = nn.Sequential(
    nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
)

out = layer2(out)
print(out.shape)

out = out.view(1,-1)
print(out.shape)