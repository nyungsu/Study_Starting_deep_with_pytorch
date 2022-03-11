from random import shuffle
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.x_data = torch.FloatTensor([[1,2],
                                         [2,3],
                                         [3,1],
                                         [4,3],
                                         [5,3],
                                         [6,2]])
        self.y_data = torch.FloatTensor([[0],
                                         [0],
                                         [0],
                                         [1],
                                         [1],
                                         [1]])
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self,idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y
    
class MultivariableLinearRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        return self.sigmoid(self.linear(x))
    
dataset = CustomDataset()
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=2)

model = MultivariableLinearRegression()
optimizer = optim.SGD(model.parameters(), lr = 1)

nb_epoch = 1000

for epoch in range(nb_epoch+1):
    for batch_idx, sample in enumerate(dataloader):
        x_train, y_train = sample
        
        hypothesis = model(x_train)
        
        cost = F.binary_cross_entropy(hypothesis, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if epoch %200 ==0:
            pred = hypothesis >= torch.FloatTensor([0.5])
            correct_pred = pred.float() == y_train
            accurcy = correct_pred.sum().item() / len(correct_pred)