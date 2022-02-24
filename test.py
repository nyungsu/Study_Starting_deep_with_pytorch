import nbformat
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.x_data = torch.FloatTensor([[73,80,75],
                                         [93,88,93],
                                         [89,91,80],
                                         [96,98,100],
                                         [73,66,70]])
        self.y_data = torch.FloatTensor([[152],
                                         [185],
                                         [180],
                                         [196],
                                         [142]])
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self,idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        
        return x,y
        
class MultivariableLinearRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3,1)
        
    def forward(self,x):
        return self.linear(x)
    
dataset = Dataset()
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=2)

model = MultivariableLinearRegression()
optimizer = optim.SGD(model.parameters(), lr= 1e-5)

nb_epochs = 10

for epoch in range(nb_epochs+1):
    for batch_idx ,sample in enumerate(dataloader):
        x_train, y_train = sample
        
        hypothesis = model(x_train)
        
        cost = F.mse_loss(hypothesis, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print(f'epoch : {epoch}/{nb_epochs}')
        print(f'batch dix : {batch_idx+1}/{len(dataloader)}')
        print(f'cost : {cost:.2f}')
        print()
        
print(f"final w,b : {list(model.parameters())}")

new_value = torch.FloatTensor([70,80,90])
print(f'new value : {new_value} 일 때, prediction :{model.forward(new_value)}')        
        

