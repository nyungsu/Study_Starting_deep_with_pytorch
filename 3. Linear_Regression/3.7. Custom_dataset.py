import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self) -> None: # 데이터의 전처리를 해주는 곳
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
    
    def __len__(self): # len(dataset)을 했을 때 데이터셋의 크기를 리턴
        return len(self.x_data)
    
    # dataset[i]을 했을 때
    # i번째 샘플을 가져오도록 하는
    # 인덱싱을 위한 get_item
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y
    
class MultivariableLinearRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3,1)
        
    def forward(self,x):
        return self.linear(x)

dataset = CustomDataset()
dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=2)


model = MultivariableLinearRegression()
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20

for epoch in range(nb_epochs+1):
    for batch_idx, sample in enumerate(dataloader):
        x_train,y_train = sample
        
        # h(x)
        hypothesis = model(x_train)
        
        # cost
        cost = F.mse_loss(hypothesis,y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        
        print(f'epoch : {epoch}/{nb_epochs}')
        print(f'batch idx : {batch_idx+1}/{len(dataloader)}')
        print(f'cost : {cost:.2f}')
        print()
        
print(list(model.parameters()))

new_values = torch.FloatTensor([70,80,90])

print(f'new prediction : {model.forward(new_values)}')

