import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

x_train = torch.FloatTensor([[1],
                             [2],
                             [3]])

y_train = torch.FloatTensor([[2],
                             [4],
                             [6]])

class LinearRegreesionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1,1)
        
    def forward(self,x):
        return self.linear(x)


model = LinearRegreesionModel()

optimizer = optim.SGD(model.parameters(), lr = 0.01)
                      
nb_epochs = 2000

for epoch in range(nb_epochs+1):
    
    # h(x)
    hypothesis = model(x_train)
    
    # cost
    cost = F.mse_loss(hypothesis,y_train)
    
    # 학습
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 ==0 :
        print(f'epoch : {epoch}/{nb_epochs}')
        print(f'cost : {cost:.2f}')
    
print(list(model.parameters()))