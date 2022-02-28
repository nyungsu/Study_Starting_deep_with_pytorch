import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as opitm

x_train = torch.FloatTensor([[73,80,75],
                            [93,88,93],
                            [89,91,80],
                            [96,98,100],
                            [73,66,70]])

y_train = torch.FloatTensor([[152],
                            [185],
                            [180],
                            [196],
                            [142]])

class MultivariabelLinearRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3,1)
        
    def forward(self,x):
        return self.linear(x)
    
    
model = MultivariabelLinearRegression()
optimizer = opitm.SGD(model.parameters(), lr=1e-5)
nb_epochs = 20

for epoch in range(nb_epochs+1):
    # h(x)
    hypothesis = model(x_train)
    
    # cost
    cost = F.mse_loss(hypothesis,y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    print(f'epoch : {epoch}/{nb_epochs}')
    print(f'cost : {cost:.2f}')
    
print(list(model.parameters()))
print(model.forward(x_train))
    