import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(1,1)
'''
nn.linear(input_dim, output_dim)
입력되는 x의 차원과 y의 차원을 입력해주면 된다.

model.parameters()를 list로 뽑아보면
두 개의 인자가 나오는데
첫 번째는 w
두 번째가 b 이다.
'''

optimizer= optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 1000

for epoch in range(nb_epochs+1):
    
    # hypothesis = x_train.matmul(w) + b 대신
    hypothesis = model(x_train)
    
    # cost = torch.mean((y_train-hypothesis)**2)
    cost = F.mse_loss(hypothesis, y_train)

    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


print(list(model.parameters()))     # 첫 번째가 w 두 번째가 b
# [Parameter containing:
# tensor([[1.9930]], requires_grad=True), Parameter containing:
# tensor([0.0159], requires_grad=True)]