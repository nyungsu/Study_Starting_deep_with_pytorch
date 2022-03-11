import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

class SoftmaxRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4,3)
        # w행렬의 크기
        
    def forward(self,x):
        return self.linear(x)
    
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

model = SoftmaxRegression()
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000

for epoch in range(nb_epochs+1):
    
    hypothesis = model(x_train)
    
    cost = F.cross_entropy(hypothesis,y_train)
    # cost 함수에서 cross entropy를 쓰면 
    # hypothesis 함수에서 softmax 를 쓰면 안 됨
    # 근데 이렇게 되면 model.forward를 할 때는 어떻게 softmax 먹이지 ?
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'epcho : {epoch}, cost : {cost:.2f}')
        