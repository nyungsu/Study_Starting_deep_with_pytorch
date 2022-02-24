import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 


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


model = nn.Linear(3,1)          # 피쳐 갯수가 dim 인듯 ???????
'''
nn.linear(input_dim, output_dim)
input 차원과 output 차원을 입력해주면 된다.
hypothesis 대신 쓰는 것

model.marameters()를 list로 뽑아보면
두 개의 인자가 나오는데
첫 번째는 w
두 번째가 b 이다.
'''
'''
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3,1)
        
    def forward(self,x):
        return self.linear(x)
        
이 클래스로 model 대신 할 수 있음
'''


optimizer = optim.SGD(model.parameters(),lr=1e-5)

nb_epochs = 20

for epoch in range(nb_epochs+1):
    hypothesis = model(x_train)

    cost = F.mse_loss(hypothesis,y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    print(f'epoch :{epoch}/{nb_epochs}')
    print(f'cost : {cost:.2f}')
    


print(list(model.parameters()))
print()

print(model(x_train))