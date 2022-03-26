import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

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

class MultivariableLinearRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3,1)
        
    def forward(self,x):
        return self.linear(x)
    

dataset = TensorDataset(x_train,y_train)
# 데이터 셋을 정의해준다

dataloader = DataLoader(dataset,shuffle=True,batch_size=2)
'''
# 배치사이즈는 2의 배수로,
# cpu와 gpu의 메모리가 2의 배수이므로
# 배치크기가 2의 배수일 때,
# 전송 효율을 높일 수 있음
'''

model = MultivariableLinearRegression()

optimizer = optim.SGD(model.parameters(), lr = 1e-5)

nb_epochs = 20

for epoch in range(nb_epochs+1):
    for batch_idx , samples in enumerate(dataloader):
        x_train, y_train = samples
        
        # h(x)
        hypothesis = model(x_train)
        
        # cost
        cost = F.mse_loss(hypothesis, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print(f'epoch : {epoch}/{nb_epochs}')
        print(f'batch : {batch_idx+1}/{len(dataloader)}')
        print(f'cost : {cost}')
        print()


new_var = torch.FloatTensor([73,80,75])

pred_y = model(new_var)


print(f'새로운 모델 {new_var}로 예측한 값 :{pred_y}')