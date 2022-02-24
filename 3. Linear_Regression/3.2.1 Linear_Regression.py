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

print(x_train.shape)    # x_train (3,1)
print(y_train.shape)    # y_train (3,1)



# 모델의 w와 b를 초기화
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w,b], lr =0.1)


nb_epochs = 1000
for epoch in range(nb_epochs+1):
    # 가설 함수
    hypothesis = x_train *w +b
    # 비용 함수
    cost = torch.mean((hypothesis - y_train)**2)

    # GD
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 ==0:
        print(f'epoch :{epoch/nb_epochs*100}%, cost : {cost}') 
        print()

print(w,b)