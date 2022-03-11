import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

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

w = torch.zeros((4,3), requires_grad=True)
b = torch.zeros(1,requires_grad=True)

optimizer = optim.SGD([w,b], lr=0.1)

nb_epochs = 1000

y_one_hot = torch.zeros(8,3)
y_one_hot.scatter_(1,y_train.unsqueeze_(1),1)

for epoch in range(nb_epochs+1):
    
    hypothesis = F.softmax(x_train.matmul(w)+b, dim =1)
    
    cost = (y_one_hot* - torch.log(hypothesis)).sum(dim=1).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'epcho : {epoch}, cost : {cost:.2f}')