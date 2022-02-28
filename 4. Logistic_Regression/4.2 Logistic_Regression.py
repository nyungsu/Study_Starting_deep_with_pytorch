import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader





# class Dataset(torch.utils.data.Dataset):
#     def __init__(self) -> None:
#         super().__init__()
#         self.x_data = torch.FloatTensor([[1,2],
#                                          [2,3],
#                                          [3,1],
#                                          [4,3],
#                                          [5,3],
#                                          [6,2]])
#         self.y_data = torch.FloatTensor([[0],
#                                          [0],
#                                          [0],
#                                          [1],
#                                          [1],
#                                          [1]])
         
#     def __len__(self):
#         return len(self.x_data)
    
#     def __getitem__(self,idx):
#         x = torch.FloatTensor(self.x_data[idx])
#         y = torch.FloatTensor(self.y_data[idx])
        
#         return x,y
    
# dataset = Dataset()
# dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=2)

x_train = torch.FloatTensor([[1,2],
                             [2,3],
                             [3,1],
                             [4,3],
                             [5,3],
                             [6,2]])

y_train = torch.FloatTensor([[0],
                             [0],
                             [0],
                             [1],
                             [1],
                             [1]])

w = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w,b], lr = 1)
nb_epochs = 2000


for epoch in range(nb_epochs+1):
    hypothesis = torch.sigmoid(x_train.matmul(w)+b)

    # loss = -(y_train[0] * torch.log(hypothesis[0]) 
    # + (1-y_train[0])* torch.log(1-hypothesis[0]))

    # print(loss)

    # losses = -(y_train * torch.log(hypothesis) +
    #            (1-y_train)* torch.log(1-hypothesis))
    # cost = losses.mean()
    # print(cost)

    cost = F.binary_cross_entropy(hypothesis, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    
    if epoch % 100 ==0:
        print(f'epoch : {epoch}/{nb_epochs}')
        print(f'cost : {cost:.2f}')
    
print(f'model : {w},{b}')

new_value = torch.FloatTensor([1,3])
result = new_value.matmul(w)+b

print(f'new value in mode : {result}')

hypo_result = torch.sigmoid(result)
print(hypo_result)

print(f'result in sigmoid : {hypo_result}')