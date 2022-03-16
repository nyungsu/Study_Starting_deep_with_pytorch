import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

# conda에서 python -m visdom.server를 실행시켜서 서버를 열어줘야 함
import visdom
vis = visdom.Visdom()       

# text를 띄워줌
vis.text('hello world!',env='main')

# image 출력
a = torch.randn(3,200,200)
vis.image(a)

# MNIST 출력하기
MNIST_train = dsets.MNIST(root='data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

loader_train = DataLoader(dataset=MNIST_train,
                          shuffle=True,
                          batch_size=32,
                          drop_last=True)

for num,value in enumerate(loader_train):
    value = value[0]
    vis.images(value)
    break

vis.close(env='main')

# Line Plot
x_data = torch.Tensor([1,2,3,4,5])
y_data = torch.randn(5)
plt = vis.line(X=x_data, Y=y_data)

# line update
x_append = torch.Tensor([6])
y_append = torch.randn(1)
plt = vis.line(X=x_append, Y=y_append, win=plt, update='append')
# 기존의 plt라는 window에 append 라는 방식으로 업데이트 하겠다.

# close


# muliple line on single windows
x_data = torch.Tensor([1,2,3,4,5])              # (, 5)
x_data = x_data.unsqueeze_(0).view(5,-1)        # (1, 5) -> (5, 1)
x_data = torch.cat((x_data, x_data), dim=1)     # (5, 2)

y_data = torch.randn(5,2)                       # (5, 2)

plt2 = vis.line(X=x_data, Y=y_data,  # 이렇게 세로로 늘여져야 두 줄
                opts= dict(title='test', legend=['파랑','주황']))  


# function update line
def loss_tracker(loss_plot, loss_value, num):
    vis.line(X=num,
             Y=loss_value,
             win=loss_plot,
             update='append',
             opts=dict(title='title'))
    
