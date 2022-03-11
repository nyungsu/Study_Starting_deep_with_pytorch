from sympy import root
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import visdom
# conda에서 python -m visdom.server를 실행시켜서 서버를 열어줘야 함
vis = visdom.Visdom()       

vis.text('hello world!',env='main')
# text를 띄워줌

a = torch.randn(3,200,200)
vis.image(a)
# image 출력

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