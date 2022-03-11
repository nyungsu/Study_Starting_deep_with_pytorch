import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset,DataLoader

import random
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
nb_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root='data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='data/',
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)

x_train_origin = mnist_train.data
y_train_origin = mnist_train.train_labels

x_test_origin = mnist_test.data
y_test_origin = mnist_test.test_labels

loader_train = DataLoader(dataset=mnist_train,
                          shuffle=True,
                          batch_size=batch_size,
                          drop_last=True)

loader_test = DataLoader(dataset=mnist_test,
                         shuffle=True,
                         batch_size=batch_size,
                         drop_last=True)


model = nn.Sequential(
    nn.Linear(784,100),
    nn.ReLU(),
    
    nn.Linear(100,100),
    nn.ReLU(),
    
    nn.Linear(100,10)
)

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    
    # x_train : [100,1,28,28] : batch size 갯수 만큼의 1x28x28의 텐서
    # y_train : [100] : batch size 갯수 만큼의 레이블
    # 60000개의 데이터, batch_size : 100 => enumerate : 1~600
    for batch_idx,sample in enumerate(loader_train):
        x_train, y_train = sample
                
        x_train = x_train.view(-1,784)      # [100,784]
        y_train = y_train                   # [100]
        
        hypothesis = model(x_train)
        cost = F.cross_entropy(hypothesis, y_train)
            
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        
    print(f'epcoh{epoch} : 완료')

def test():
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for x_test,y_test in loader_test:
            x_test = x_test.view(-1,784)
            
            outputs = model(x_test)
            
            _,prediction = torch.max(outputs,1)
            # torch.max(input, dim, keepdim)
            # input : 입력텐서
            # dim : 시행 축
            # out (max값, max_index)
            
            correct += prediction.eq(y_test.view_as(prediction)).sum()
            
        
    data_num = len(loader_test.dataset)
    print(f'테스트 데이트에서 예측 정확도 : {correct}/{data_num}')

for epoch in range(1):
    train(epoch)
    
test()

