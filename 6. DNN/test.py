from sympy import root
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset,DataLoader

BATCH_SIZE = 100

dataset_train = dsets.MNIST('data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

dataset_test = dsets.MNIST(root='data/',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)

loader_train = DataLoader(dataset=dataset_train,
                          shuffle=True,
                          batch_size=BATCH_SIZE,
                          drop_last=True)

loader_test = DataLoader(dataset=dataset_test,
                         shuffle=True,
                         batch_size=BATCH_SIZE,
                         drop_last=True)

x_train_origin = dataset_train.data
y_train_origin = dataset_train.train_labels

x_test_origin = dataset_test.data
y_test_origin = dataset_test.test_labels

model = nn.Sequential(
    nn.Linear(28*28,100),
    nn.ReLU(),
    
    nn.Linear(100,100),
    nn.ReLU(),
    
    nn.Linear(100,10)
)

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    for batch_idx, sample in enumerate(loader_train):
        x_train, y_train = sample
        x_train = x_train.view(-1,28*28)
        y_train = y_train
        
        hypothesis = model(x_train)
        
        cost = F.cross_entropy(hypothesis, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
def test():
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for x_test, y_test in dataset_test:
            x_test = x_test.view(-1,784),
            y_test = y_test
            
            outputs = model(x_test)
            
            _, pred = torch.max(input=outputs,dim=1)
            
            correct += pred.eq(y_test,1).sum()