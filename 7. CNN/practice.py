import enum
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 100
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 15

mnist_train = dsets.MNIST(root='data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

x_train_original = mnist_train.data
y_train_original = mnist_train.train_labels

x_test_original = mnist_test.data
y_test_original = mnist_test.test_labels

loader_train = DataLoader(dataset=mnist_train,
                          shuffle=True,
                          batch_size=BATCH_SIZE,
                          drop_last=True)

loader_test = DataLoader(dataset=mnist_test,
                         shuffle=True,
                         batch_size=BATCH_SIZE,
                         drop_last=True)

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.FC = nn.Linear(7*7*64,10)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        
        out = out.view(out.size(0),-1)
        print(out.shape)
        out = self.FC(out)
        
        return out
    
model = CNN()
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
total_batch = len(loader_train)

def train(epoch):
    for epoch in range(TRAINING_EPOCHS+1):
        avg_cost = 0
        for batch_idx, sample in enumerate(loader_train):
            x_train, y_train = sample
            
            hypothesis = model(x_train)
            
            cost = F.cross_entropy(hypothesis,y_train)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            if batch_idx % 100 ==0:
                print(f'epoch : {epoch}/{TRAINING_EPOCHS}')
                print(f'batch idx : {batch_idx+1}/{total_batch}')
                print(f'cost : {cost:.2f}')
                
train(2)