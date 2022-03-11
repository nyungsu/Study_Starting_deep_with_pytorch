import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

BATCH_SIZE = 100
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 15

train_dataset = dsets.MNIST(root='data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='data/',
                          train=False,
                          transform=transforms.ToTensor(),
                          download=True)

loader_train = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          drop_last=True)

loader_test = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         drop_last=True)


# for i, (image, label) in enumerate(loader_train):
#     print(image.shape)
#     print(label.shape)
#     print()

# test = torch.FloatTensor(100,1,28,28)
# layer1 = nn.Sequential(
#     nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2)
# )
        
# layer2 = nn.Sequential(
#         nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2)
#         )
# fc = nn.Sequential(
#     nn.Linear(7*7*64,10)
# )

# out = layer1(test)
# out = layer2(out)
# out = out.view(100,-1)
# out = fc(out)
# print(out.shape)

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
        
        
        self.fc = nn.Linear(7*7*64,10)
    
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        
        return out

model = CNN()
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

def train(epoch):
    model.train()
    avg_cost = 0
    
    for epoch in range(TRAINING_EPOCHS+1):
        
        for i,(x_train, y_train) in enumerate(loader_train):
            print(x_train.shape)
            print(y_train.shape)
            print

